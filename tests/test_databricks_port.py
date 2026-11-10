"""The parts of the Databricks port that don't need Spark: type mapping, paths and the reproduction checks."""
import pandas as pd
import pytest

from segmentation.databricks_session import _Frame, records, spark_ddl, volume_path
from segmentation.reproduce import compare_cleaning, compare_k, compare_schemes, compare_segments


def test_ddl_types_all_null_columns_from_their_dtype():
    df = pd.DataFrame({"A": pd.array([None, None], dtype="string"), "B": pd.array([None, None], dtype="Float64"),
                       "C": [1, 2], "D": [True, False], "E": [0.5, float("nan")]})
    assert spark_ddl(df) == "`A` STRING, `B` DOUBLE, `C` BIGINT, `D` BOOLEAN, `E` DOUBLE"


def test_every_missing_value_becomes_none():
    df = pd.DataFrame({"A": pd.array(["x", None], dtype="string"), "B": pd.array([1.5, None], dtype="Float64"),
                       "C": [2.5, float("nan")], "D": [1, 2]})
    assert records(df) == [["x", 1.5, 2.5, 1], [None, None, None, 2]]


def test_stage_paths_map_to_the_volume():
    assert volume_path("/Volumes/workspace/segmentation/models/", "@workspace.segmentation.MODELS/production/two_axis.joblib") \
        == "/Volumes/workspace/segmentation/models/production/two_axis.joblib"
    with pytest.raises(ValueError):
        volume_path("/Volumes/x", "not/a/stage")


def test_decimals_are_read_as_doubles_and_names_upper_cased():
    class DecimalType:
        pass

    class StringType:
        pass

    class Field:
        def __init__(self, name, data_type):
            self.name, self.dataType = name, data_type

    class Schema:
        fields = [Field("next_net_revenue", DecimalType()), Field("customer_id", StringType())]

    class FakeSparkFrame:
        schema = Schema()

        def selectExpr(self, *columns):
            self.columns = columns
            return self

        def toPandas(self):
            return pd.DataFrame({"next_net_revenue": [1.5], "customer_id": ["a"]})

    spark_frame = FakeSparkFrame()
    out = _Frame(spark_frame).to_pandas()
    assert spark_frame.columns == ("CAST(`next_net_revenue` AS DOUBLE) AS `next_net_revenue`", "`customer_id`")
    assert list(out.columns) == ["NEXT_NET_REVENUE", "CUSTOMER_ID"]


def test_cleaning_counts_must_match_exactly():
    ref = pd.DataFrame({"STEP": [1, 2], "STAGE": ["a", "b"], "ROWS_REMAINING": [10, 9]})
    assert all(c["PASSED"] for c in compare_cleaning(ref, ref))
    off_by_one = ref.assign(ROWS_REMAINING=[10, 8])
    assert [c["PASSED"] for c in compare_cleaning(off_by_one, ref)] == [True, False]


def test_chosen_k_reads_booleans_from_csv_and_table():
    ref = pd.DataFrame({"RUN_ID": ["production"] * 2, "AXIS": ["value", "value"], "K": [4, 5], "CHOSEN": ["True", "False"]})
    new = pd.DataFrame({"RUN_ID": ["production"] * 2, "AXIS": ["value", "value"], "K": [4, 5], "CHOSEN": [True, False]})
    assert all(c["PASSED"] for c in compare_k(new, ref))
    assert not compare_k(new.assign(CHOSEN=[False, True]), ref)[0]["PASSED"]


def _schemes(auc=0.75, actual=0.5, n=12):
    return pd.DataFrame([{"RUN_ID": "backtest_q2", "SCHEME": "hybrid", "NEXT_ACTIVE_AUC": auc, "REVENUE_SPEARMAN": 0.5,
                          "TOP20_REVENUE_CAPTURE": 0.7, "VALUE_ETA2": 0.3, "BEHAVIOR_ETA2": 0.3, "PERSISTENCE": 0.6,
                          "ACTUAL_ACTIVE_RATE": actual, "N_SEGMENTS": n}])


def test_scheme_metrics_use_their_tolerances():
    assert all(c["PASSED"] for c in compare_schemes(_schemes(auc=0.755), _schemes()))
    assert not all(c["PASSED"] for c in compare_schemes(_schemes(auc=0.77), _schemes()))
    assert not all(c["PASSED"] for c in compare_schemes(_schemes(actual=0.51), _schemes()))
    assert not all(c["PASSED"] for c in compare_schemes(_schemes(n=9), _schemes()))


def test_a_missing_scheme_fails():
    other = _schemes().assign(SCHEME="something_else")
    assert not all(c["PASSED"] for c in compare_schemes(other, _schemes()))


def _segments(customers=(100, 900), revenue=(1000.0, 2000.0)):
    return pd.DataFrame({"SEGMENT": ["C1-B1", "C2-B1"], "CUSTOMERS": list(customers), "EXPECTED_REVENUE": list(revenue),
                         "MEAN_P_ACTIVE": [0.8, 0.5]})


def test_segment_total_is_exact_but_sizes_have_slack():
    assert all(c["PASSED"] for c in compare_segments(_segments(customers=(104, 896)), _segments()))
    total = compare_segments(_segments(customers=(104, 897)), _segments())[0]
    assert total["ITEM"] == "total customers" and not total["PASSED"]
    assert not all(c["PASSED"] for c in compare_segments(_segments(customers=(200, 800)), _segments()))


class _BehaviorStub:  # module level, so joblib can pickle it
    def predict(self, df):
        return pd.DataFrame({"behavior_segment": ["B1" if x > 0 else "B2" for x in df.x]}, index=df.index)


class _OutcomeStub:
    def predict(self, df):
        return pd.DataFrame({"segment": ["C1" if x > 1 else "C2" for x in df.x], "p_active": 0.5,
                             "expected_revenue": df.x * 10.0}, index=df.index)


def test_registered_model_scores_like_the_pipeline(tmp_path, monkeypatch):
    """The Unity Catalog pyfunc gives the same cells as hybrid_assignments, which the pipeline and re-score use."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())  # not an mlflow.db in the repo
    mlflow = pytest.importorskip("mlflow")
    import joblib

    from segmentation.procedures import hybrid_assignments
    from segmentation.registry import HybridSegmenter

    artifacts = {"two_axis": str(tmp_path / "two_axis.joblib"), "outcome_guided": str(tmp_path / "outcome.joblib")}
    joblib.dump(_BehaviorStub(), artifacts["two_axis"])
    joblib.dump(_OutcomeStub(), artifacts["outcome_guided"])
    mlflow.pyfunc.save_model(str(tmp_path / "model"), python_model=HybridSegmenter(), artifacts=artifacts,
                             pip_requirements=["pandas"])
    loaded = mlflow.pyfunc.load_model(str(tmp_path / "model"))
    features = pd.DataFrame({"x": [-1.0, 0.5, 2.0]})
    expected = hybrid_assignments(_BehaviorStub(), _OutcomeStub(), features).reset_index(drop=True)
    pd.testing.assert_frame_equal(loaded.predict(features), expected)
    assert list(expected.segment) == ["C2-B2", "C2-B1", "C1-B1"]
