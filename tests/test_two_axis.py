import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from segmentation.features import BEHAVIOR, VALUE
from segmentation.two_axis import TwoAxisSegmenter, preprocessor, select_k


@pytest.fixture(scope="module")
def fitted(customers):
    return TwoAxisSegmenter().fit(customers)


def test_preprocessor_clips_and_keeps_names(customers):
    out = preprocessor(BEHAVIOR).fit_transform(customers[BEHAVIOR])
    assert list(out.columns) == BEHAVIOR
    assert out.to_numpy().max() <= 3 and out.to_numpy().min() >= -3


def test_behavior_axis_recovers_the_three_buying_styles(customers):
    k, table = select_k(customers, BEHAVIOR, "behavior")
    assert k == 3
    assert table.loc[table.chosen, "eligible"].item()


def test_segments_match_planted_styles(fitted, customers):
    labels = fitted.predict(customers).behavior_segment
    assert adjusted_rand_score(customers["style"], labels) > 0.9


def test_codes_are_ordered_by_meaning(fitted, customers):
    seg = fitted.predict(customers)
    units = customers.groupby(seg.behavior_segment.to_numpy())["avg_units_per_order"].median()
    spend = customers.groupby(seg.value_segment.to_numpy())["net_revenue"].median()
    assert units.idxmax() == "B1"
    assert spend.idxmax() == "V1"
    assert (seg.segment == seg.value_segment + "-" + seg.behavior_segment).all()


def test_rules_are_recorded_for_both_axes(fitted):
    assert set(fitted.rules_) == {"value", "behavior"}
    assert fitted.rules_["behavior"]["fidelity"] > 0.8
    assert "avg_units_per_order" in fitted.rules_["behavior"]["rules"] or \
           "avg_unit_price" in fitted.rules_["behavior"]["rules"]


def test_predict_is_deterministic_for_new_data(fitted, customers):
    a = fitted.predict(customers.iloc[:50])
    b = fitted.predict(customers.iloc[:50].copy())
    assert a.equals(b)


def test_transform_has_both_axes(fitted, customers):
    Z = fitted.transform(customers)
    assert list(Z.columns) == VALUE + BEHAVIOR
    assert np.isfinite(Z.to_numpy()).all()
