"""The monthly re-score must assign the same cells as the production hybrid."""
import numpy as np

from segmentation.feature_sql import create_features_sql, replace_snapshot_sql
from segmentation.naming import case_sql, segment_name
from segmentation.outcome_guided import OutcomeGuidedSegmenter
from segmentation.procedures import ASSIGNMENT_COLS, _assignments, hybrid_assignments
from segmentation.two_axis import TwoAxisSegmenter


def test_hybrid_crosses_value_tier_with_behavior(customers, pooled_customers):
    two_axis = TwoAxisSegmenter().fit(customers)
    outcome = OutcomeGuidedSegmenter(4).fit(pooled_customers)
    h = hybrid_assignments(two_axis, outcome, customers)

    b, c = two_axis.predict(customers), outcome.predict(customers)
    assert (h.segment == c.segment + "-" + b.behavior_segment).all()
    assert h.value_segment.str.fullmatch(r"C\d").all()
    assert h.behavior_segment.str.fullmatch(r"B\d").all()
    np.testing.assert_allclose(h.p_active, c.p_active)
    np.testing.assert_allclose(h.expected_revenue, c.expected_revenue)

    rows = _assignments(customers, h, "rescore", "hybrid", "2011-09-10")
    assert list(rows.columns) == ASSIGNMENT_COLS
    assert (rows.RUN_ID == "rescore").all() and rows.CUSTOMER_ID.is_unique


def test_snapshot_sql_rebuilds_one_snapshot():
    delete, insert = replace_snapshot_sql("2011-12-10", "2011-12-09")
    assert delete.endswith("WHERE snapshot_date = '2011-12-10'")
    assert "FROM VALUES ('2011-12-10')" in insert and insert.startswith("INSERT INTO")
    full = create_features_sql(["2011-09-10", "2011-12-10"], "2011-12-09")
    assert "FROM VALUES ('2011-09-10'), ('2011-12-10')" in full


def test_names():
    assert segment_name("C1-B2") == "Top spenders / Holiday buyers"
    assert case_sql("x", {"C1": "Top"}) == "CASE x WHEN 'C1' THEN 'Top' END"
