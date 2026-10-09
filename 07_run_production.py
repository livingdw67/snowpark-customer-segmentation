"""Production run: fit on the latest snapshot with known outcomes, score "today".

Option B is fit on 2011-09-10 and applied (frozen) to every snapshot, so segment migration
quarter over quarter reflects customers changing, not the model changing. Option C is fit on
2011-09-10 (its outcomes run to the end of the data) and scores 2011-12-10.
"""
import argparse
import json

from common import DATABASE, get_session, run_sql
from segmentation.naming import BEHAVIOR_NAMES, TIER_NAMES, case_sql

ML = f"{DATABASE}.ML"
RUN_ID = "production"
FIT = "2011-09-10"
TODAY = "2011-12-10"
ALL_SNAPSHOTS = "2010-12-10,2011-03-10,2011-06-10,2011-09-10,2011-12-10"

MIGRATION_SQL = f"""
CREATE OR REPLACE TABLE {ML}.SEGMENT_MIGRATION AS
WITH a AS (
    SELECT snapshot_date, customer_id, value_segment
    FROM {ML}.SEGMENT_ASSIGNMENTS
    WHERE run_id = '{RUN_ID}' AND method = 'two_axis'
),
pairs AS (
    SELECT snapshot_date AS from_snapshot,
           LEAD(snapshot_date) OVER (ORDER BY snapshot_date) AS to_snapshot
    FROM (SELECT DISTINCT snapshot_date FROM a)
),
-- Everyone active in either snapshot of each consecutive pair.
pair_customers AS (
    SELECT DISTINCT p.from_snapshot, p.to_snapshot, x.customer_id
    FROM pairs p
    JOIN a x ON x.snapshot_date IN (p.from_snapshot, p.to_snapshot)
    WHERE p.to_snapshot IS NOT NULL
)
SELECT pc.from_snapshot, pc.to_snapshot,
       -- Not in the earlier snapshot = no purchase in the prior 12 months.
       COALESCE(f.value_segment, 'NEW OR RETURNING') AS from_segment,
       COALESCE(t.value_segment, 'LAPSED (12M)')     AS to_segment,
       COUNT(*)                                      AS customers
FROM pair_customers pc
LEFT JOIN a f ON f.snapshot_date = pc.from_snapshot AND f.customer_id = pc.customer_id
LEFT JOIN a t ON t.snapshot_date = pc.to_snapshot   AND t.customer_id = pc.customer_id
GROUP BY ALL
"""


def check_behavior_names(session):
    p = session.sql(f"""
        SELECT segment, median_avg_units_per_order AS units, median_q4_revenue_share AS q4,
               median_avg_unit_price AS price
        FROM {ML}.SEGMENT_PROFILES
        WHERE run_id = '{RUN_ID}' AND method = 'two_axis' AND axis = 'behavior'
    """).to_pandas().set_index("SEGMENT")
    assert set(p.index) == set(BEHAVIOR_NAMES), f"behavior segments changed: {list(p.index)}"
    assert p.UNITS.idxmax() == "B1", "B1 is no longer the largest-basket group"
    assert p.Q4.idxmax() == "B2", "B2 is no longer the holiday-heavy group"
    assert p.PRICE.idxmax() == "B3", "B3 is no longer the small-basket, higher-price group"


def current_view_sql() -> str:
    """Latest snapshot per customer. Rows from the monthly RESCORE_CUSTOMERS Task (run_id
    'rescore') take precedence over this production run when both cover the same snapshot."""
    tier = case_sql("h.value_segment", TIER_NAMES)
    behavior = case_sql("h.behavior_segment", BEHAVIOR_NAMES)
    return f"""
CREATE OR REPLACE VIEW {ML}.CUSTOMER_SEGMENTS_CURRENT
COMMENT = 'One row per active customer at the latest snapshot: hybrid segment, names, scores'
AS
WITH scored AS (
    SELECT * FROM {ML}.SEGMENT_ASSIGNMENTS
    WHERE method = 'hybrid'
      AND (run_id = 'rescore' OR (run_id = '{RUN_ID}' AND snapshot_date = '{TODAY}'))
)
SELECT h.snapshot_date, h.customer_id,
       h.segment                                   AS segment_code,
       {tier} || ' / ' ||
       {behavior}                                  AS segment_name,
       h.value_segment                             AS value_tier,
       {tier}                                      AS value_tier_name,
       h.behavior_segment,
       {behavior}                                  AS behavior_name,
       ROUND(h.p_active, 3)                        AS p_buys_next_quarter,
       ROUND(h.expected_revenue, 2)                AS expected_revenue_next_quarter,
       h.run_id                                    AS scored_by
FROM scored h
WHERE h.snapshot_date = (SELECT MAX(snapshot_date) FROM scored)
QUALIFY ROW_NUMBER() OVER (PARTITION BY h.customer_id
                           ORDER BY IFF(h.run_id = 'rescore', 0, 1)) = 1
"""


def call(session, sql):
    result = session.sql(f"CALL {ML}.{sql}").collect()[0][0]
    print(f"{sql.split('(')[0]}: {result}")
    return result


def production():
    session = get_session()
    two_axis = json.loads(call(session, f"TRAIN_TWO_AXIS('{RUN_ID}', '{FIT}', '{ALL_SNAPSHOTS}')"))
    k_value, k_grid = two_axis["k_value"], two_axis["k_value"] * two_axis["k_behavior"]
    call(session, f"TRAIN_OUTCOME_GUIDED('{RUN_ID}', 'outcome_guided', '{FIT}', '{TODAY}', {k_value})")
    call(session, f"TRAIN_OUTCOME_GUIDED('{RUN_ID}', 'outcome_guided_grid_size', '{FIT}', '{TODAY}', {k_grid})")
    call(session, f"BUILD_HYBRID('{RUN_ID}', '{FIT},{TODAY}')")
    check_behavior_names(session)
    run_sql(session, current_view_sql())
    print("Built CUSTOMER_SEGMENTS_CURRENT")
    call(session, f"BUILD_SEGMENT_MAP('{RUN_ID}', '{TODAY}', 2000)")
    run_sql(session, MIGRATION_SQL)
    print("Built SEGMENT_MIGRATION")
    session.close()


def rebuild_view():
    """Recreate CUSTOMER_SEGMENTS_CURRENT without retraining (e.g. after a view change)."""
    session = get_session()
    check_behavior_names(session)
    run_sql(session, current_view_sql())
    print("Built CUSTOMER_SEGMENTS_CURRENT")
    session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-only", action="store_true",
                        help="only rebuild CUSTOMER_SEGMENTS_CURRENT from existing assignments")
    if parser.parse_args().view_only:
        rebuild_view()
    else:
        production()
