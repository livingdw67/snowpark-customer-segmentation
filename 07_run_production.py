"""Production run: fit on the latest snapshot with known outcomes, score "today".

Option B is fit on 2011-09-10 and applied (frozen) to every snapshot, so segment migration
quarter over quarter reflects customers changing, not the model changing. Option C is fit on
2011-09-10 (its outcomes run to the end of the data) and scores 2011-12-10.
"""
import json

from common import DATABASE, get_session, run_sql

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


# Persona names for the published view. Codes are ordered by meaning (C1 = highest predicted
# value, B1 = largest baskets), and check_behavior_names() confirms each behavior group still
# matches its name before the view is built.
TIER_NAMES = {"C1": "Top spenders", "C2": "Core", "C3": "Occasional", "C4": "Low value"}
BEHAVIOR_NAMES = {"B1": "Bulk buyers", "B2": "Holiday buyers", "B3": "Small-basket buyers"}


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
    tier = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in TIER_NAMES.items())
    behavior = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in BEHAVIOR_NAMES.items())
    return f"""
CREATE OR REPLACE VIEW {ML}.CUSTOMER_SEGMENTS_CURRENT
COMMENT = 'One row per active customer at the latest snapshot: hybrid segment, names, scores'
AS
SELECT h.snapshot_date, h.customer_id,
       h.segment                                   AS segment_code,
       CASE h.value_segment {tier} END || ' / ' ||
       CASE h.behavior_segment {behavior} END      AS segment_name,
       h.value_segment                             AS value_tier,
       CASE h.value_segment {tier} END             AS value_tier_name,
       h.behavior_segment,
       CASE h.behavior_segment {behavior} END      AS behavior_name,
       ROUND(h.p_active, 3)                        AS p_buys_next_quarter,
       ROUND(h.expected_revenue, 2)                AS expected_revenue_next_quarter
FROM {ML}.SEGMENT_ASSIGNMENTS h
WHERE h.run_id = '{RUN_ID}' AND h.method = 'hybrid' AND h.snapshot_date = '{TODAY}'
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


if __name__ == "__main__":
    production()
