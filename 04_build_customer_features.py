"""Build MARTS.CUSTOMER_FEATURES: one row per customer per quarterly snapshot.

Each snapshot looks back 365 days for behavior features and forward 91 days for outcomes,
so segments fit on one snapshot can be checked against what customers did next, and the same
customer can be followed across snapshots. Only lines before SNAPSHOT_DATE feed the features.

The last snapshot (2011-12-10) is "today" for the dataset: it has features but no outcomes,
and is the one the production segmentation scores.
"""
from common import get_session, run_sql
from segmentation.feature_sql import FEATURE_TABLE as FEATURES, create_features_sql

SNAPSHOTS = ["2010-12-10", "2011-03-10", "2011-06-10", "2011-09-10", "2011-12-10"]
LAST_DATA_DATE = "2011-12-09"


def build_features():
    session = get_session()
    run_sql(session, create_features_sql(SNAPSHOTS, LAST_DATA_DATE))
    print(session.sql(f"""
        SELECT snapshot_date, COUNT(*) AS customers,
               ROUND(AVG(next_active::INT), 3) AS next_active_rate,
               ROUND(MEDIAN(net_revenue)) AS median_net_revenue,
               ROUND(MEDIAN(orders), 1) AS median_orders
        FROM {FEATURES} GROUP BY 1 ORDER BY 1
    """).to_pandas().to_string(index=False))
    session.close()


if __name__ == "__main__":
    build_features()
