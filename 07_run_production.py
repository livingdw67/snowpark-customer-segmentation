"""Production run: fit on the latest snapshot with known outcomes, score "today".

The run settings and SQL are in segmentation/production.py, shared with the Databricks build.
"""
import argparse
import json

from common import DATABASE, get_session, run_sql
from segmentation.production import (ALL_SNAPSHOTS, FIT, MIGRATION_SQL, RUN_ID, TODAY, check_behavior_names,
                                     current_view_sql)

ML = f"{DATABASE}.ML"


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
