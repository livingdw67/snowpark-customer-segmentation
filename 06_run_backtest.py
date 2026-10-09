"""Out-of-time comparison of option B (two-axis) and option C (outcome-guided).

Each fold fits both methods on one snapshot and judges them on the next quarter's snapshot
against what customers actually did over the following 91 days. Two folds, so a ranking that
holds in only one quarter is visible as such. C is run at two sizes so it can be compared with
B's value axis and with B's full grid at the same number of segments. The hybrid crosses C's
value tiers with B's behavior axis.
"""
import json

from common import DATABASE, get_session

ML = f"{DATABASE}.ML"
FOLDS = {"backtest_q2": ("2011-03-10", "2011-06-10"),
         "backtest_q3": ("2011-06-10", "2011-09-10")}
COLS = ["RUN_ID", "SCHEME", "N_SEGMENTS", "NEXT_ACTIVE_AUC", "REVENUE_SPEARMAN",
        "TOP20_REVENUE_CAPTURE", "VALUE_ETA2", "BEHAVIOR_ETA2", "RULE_FIDELITY",
        "MIN_SEGMENT_SHARE", "PERSISTENCE"]


def call(session, sql):
    result = session.sql(f"CALL {ML}.{sql}").collect()[0][0]
    print(f"{sql.split('(')[0]}: {result}")
    return result


def run_fold(session, run_id: str, train: str, eval_: str):
    two_axis = json.loads(call(session, f"TRAIN_TWO_AXIS('{run_id}', '{train}', '{eval_}')"))
    k_value, k_grid = two_axis["k_value"], two_axis["k_value"] * two_axis["k_behavior"]
    call(session, f"TRAIN_OUTCOME_GUIDED('{run_id}', 'outcome_guided', '{train}', '{eval_}', {k_value})")
    call(session, f"TRAIN_OUTCOME_GUIDED('{run_id}', 'outcome_guided_grid_size', '{train}', '{eval_}', {k_grid})")
    call(session, f"BUILD_HYBRID('{run_id}', '{train},{eval_}')")
    call(session, f"COMPARE_METHODS('{run_id}', '{train}', '{eval_}')")


def backtest():
    session = get_session()
    # Derived output: rebuilt from scratch so its columns always match the current code.
    session.sql(f"DROP TABLE IF EXISTS {ML}.METHOD_COMPARISON").collect()
    for run_id, (train, eval_) in FOLDS.items():
        run_fold(session, run_id, train, eval_)
    table = session.table(f"{ML}.METHOD_COMPARISON").filter("RUN_ID LIKE 'backtest_%'").to_pandas()
    print(table[COLS].sort_values(["SCHEME", "RUN_ID"]).round(3).to_string(index=False))
    session.close()


if __name__ == "__main__":
    backtest()
