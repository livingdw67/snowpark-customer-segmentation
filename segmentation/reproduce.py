"""Does the Databricks build reproduce the Snowflake results committed in results/?

Anything that depends only on the data and the SQL must match exactly. Anything a fitted model
produces gets a tolerance, because row order and CPU architecture can move it slightly. These
thresholds were fixed before the first Databricks run.
"""
from __future__ import annotations

import pandas as pd

SCHEME_TOLERANCE = {"NEXT_ACTIVE_AUC": 0.01, "REVENUE_SPEARMAN": 0.01, "TOP20_REVENUE_CAPTURE": 0.02,
                    "VALUE_ETA2": 0.02, "BEHAVIOR_ETA2": 0.02, "PERSISTENCE": 0.02}
# Data only: the share of customers who actually ordered can't depend on the model.
ACTUAL_RATE_TOLERANCE = 0.0005
SEGMENT_SIZE_TOLERANCE = 0.03  # relative to the Snowflake size, with at least 5 customers of slack
SEGMENT_REVENUE_TOLERANCE = 0.05  # relative, per segment
TOTAL_REVENUE_TOLERANCE = 0.02  # relative, all segments
P_ACTIVE_TOLERANCE = 0.02


def _check(area: str, item: str, expected, actual, passed: bool) -> dict:
    return {"AREA": area, "ITEM": item, "SNOWFLAKE": str(expected), "DATABRICKS": str(actual), "PASSED": bool(passed)}


def _within(expected: float, actual: float, tolerance: float) -> bool:
    return not (pd.isna(expected) or pd.isna(actual)) and abs(float(expected) - float(actual)) <= tolerance


def compare_cleaning(actual: pd.DataFrame, reference: pd.DataFrame) -> list[dict]:
    got = actual.set_index("STEP").ROWS_REMAINING
    return [_check("cleaning", r.STAGE, int(r.ROWS_REMAINING), int(got.get(r.STEP, -1)),
                   int(got.get(r.STEP, -1)) == int(r.ROWS_REMAINING))
            for r in reference.itertuples()]


def compare_k(actual: pd.DataFrame, reference: pd.DataFrame) -> list[dict]:
    def chosen(df):
        df = df[df.RUN_ID == "production"]
        return df[df.CHOSEN.astype(str).str.lower() == "true"].set_index("AXIS").K.astype(int).to_dict()

    expected, got = chosen(reference), chosen(actual)
    return [_check("chosen k", axis, k, got.get(axis), got.get(axis) == k) for axis, k in sorted(expected.items())]


def compare_schemes(actual: pd.DataFrame, reference: pd.DataFrame) -> list[dict]:
    keys = ["RUN_ID", "SCHEME"]
    merged = reference.merge(actual, on=keys, how="left", suffixes=("_REF", "_NEW"))
    checks = []
    for r in merged.to_dict("records"):
        label = f"{r['RUN_ID']} {r['SCHEME']}"
        for col, tolerance in SCHEME_TOLERANCE.items():
            if not pd.isna(r[f"{col}_REF"]):
                checks.append(_check("backtest", f"{label} {col}", round(r[f"{col}_REF"], 4), r[f"{col}_NEW"],
                                     _within(r[f"{col}_REF"], r[f"{col}_NEW"], tolerance)))
        if not pd.isna(r["ACTUAL_ACTIVE_RATE_REF"]):
            checks.append(_check("backtest", f"{label} ACTUAL_ACTIVE_RATE", round(r["ACTUAL_ACTIVE_RATE_REF"], 4),
                                 r["ACTUAL_ACTIVE_RATE_NEW"],
                                 _within(r["ACTUAL_ACTIVE_RATE_REF"], r["ACTUAL_ACTIVE_RATE_NEW"], ACTUAL_RATE_TOLERANCE)))
        if not pd.isna(r["N_SEGMENTS_REF"]):
            checks.append(_check("backtest", f"{label} N_SEGMENTS", int(r["N_SEGMENTS_REF"]), r["N_SEGMENTS_NEW"],
                                 r["N_SEGMENTS_NEW"] == r["N_SEGMENTS_REF"]))
    return checks


def compare_segments(actual: pd.DataFrame, reference: pd.DataFrame) -> list[dict]:
    new, ref = actual.set_index("SEGMENT"), reference.set_index("SEGMENT")
    checks = [_check("segments", "total customers", int(ref.CUSTOMERS.sum()), int(new.CUSTOMERS.sum()),
                     int(ref.CUSTOMERS.sum()) == int(new.CUSTOMERS.sum())),
              _check("segments", "segment count", len(ref), len(new), len(ref) == len(new)),
              _check("segments", "total expected revenue", round(ref.EXPECTED_REVENUE.sum()), round(new.EXPECTED_REVENUE.sum()),
                     _within(ref.EXPECTED_REVENUE.sum(), new.EXPECTED_REVENUE.sum(),
                             TOTAL_REVENUE_TOLERANCE * ref.EXPECTED_REVENUE.sum()))]
    for code, r in ref.iterrows():
        if code not in new.index:
            checks.append(_check("segments", f"{code} present", True, False, False))
            continue
        n = new.loc[code]
        checks += [
            _check("segments", f"{code} customers", int(r.CUSTOMERS), int(n.CUSTOMERS),
                   _within(r.CUSTOMERS, n.CUSTOMERS, max(5, SEGMENT_SIZE_TOLERANCE * r.CUSTOMERS))),
            _check("segments", f"{code} expected revenue", round(r.EXPECTED_REVENUE), round(n.EXPECTED_REVENUE),
                   _within(r.EXPECTED_REVENUE, n.EXPECTED_REVENUE, SEGMENT_REVENUE_TOLERANCE * r.EXPECTED_REVENUE)),
            _check("segments", f"{code} P(buys again)", round(r.MEAN_P_ACTIVE, 3), round(n.MEAN_P_ACTIVE, 3),
                   _within(r.MEAN_P_ACTIVE, n.MEAN_P_ACTIVE, P_ACTIVE_TOLERANCE)),
        ]
    return checks
