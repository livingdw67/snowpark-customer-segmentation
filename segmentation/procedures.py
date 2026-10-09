"""Stored procedure handlers. Each reads from and writes to Snowflake tables only."""
import io
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.manifold import ClassicalMDS

from .evaluation import forecast_bias, metrics, persistence, rfm_baseline, score_scheme
from .features import DATABASE, load_snapshots
from .outcome_guided import OutcomeGuidedSegmenter
from .profiles import describe
from .two_axis import TwoAxisSegmenter

ML = f"{DATABASE}.ML"
ASSIGNMENT_COLS = ["RUN_ID", "METHOD", "FIT_SNAPSHOT", "SNAPSHOT_DATE", "CUSTOMER_ID", "SEGMENT",
                   "VALUE_SEGMENT", "BEHAVIOR_SEGMENT", "P_ACTIVE", "EXPECTED_REVENUE"]


def _replace(session, df: pd.DataFrame, table: str, run_id: str, method: str | None = None):
    """Overwrite this run's (and method's) rows, leaving other runs in place."""
    df = df.copy()
    df.columns = [c.upper() for c in df.columns]
    if session.sql(f"SHOW TABLES LIKE '{table}' IN SCHEMA {ML}").collect():
        where = f"RUN_ID = '{run_id}'" + (f" AND METHOD = '{method}'" if method else "")
        session.sql(f"DELETE FROM {ML}.{table} WHERE {where}").collect()
    session.write_pandas(df, table, database=DATABASE, schema="ML",
                         auto_create_table=True, use_logical_type=True)


def _save_model(session, model, run_id: str, name: str):
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    session.file.put_stream(buf, f"@{ML}.MODELS/{run_id}/{name}.joblib",
                            auto_compress=False, overwrite=True)


def _load_model(session, run_id: str, name: str):
    return joblib.load(session.file.get_stream(f"@{ML}.MODELS/{run_id}/{name}.joblib"))


def _assignments(df, scored, run_id, method, fit_date) -> pd.DataFrame:
    out = pd.DataFrame({
        "RUN_ID": run_id, "METHOD": method, "FIT_SNAPSHOT": fit_date,
        "SNAPSHOT_DATE": df["snapshot_date"].to_numpy(),
        "CUSTOMER_ID": df["customer_id"].to_numpy(),
    })
    # Typed nulls for columns a method doesn't produce, so both methods share one table schema.
    for col, dtype in (("segment", "string"), ("value_segment", "string"),
                       ("behavior_segment", "string"), ("p_active", "Float64"),
                       ("expected_revenue", "Float64")):
        values = scored[col].to_numpy() if col in scored else [None] * len(df)
        out[col.upper()] = pd.array(values, dtype=dtype)
    return out[ASSIGNMENT_COLS]


def _labeled_snapshots(session, up_to: str) -> list[str]:
    rows = session.sql(f"""
        SELECT DISTINCT snapshot_date::STRING AS d FROM {DATABASE}.MARTS.CUSTOMER_FEATURES
        WHERE next_active IS NOT NULL AND snapshot_date <= '{up_to}' ORDER BY 1
    """).collect()
    return [r["D"] for r in rows]


def _split(df, fit_date):
    return df[df.snapshot_date == fit_date].reset_index(drop=True)


def train_two_axis(session, run_id: str, fit_date: str, score_dates: str) -> str:
    dates = sorted({fit_date, *score_dates.split(",")})
    df = load_snapshots(session, dates).reset_index(drop=True)
    model = TwoAxisSegmenter().fit(_split(df, fit_date))

    _replace(session, _assignments(df, model.predict(df), run_id, "two_axis", fit_date),
             "SEGMENT_ASSIGNMENTS", run_id, "two_axis")
    _replace(session, model.profiles_.assign(run_id=run_id, method="two_axis",
                                           rule=pd.array([None] * len(model.profiles_), dtype="string")),
             "SEGMENT_PROFILES", run_id, "two_axis")
    _replace(session, model.k_selection_.assign(run_id=run_id, method="two_axis"),
             "K_SELECTION", run_id, "two_axis")
    rules = pd.DataFrame([{"run_id": run_id, "method": "two_axis", "axis": axis,
                           "fidelity": r["fidelity"], "fidelity_metric": "surrogate tree accuracy (5-fold)",
                           "rules": r["rules"]} for axis, r in model.rules_.items()])
    _replace(session, rules, "SEGMENT_RULES", run_id, "two_axis")
    _save_model(session, model, run_id, "two_axis")
    return json.dumps({"k_value": model.k_value_, "k_behavior": model.k_behavior_,
                       "rule_fidelity": {a: round(r["fidelity"], 3) for a, r in model.rules_.items()}})


def train_outcome_guided(session, run_id: str, method: str, fit_date: str,
                         score_dates: str, n_segments: int) -> str:
    """Trains on every snapshot up to fit_date whose outcomes are known (pooled)."""
    score = set(score_dates.split(","))
    df = load_snapshots(session, sorted(score | {fit_date}))
    history = load_snapshots(session, _labeled_snapshots(session, fit_date))
    model = OutcomeGuidedSegmenter(int(n_segments)).fit(history)
    df = df.reset_index(drop=True)

    _replace(session, _assignments(df, model.predict(df), run_id, method, fit_date),
             "SEGMENT_ASSIGNMENTS", run_id, method)
    _replace(session, model.profiles_.assign(run_id=run_id, method=method),
             "SEGMENT_PROFILES", run_id, method)
    rules = pd.DataFrame([{"run_id": run_id, "method": method, "axis": "outcome",
                           "fidelity": model.tree_fidelity_r2_,
                           "fidelity_metric": "R^2 of rule tree vs. model's log expected revenue",
                           "rules": "\n".join(f"{s}: {r}" for s, r in sorted(model.rules_.items()))}])
    _replace(session, rules, "SEGMENT_RULES", run_id, method)
    _save_model(session, model, run_id, method)
    return json.dumps({"n_segments": len(model.rules_), "trained_on": sorted(history.snapshot_date.unique()),
                       "tree_fidelity_r2": round(model.tree_fidelity_r2_, 3)})


def build_hybrid(session, run_id: str, snapshot_dates: str) -> str:
    """Cross C's value tiers (method outcome_guided) with B's behavior axis (method two_axis).

    No new model: both inputs are already scored for these snapshots in this run. Profiles
    describe the latest snapshot, with each cell's predicted next-quarter value attached.
    """
    dates = snapshot_dates.split(",")
    date_list = ", ".join(f"'{d}'" for d in dates)
    a = session.sql(f"""
        SELECT c.fit_snapshot, c.snapshot_date::STRING AS snapshot_date, c.customer_id,
               c.segment AS value_tier, b.behavior_segment, c.p_active, c.expected_revenue
        FROM {ML}.SEGMENT_ASSIGNMENTS c
        JOIN {ML}.SEGMENT_ASSIGNMENTS b
          ON b.run_id = c.run_id AND b.snapshot_date = c.snapshot_date
         AND b.customer_id = c.customer_id AND b.method = 'two_axis'
        WHERE c.run_id = '{run_id}' AND c.method = 'outcome_guided'
          AND c.snapshot_date IN ({date_list})
    """).to_pandas()
    a.columns = a.columns.str.lower()
    scored = pd.DataFrame({"segment": a.value_tier + "-" + a.behavior_segment,
                           "value_segment": a.value_tier, "behavior_segment": a.behavior_segment,
                           "p_active": a.p_active.astype(float),
                           "expected_revenue": a.expected_revenue.astype(float)})
    _replace(session, _assignments(a, scored, run_id, "hybrid", a.fit_snapshot.iloc[0]),
             "SEGMENT_ASSIGNMENTS", run_id, "hybrid")

    latest = max(dates)
    df = load_snapshots(session, [latest]).merge(
        a[a.snapshot_date == latest].assign(segment=scored.segment), on=["snapshot_date", "customer_id"])
    profiles = describe(df, df.segment).assign(axis="hybrid", run_id=run_id, method="hybrid",
                                               rule=pd.array([None] * df.segment.nunique(), dtype="string"))
    cell = df.groupby("segment").agg(mean_p_active=("p_active", "mean"),
                                     expected_revenue=("expected_revenue", "sum"))
    profiles = profiles.merge(cell, left_on="segment", right_index=True)
    _replace(session, profiles, "HYBRID_PROFILES", run_id)
    return f"{len(a)} customers, {df.segment.nunique()} cells on {latest}"


def compare_methods(session, run_id: str, train_date: str, eval_date: str) -> str:
    df = load_snapshots(session, [train_date, eval_date])
    a = session.table(f"{ML}.SEGMENT_ASSIGNMENTS").filter(f"RUN_ID = '{run_id}'").to_pandas()
    a.columns = a.columns.str.lower()
    a["snapshot_date"] = a["snapshot_date"].astype(str)
    fid = session.table(f"{ML}.SEGMENT_RULES").filter(f"RUN_ID = '{run_id}'").to_pandas()
    fid = {(m, ax): f for m, ax, f in zip(fid.METHOD, fid.AXIS, fid.FIDELITY)}

    def frame(method, date):
        part = a[(a.method == method) & (a.snapshot_date == date)]
        return df[df.snapshot_date == date].merge(part, on=["snapshot_date", "customer_id"])

    rows = []
    tr, ev = frame("two_axis", train_date), frame("two_axis", eval_date)
    base_tr, base_ev = rfm_baseline(tr, tr), rfm_baseline(tr, ev)
    rows.append(score_scheme("rfm_score_baseline", tr, ev, base_tr, base_ev) |
                {"persistence": persistence(tr.customer_id, base_tr, ev.customer_id, base_ev)})
    for name, col, fkey in (("two_axis_value", "value_segment", "value"),
                            ("two_axis_grid", "segment", None)):
        r = score_scheme(name, tr, ev, tr[col], ev[col],
                         fid.get(("two_axis", fkey)) if fkey else
                         min(fid[("two_axis", "value")], fid[("two_axis", "behavior")]))
        rows.append(r | {"persistence": persistence(tr.customer_id, tr[col], ev.customer_id, ev[col])})

    for method in sorted(a.method.unique()):
        if not (method.startswith("outcome_guided") or method == "hybrid"):
            continue
        tr, ev = frame(method, train_date), frame(method, eval_date)
        r = score_scheme(method, tr, ev, tr.segment, ev.segment, fid.get((method, "outcome")))
        rows.append(r | {"persistence": persistence(tr.customer_id, tr.segment, ev.customer_id, ev.segment)})
    # Reference ceiling: the outcome models' own continuous scores, no segmentation at all.
    ev = frame("outcome_guided", eval_date)
    rows.append(metrics("outcome_model_continuous", ev, ev.p_active.to_numpy(),
                        np.log1p(ev.expected_revenue.to_numpy()))
                | forecast_bias(ev, ev.p_active.to_numpy(), ev.expected_revenue.to_numpy()))

    out = pd.DataFrame(rows).assign(run_id=run_id, train_snapshot=train_date, eval_snapshot=eval_date)
    _replace(session, out, "METHOD_COMPARISON", run_id)
    return f"{len(out)} schemes compared"


def build_segment_map(session, run_id: str, snapshot_date: str, n_sample: int) -> str:
    """2-D map of customers with ClassicalMDS (scikit-learn 1.8) on the two-axis feature space."""
    df = load_snapshots(session, [snapshot_date])
    df = df.sample(n=min(int(n_sample), len(df)), random_state=0).reset_index(drop=True)
    model = _load_model(session, run_id, "two_axis")
    coords = ClassicalMDS(n_components=2).fit_transform(model.transform(df).to_numpy())
    seg = model.predict(df)
    out = pd.DataFrame({"run_id": run_id, "snapshot_date": snapshot_date,
                        "customer_id": df.customer_id, "x": coords[:, 0], "y": coords[:, 1],
                        "value_segment": seg.value_segment, "behavior_segment": seg.behavior_segment})
    _replace(session, out, "SEGMENT_MAP", run_id)
    return f"{len(out)} customers mapped"
