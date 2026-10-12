"""Read-only data access for the assistant, with the same logical tables in both modes.

local - DuckDB over the committed results/*.csv. No account needed and no customer-level
        data: the repo only publishes aggregated results.
live  - Snowflake. The same tables are session-scoped temporary views over the ML schema,
        plus customer-level CUSTOMERS and MIGRATION.

The agent writes SQL against these names only, so prompts and queries are identical across
modes apart from the SQL dialect.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from segmentation.naming import BEHAVIOR_NAMES, TIER_NAMES, case_sql, segment_name

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MAX_ROWS = 200
MAX_HINT_VALUES = 15
NO_HINT_COLUMNS = {"TRAITS", "RULE"}  # free text: listing values wouldn't help and bloats the prompt

SEGMENT_COLUMNS = [
    "SEGMENT", "SEGMENT_NAME", "VALUE_TIER", "VALUE_TIER_NAME", "BEHAVIOR_SEGMENT", "BEHAVIOR_NAME",
    "CUSTOMERS", "SHARE", "TRAITS", "MEDIAN_RECENCY_DAYS", "MEDIAN_ORDERS", "MEDIAN_NET_REVENUE",
    "MEDIAN_AVG_UNITS_PER_ORDER", "MEDIAN_AVG_UNIT_PRICE", "MEDIAN_AVG_LINES_PER_ORDER",
    "MEDIAN_RETURN_RATE", "MEDIAN_Q4_REVENUE_SHARE", "MEDIAN_AVG_ORDER_VALUE", "MEDIAN_TENURE_DAYS",
    "UK_SHARE", "MEAN_P_ACTIVE", "EXPECTED_REVENUE",
]

TABLE_NOTES = {
    "SEGMENTS": (
        "One row per hybrid segment at the latest snapshot (2011-12-10). SEGMENT codes are "
        "<value tier>-<buying style>: C1 = highest predicted next-quarter value ... C4 = lowest; "
        "B1 = bulk buyers, B2 = holiday buyers, B3 = small-basket buyers. MEDIAN_* describe the "
        "typical customer over the last 365 days (money in GBP). SHARE, UK_SHARE, "
        "MEDIAN_RETURN_RATE and MEDIAN_Q4_REVENUE_SHARE are fractions (0.07 = 7%). "
        "MEAN_P_ACTIVE = average predicted probability of buying in the next 91 days. "
        "EXPECTED_REVENUE = predicted next-quarter revenue summed over the segment's customers."),
    "BACKTEST": (
        "Out-of-time comparison of segmentation schemes: one row per SCHEME per fold "
        "(RUN_ID backtest_q2 = fit 2011-03-10, scored 2011-06-10; backtest_q3 = fit 2011-06-10, "
        "scored 2011-09-10). Schemes: rfm_score_baseline, two_axis_value, two_axis_grid, "
        "outcome_guided, outcome_guided_grid_size, hybrid, outcome_model_continuous (an "
        "unsegmented ceiling). NEXT_ACTIVE_AUC = how well segments rank who buys again; "
        "TOP20_REVENUE_CAPTURE = share of next-quarter revenue held by the top 20% of customers "
        "ranked by segment; VALUE_ETA2 / BEHAVIOR_ETA2 = variance in value / buying-style "
        "features explained by segment; PERSISTENCE = share of customers keeping their segment."),
    "K_CHOICES": (
        "How the number of segments (K) was chosen for each AXIS of the two-axis method in "
        "production. STABILITY_ARI = bootstrap stability (>= 0.85 required), MIN_SHARE = "
        "smallest segment's share (>= 0.05 required), CRITERION = what was maximized, "
        "CHOSEN = the selected K."),
    "CLEANING": (
        "Row counts after each data-cleaning step, from raw transactions to product lines, in "
        "STEP order. ROWS_REMAINING = rows left after the step; CHANGE = rows added by the step "
        "(negative: -11812 means 11,812 rows were removed; empty for the first step)."),
    "CUSTOMERS": (
        "Live only. One row per active customer at the latest snapshot: SEGMENT_CODE, "
        "SEGMENT_NAME, VALUE_TIER, BEHAVIOR_SEGMENT, P_BUYS_NEXT_QUARTER, "
        "EXPECTED_REVENUE_NEXT_QUARTER (GBP), SCORED_BY ('production' or 'rescore')."),
    "MIGRATION": (
        "Live only. Customers moving between two-axis value segments from FROM_SNAPSHOT to "
        "TO_SNAPSHOT ('NEW OR RETURNING' / 'LAPSED (12M)' for customers entering or leaving)."),
}

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|CALL|COPY|PUT|GET|"
    r"REMOVE|USE|EXECUTE|SET|UNSET|ATTACH|DETACH|INSTALL|LOAD|PRAGMA|EXPORT|IMPORT|UNDROP)\b",
    re.IGNORECASE)


class UnsafeQuery(ValueError):
    pass


def validate_sql(sql: str) -> str:
    """Allow exactly one read-only SELECT/WITH statement."""
    sql = sql.strip().rstrip(";").strip()
    code = re.sub(r"'(?:[^']|'')*'", "''", sql)  # ignore keywords inside string literals
    if ";" in code:
        raise UnsafeQuery("only one statement is allowed")
    if not re.match(r"\s*(SELECT|WITH)\b", code, re.IGNORECASE):
        raise UnsafeQuery("only SELECT queries are allowed")
    if m := _FORBIDDEN.search(code):
        raise UnsafeQuery(f"'{m.group(1).upper()}' is not allowed")
    return sql


def add_names(profiles: pd.DataFrame) -> pd.DataFrame:
    """hybrid_profiles rows -> SEGMENTS columns."""
    df = profiles.copy()
    df["VALUE_TIER"] = df.SEGMENT.str.split("-").str[0]
    df["BEHAVIOR_SEGMENT"] = df.SEGMENT.str.split("-").str[1]
    df["SEGMENT_NAME"] = df.SEGMENT.map(segment_name)
    df["VALUE_TIER_NAME"] = df.VALUE_TIER.map(TIER_NAMES)
    df["BEHAVIOR_NAME"] = df.BEHAVIOR_SEGMENT.map(BEHAVIOR_NAMES)
    return df[SEGMENT_COLUMNS].sort_values("SEGMENT").reset_index(drop=True)


class LocalData:
    mode = "local"
    dialect = "DuckDB"

    def __init__(self, results_dir: Path = RESULTS):
        import duckdb

        self.frames = {
            "SEGMENTS": add_names(pd.read_csv(results_dir / "hybrid_profiles.csv")),
            "BACKTEST": pd.read_csv(results_dir / "method_comparison.csv"),
            "K_CHOICES": pd.read_csv(results_dir / "k_selection.csv"),
            "CLEANING": pd.read_csv(results_dir / "cleaning_audit.csv"),
        }
        self._con = duckdb.connect(":memory:")
        for name, df in self.frames.items():
            # Real tables, not registered views: registrations are invisible to the
            # per-query cursors that keep concurrent Streamlit sessions apart.
            self._con.register("_frame", df)
            self._con.execute(f"CREATE TABLE {name} AS SELECT * FROM _frame")
            self._con.unregister("_frame")
        # Generated SQL may only see the registered tables: no files, URLs or extensions.
        self._con.execute("SET enable_external_access = false")
        self._con.execute("SET lock_configuration = true")

    def columns(self, table: str) -> list[tuple[str, str]]:
        return [(c, str(t)) for c, t in self.frames[table].dtypes.items()]

    def query(self, sql: str) -> pd.DataFrame:
        return self._con.cursor().execute(validate_sql(sql)).df().head(MAX_ROWS)

    def values(self, table: str, column: str) -> list:
        return sorted(self.frames[table][column].dropna().unique().tolist())[:MAX_HINT_VALUES + 1]

    def segments(self) -> pd.DataFrame:
        return self.frames["SEGMENTS"].copy()

    @property
    def tables(self) -> list[str]:
        return list(self.frames)


class SnowflakeData:
    """Same tables as LocalData as temporary views, plus customer-level data.

    Use a role with read-only access to CUSTOMER_SEGMENTATION in production; validate_sql is
    a guard rail, not a permission system.
    """

    mode = "live"
    dialect = "Snowflake"

    def __init__(self, session):
        self.session = session
        ml = "CUSTOMER_SEGMENTATION.ML"
        session.use_schema(ml)
        tier = case_sql("SPLIT_PART(SEGMENT, '-', 1)", TIER_NAMES)
        behavior = case_sql("SPLIT_PART(SEGMENT, '-', 2)", BEHAVIOR_NAMES)
        views = {
            "SEGMENTS": f"""
                SELECT SEGMENT, {tier} || ' / ' || {behavior} AS SEGMENT_NAME,
                       SPLIT_PART(SEGMENT, '-', 1) AS VALUE_TIER, {tier} AS VALUE_TIER_NAME,
                       SPLIT_PART(SEGMENT, '-', 2) AS BEHAVIOR_SEGMENT, {behavior} AS BEHAVIOR_NAME,
                       {", ".join(SEGMENT_COLUMNS[6:])}
                FROM {ml}.HYBRID_PROFILES WHERE RUN_ID = 'production'""",
            "BACKTEST": f"SELECT * FROM {ml}.METHOD_COMPARISON WHERE RUN_ID LIKE 'backtest_%'",
            "K_CHOICES": f"SELECT * FROM {ml}.K_SELECTION WHERE RUN_ID = 'production'",
            "CLEANING": "SELECT * FROM CUSTOMER_SEGMENTATION.STAGING.CLEANING_AUDIT",
            "CUSTOMERS": f"SELECT * FROM {ml}.CUSTOMER_SEGMENTS_CURRENT",
            "MIGRATION": f"SELECT * FROM {ml}.SEGMENT_MIGRATION",
        }
        for name, sql in views.items():
            session.sql(f"CREATE OR REPLACE TEMPORARY VIEW {name} AS {sql}").collect()
        self.tables = list(views)

    def columns(self, table: str) -> list[tuple[str, str]]:
        return [(f.name, str(f.datatype)) for f in self.session.table(table).schema.fields]

    def query(self, sql: str) -> pd.DataFrame:
        return self.session.sql(validate_sql(sql)).limit(MAX_ROWS).to_pandas()

    def values(self, table: str, column: str) -> list:
        rows = self.session.sql(f'SELECT DISTINCT "{column}" AS V FROM {table} '
                                f'WHERE "{column}" IS NOT NULL ORDER BY 1 LIMIT {MAX_HINT_VALUES + 1}').collect()
        return [r["V"] for r in rows]

    def segments(self) -> pd.DataFrame:
        return self.query("SELECT * FROM SEGMENTS ORDER BY SEGMENT")


def _is_text(dtype: str) -> bool:
    return dtype in ("object", "str", "string") or dtype.startswith("StringType")


def schema_text(data) -> str:
    """Table and column reference for the SQL-writing prompt.

    Text columns with few distinct values list them, so the model filters on real values
    (BEHAVIOR_NAME = 'Holiday buyers') instead of guessing. Built once per data source.
    """
    cached = getattr(data, "_schema_text", None)
    if cached:
        return cached
    parts = []
    for table in data.tables:
        cols, hints = [], []
        for c, t in data.columns(table):
            cols.append(f"{c} {t}")
            if _is_text(t) and c not in NO_HINT_COLUMNS:
                values = data.values(table, c)
                if len(values) <= MAX_HINT_VALUES:
                    hints.append(f"{c} in (" + ", ".join(f"'{v}'" for v in values) + ")")
        text = f"{table}: {TABLE_NOTES[table]}\n  Columns: {', '.join(cols)}"
        if hints:
            text += "\n  Values: " + "; ".join(hints)
        parts.append(text)
    data._schema_text = "\n\n".join(parts)
    return data._schema_text


def open_data(mode: str | None = None):
    """ASSISTANT_MODE=local|live|auto (auto = live when Snowflake credentials are configured)."""
    mode = (mode or os.getenv("ASSISTANT_MODE", "auto")).lower()
    if mode == "auto":
        from dotenv import load_dotenv

        load_dotenv(os.getenv("SNOWFLAKE_ENV_FILE", ROOT / ".env"))
        mode = "live" if os.getenv("SNOWFLAKE_ACCOUNT") else "local"
    if mode == "local":
        return LocalData()
    if mode == "live":
        from common import get_session

        return SnowflakeData(get_session())
    raise ValueError(f"ASSISTANT_MODE must be local, live or auto, not {mode!r}")
