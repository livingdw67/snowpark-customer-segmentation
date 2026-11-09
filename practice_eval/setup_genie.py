"""Put a frozen copy of the eval tables and a Genie space on Databricks, ported from the Cortex Analyst model.

The 175 held-out questions' truths were computed from results/*.csv, and Cortex Analyst answered over those
same numbers. So Genie reads a frozen copy of them in its own schema, not the pipeline's tables, which a rebaseline
can change. Everything Genie is told comes mechanically from semantic_model/segmentation_r8.yaml, the model Analyst
used on these questions:

    field expressions            -> columns of the frozen tables, under the model's logical names
    descriptions and synonyms    -> table and column comments
    custom_instructions, metrics -> the space's text instructions
    verified_queries             -> the space's example SQL

Safe to rerun: it replaces the tables and updates the space. Needs databricks-sdk, pandas and pyyaml.

    python setup_genie.py [--profile free] [--tables-only]
"""
import argparse
import json
import math
import os
import re
import sys
import time
import uuid

import pandas as pd
import yaml
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from assistant.data import add_names  # noqa: E402

CATALOG, SCHEMA = "workspace", "segmentation_eval"
SPACE_TITLE = "Segment Analyst rematch (frozen eval data)"
MODEL = os.path.join(HERE, "semantic_model", "segmentation_r8.yaml")
RESULTS = os.path.join(HERE, "frozen_results")  # the published Snowflake tables the truths came from


def frozen_frames() -> dict[str, pd.DataFrame]:
    """The tables under Analyst's base-table names, as the Assistant reads them."""
    return {"SEGMENTS": add_names(pd.read_csv(os.path.join(RESULTS, "hybrid_profiles.csv"))),
            "BACKTEST": pd.read_csv(os.path.join(RESULTS, "method_comparison.csv")),
            "K_CHOICES": pd.read_csv(os.path.join(RESULTS, "k_selection.csv")),
            "CLEANING": pd.read_csv(os.path.join(RESULTS, "cleaning_audit.csv"))}


def table_frame(spec: dict, frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[dict]]:
    """Evaluate the model's field expressions: one column per dimension or fact, named as in the model."""
    source = frames[spec["base_table"]["table"]]
    fields = spec.get("dimensions", []) + spec.get("facts", [])
    return pd.DataFrame({f["name"]: source.eval(f["expr"]) for f in fields}), fields


def spark_type(field: dict, values: pd.Series) -> str:
    kind = field["data_type"].upper()
    if kind == "VARCHAR":
        return "STRING"
    if kind == "BOOLEAN":
        return "BOOLEAN"
    return "BIGINT" if (values.dropna() % 1 == 0).all() else "DOUBLE"


def quoted(text: str) -> str:
    return "'" + str(text).replace("\\", "\\\\").replace("'", "\\'") + "'"


def literal(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NULL"
    if kind == "STRING":
        return quoted(value)
    if kind == "BOOLEAN":
        return "TRUE" if str(value).lower() == "true" else "FALSE"
    return str(int(value)) if kind == "BIGINT" else repr(float(value))


def comment(field: dict) -> str:
    text = field.get("description", "")
    if field.get("synonyms"):
        text = f"{text} Also called: {', '.join(field['synonyms'])}.".strip()
    return text


def table_sql(fq: str, spec: dict, df: pd.DataFrame, fields: list[dict]) -> list[str]:
    kinds = {f["name"]: spark_type(f, df[f["name"]]) for f in fields}
    columns = ",\n  ".join(f"`{n}` {k} COMMENT {quoted(comment(f))}" for f, (n, k) in zip(fields, kinds.items()))
    rows = ",\n".join("(" + ", ".join(literal(v, kinds[c]) for c, v in zip(df.columns, row)) + ")"
                      for row in df.astype(object).itertuples(index=False))
    return [f"CREATE OR REPLACE TABLE {fq}.{spec['name']} (\n  {columns}\n) COMMENT {quoted(' '.join(spec['description'].split()))}",
            f"INSERT INTO {fq}.{spec['name']} VALUES\n{rows}"]


def instructions(model: dict) -> str:
    """Analyst's custom instructions, plus the model's named metrics, which Genie has no field for."""
    metrics = [f"- {m['name']} = {m['expr']}: {m.get('description', '').strip()}"
               for t in model["tables"] for m in t.get("metrics", [])]
    text = " ".join(model["custom_instructions"].split())
    return f"{text}\n\nNamed measures on the segments table:\n" + "\n".join(metrics)


def example_sql(model: dict, fq: str) -> list[tuple[str, str]]:
    return [(v["question"], re.sub(r"\b__(\w+)\b", rf"{fq}.\1", v["sql"])) for v in model["verified_queries"]]


def serialized_space(model: dict, fq: str) -> str:
    examples = example_sql(model, fq)
    ids = sorted(uuid.uuid4().hex for _ in range(len(examples) + 1))  # the API wants each list sorted by id
    return json.dumps({
        "version": 2,
        "data_sources": {"tables": [{"identifier": f"{fq}.{name}"} for name in sorted(t["name"] for t in model["tables"])]},
        "instructions": {
            "text_instructions": [{"id": ids[0], "content": [instructions(model)]}],
            "example_question_sqls": [{"id": i, "question": [q], "sql": [s]} for i, (q, s) in zip(ids[1:], examples)],
        },
    })


def run_sql(w: WorkspaceClient, warehouse_id: str, sql: str) -> None:
    r = w.statement_execution.execute_statement(statement=sql, warehouse_id=warehouse_id, wait_timeout="50s")
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(1)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"{sql.splitlines()[0]}: {r.status.error.message if r.status.error else r.status.state}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="free")
    parser.add_argument("--tables-only", action="store_true", help="build the tables and stop")
    args = parser.parse_args()

    w = WorkspaceClient(profile=args.profile)
    warehouse_id = next(iter(w.warehouses.list())).id
    with open(MODEL, encoding="utf-8") as f:
        model = yaml.safe_load(f)
    fq = f"{CATALOG}.{SCHEMA}"
    frames = frozen_frames()

    run_sql(w, warehouse_id, f"CREATE SCHEMA IF NOT EXISTS {fq} COMMENT "
                             + quoted("Frozen copy of the published segmentation results, for evaluating Genie."))
    for spec in model["tables"]:
        df, fields = table_frame(spec, frames)
        for sql in table_sql(fq, spec, df, fields):
            run_sql(w, warehouse_id, sql)
        print(f"{fq}.{spec['name']}: {len(df)} rows, {len(fields)} columns")
    if args.tables_only:
        return

    description = "Segment-level results of a customer segmentation, frozen for evaluation. No customer-level rows."
    match = next((s for s in (w.genie.list_spaces().spaces or []) if s.title == SPACE_TITLE), None)
    body = {"warehouse_id": warehouse_id, "serialized_space": serialized_space(model, fq),
            "title": SPACE_TITLE, "description": description}
    # The REST calls, because not every SDK version has create_space and update_space.
    if match:
        space = w.api_client.do("PATCH", f"/api/2.0/genie/spaces/{match.space_id}", body=body)
    else:
        space = w.api_client.do("POST", "/api/2.0/genie/spaces", body=body)
    print(f"Genie space '{SPACE_TITLE}': {space['space_id']}")


if __name__ == "__main__":
    main()
