"""Write the published segment-level tables in results/ from the Databricks build. Uses a SQL warehouse, no job compute.

    python databricks/export_results.py [--profile free]

Run it after the pipeline job. Only aggregated tables leave the workspace. cleaning_audit.csv isn't rewritten: it
depends only on the data and the SQL, and the reproduction check holds it to an exact match. The evals keep their
own frozen copy of the earlier Snowflake tables in practice_eval/frozen_results/.
"""
import argparse
import sys
from pathlib import Path

from databricks.sdk import WorkspaceClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from assistant.data import read_statement  # noqa: E402

SCHEMA = "workspace.segmentation"
# (file, query, decimal places by column; columns not named are written as they come)
EXPORTS = [
    ("hybrid_profiles", f"SELECT * FROM {SCHEMA}.HYBRID_PROFILES WHERE RUN_ID = 'production' ORDER BY SEGMENT",
     {"SHARE": 3, "UK_SHARE": 3, "MEAN_P_ACTIVE": 3, "EXPECTED_REVENUE": 3, "MEDIAN_RETURN_RATE": 3,
      "MEDIAN_Q4_REVENUE_SHARE": 3, "MEDIAN_AVG_UNIT_PRICE": 3, "MEDIAN_NET_REVENUE": 2, "MEDIAN_AVG_UNITS_PER_ORDER": 2,
      "MEDIAN_AVG_LINES_PER_ORDER": 2, "MEDIAN_AVG_ORDER_VALUE": 2}),
    ("method_comparison", f"SELECT * FROM {SCHEMA}.METHOD_COMPARISON ORDER BY SCHEME, RUN_ID", 4),
    ("k_selection", f"SELECT * FROM {SCHEMA}.K_SELECTION WHERE RUN_ID = 'production' ORDER BY AXIS DESC, K", 4),
    ("segment_migration", f"""SELECT from_segment, to_segment, SUM(customers) AS customers FROM {SCHEMA}.SEGMENT_MIGRATION
                              WHERE to_snapshot = '2011-12-10' GROUP BY 1, 2 ORDER BY 1, 2""", {}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="free")
    args = parser.parse_args()
    client = WorkspaceClient(profile=args.profile)
    warehouse_id = next(iter(client.warehouses.list())).id
    for name, sql, decimals in EXPORTS:
        frame = read_statement(client, warehouse_id, sql)
        frame = frame.round(decimals)
        path = ROOT / "results" / f"{name}.csv"
        with open(path, "w", newline="\n", encoding="utf-8") as f:
            frame.to_csv(f, index=False)
        print(f"wrote {path.relative_to(ROOT).as_posix()}: {len(frame)} rows")


if __name__ == "__main__":
    main()
