"""Create the schema and volumes, and upload the raw data for the ingest task. Uses no compute.

    python databricks/upload_raw.py

Serverless internet access is limited, so the data goes up from here (run 01_fetch_data.py first).
pandas wrote the local parquet with nanosecond timestamps, which Spark can't read, so it is
rewritten with microseconds on the way.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

SOURCE = Path(__file__).resolve().parent.parent / "data" / "raw" / "online_retail_ii.parquet"
CATALOG, SCHEMA, PROFILE = "workspace", "segmentation", "free"
CLI = shutil.which("databricks") or "databricks"


def cli(*args: str, allow_exists: bool = False) -> None:
    done = subprocess.run([CLI, *args, "--profile", PROFILE], capture_output=True, text=True)
    if done.returncode and not (allow_exists and "already exists" in (done.stderr + done.stdout).lower()):
        raise SystemExit(f"databricks {' '.join(args)} failed:\n{done.stderr or done.stdout}")


def main() -> None:
    cli("schemas", "create", SCHEMA, CATALOG, allow_exists=True)
    for volume in ("raw_files", "models"):
        cli("volumes", "create", CATALOG, SCHEMA, volume, "MANAGED", allow_exists=True)
    table = pq.read_table(SOURCE)
    with tempfile.TemporaryDirectory() as folder:
        local = Path(folder) / SOURCE.name
        pq.write_table(table, local, coerce_timestamps="us", allow_truncated_timestamps=False)
        cli("fs", "cp", "--overwrite", str(local), f"dbfs:/Volumes/{CATALOG}/{SCHEMA}/raw_files/{SOURCE.name}")
    print(f"Uploaded {table.num_rows:,} rows to /Volumes/{CATALOG}/{SCHEMA}/raw_files/{SOURCE.name}")


if __name__ == "__main__":
    main()
