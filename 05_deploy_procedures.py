"""Zip the segmentation package to a stage and create the permanent stored procedures.

The monthly re-scoring Task is created separately from sql/rescore_task.sql.

All training, scoring and evaluation runs inside Snowflake through these procedures; the
local machine only issues CALL statements.
"""
import io
import os
import zipfile

from common import DATABASE, PROJECT_DIR, PYTHON_RUNTIME, SPROC_PACKAGES, get_session, run_sql

ML = f"{DATABASE}.ML"
PACKAGE_DIR = os.path.join(PROJECT_DIR, "segmentation")

PROCEDURES = {
    "TRAIN_TWO_AXIS": ("run_id STRING, fit_date STRING, score_dates STRING",
                       "segmentation.procedures.train_two_axis"),
    "TRAIN_OUTCOME_GUIDED": ("run_id STRING, method STRING, fit_date STRING, score_dates STRING, "
                             "n_segments INT", "segmentation.procedures.train_outcome_guided"),
    "BUILD_HYBRID": ("run_id STRING, snapshot_dates STRING",
                     "segmentation.procedures.build_hybrid"),
    "COMPARE_METHODS": ("run_id STRING, train_date STRING, eval_date STRING",
                        "segmentation.procedures.compare_methods"),
    "BUILD_SEGMENT_MAP": ("run_id STRING, snapshot_date STRING, n_sample INT",
                          "segmentation.procedures.build_segment_map"),
    # Called monthly by the Task in sql/rescore_task.sql.
    "RESCORE_CUSTOMERS": ("model_run_id STRING DEFAULT 'production'",
                          "segmentation.procedures.rescore_customers"),
}


def zip_package() -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(os.listdir(PACKAGE_DIR)):
            if name.endswith(".py"):
                zf.write(os.path.join(PACKAGE_DIR, name), f"segmentation/{name}")
    buf.seek(0)
    return buf


def deploy():
    session = get_session()
    run_sql(session, f"CREATE STAGE IF NOT EXISTS {ML}.CODE")
    run_sql(session, f"CREATE STAGE IF NOT EXISTS {ML}.MODELS")
    session.file.put_stream(zip_package(), f"@{ML}.CODE/segmentation.zip",
                            auto_compress=False, overwrite=True)

    packages = ", ".join(f"'{p}'" for p in SPROC_PACKAGES)
    for name, (args, handler) in PROCEDURES.items():
        run_sql(session, f"""
            CREATE OR REPLACE PROCEDURE {ML}.{name}({args})
            RETURNS STRING
            LANGUAGE PYTHON
            RUNTIME_VERSION = '{PYTHON_RUNTIME}'
            PACKAGES = ({packages})
            IMPORTS = ('@{ML}.CODE/segmentation.zip')
            HANDLER = '{handler}'
        """)
        print(f"Created {ML}.{name}")
    session.close()


if __name__ == "__main__":
    deploy()
