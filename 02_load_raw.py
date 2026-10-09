"""Create the database and schemas, stage the parquet file, and COPY it into RAW.TRANSACTIONS."""
import os

from common import DATABASE, PROJECT_DIR, get_session, run_sql

PARQUET = os.path.join(PROJECT_DIR, "data", "raw", "online_retail_ii.parquet").replace("\\", "/")


def load_raw():
    session = get_session()
    run_sql(session, f"CREATE DATABASE IF NOT EXISTS {DATABASE}")
    for schema in ("RAW", "STAGING", "MARTS", "ML"):
        run_sql(session, f"CREATE SCHEMA IF NOT EXISTS {DATABASE}.{schema}")
    run_sql(session, f"USE SCHEMA {DATABASE}.RAW")
    run_sql(session, "CREATE STAGE IF NOT EXISTS retail_stage FILE_FORMAT = (TYPE = PARQUET)")

    print("Uploading parquet to stage...")
    session.file.put(f"file://{PARQUET}", "@retail_stage", auto_compress=False, overwrite=True)

    run_sql(session, """
        CREATE OR REPLACE TABLE TRANSACTIONS (
            INVOICE       VARCHAR,
            STOCKCODE     VARCHAR,
            DESCRIPTION   VARCHAR,
            QUANTITY      NUMBER(38, 0),
            INVOICEDATE   TIMESTAMP_NTZ,
            PRICE         NUMBER(12, 2),
            CUSTOMERID    NUMBER(38, 0),
            COUNTRY       VARCHAR,
            SOURCESHEET   VARCHAR
        )
    """)
    # pandas writes nanosecond timestamps, so INVOICEDATE needs USE_LOGICAL_TYPE to load correctly.
    result = run_sql(session, """
        COPY INTO TRANSACTIONS
        FROM @retail_stage/online_retail_ii.parquet
        FILE_FORMAT = (TYPE = PARQUET USE_LOGICAL_TYPE = TRUE)
        MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
    """)
    print(result[0])

    n = run_sql(session, "SELECT COUNT(*) AS N FROM TRANSACTIONS")[0]["N"]
    print(f"RAW.TRANSACTIONS: {n:,} rows")
    session.close()


if __name__ == "__main__":
    load_raw()
