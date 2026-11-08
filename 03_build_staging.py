"""Clean RAW.TRANSACTIONS into STAGING.TRANSACTIONS and record what each rule removed.

The rules and the SQL are in segmentation/staging_sql.py, shared with the Databricks build.
"""
from common import get_session, run_sql
from segmentation.staging_sql import AUDIT, AUDIT_SQL, STAGING_SQL, STG


def build_staging():
    session = get_session()
    run_sql(session, STAGING_SQL)
    run_sql(session, AUDIT_SQL)
    print(session.table(AUDIT).sort("STEP").to_pandas().to_string(index=False))
    print(session.sql(f"""
        SELECT line_type, COUNT(*) AS lines, ROUND(SUM(line_amount)) AS amount,
               COUNT_IF(customer_id IS NULL) AS lines_without_customer
        FROM {STG} GROUP BY 1 ORDER BY 1
    """).to_pandas().to_string(index=False))
    session.close()


if __name__ == "__main__":
    build_staging()
