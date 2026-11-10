# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Staging: clean the transactions and record what each rule removed
# MAGIC
# MAGIC The SQL is in `segmentation/staging_sql.py`, the same statements the Snowflake build ran.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "segmentation")
dbutils.widgets.text("code_root", "")
fq = f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}"
code_root = dbutils.widgets.get("code_root")

import os
import sys

os.environ["SEGMENTATION_SCHEMA"] = fq
sys.path.insert(0, code_root if code_root.startswith("/Workspace") else f"/Workspace{code_root}")

# COMMAND ----------

from segmentation.staging_sql import AUDIT, AUDIT_SQL, STAGING_SQL, STG

spark.sql(STAGING_SQL)
spark.sql(AUDIT_SQL)
display(spark.sql(f"SELECT * FROM {AUDIT} ORDER BY step"))
display(spark.sql(f"""
    SELECT line_type, COUNT(*) AS lines, ROUND(SUM(line_amount)) AS amount,
           COUNT_IF(customer_id IS NULL) AS lines_without_customer
    FROM {STG} GROUP BY 1 ORDER BY 1
"""))

# COMMAND ----------

import json

dbutils.notebook.exit(json.dumps({r["step"]: r["rows_remaining"] for r in spark.table(AUDIT).collect()}))
