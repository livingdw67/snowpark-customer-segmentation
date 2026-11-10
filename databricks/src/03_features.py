# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Features: one row per customer per quarterly snapshot
# MAGIC
# MAGIC Each snapshot looks back 365 days for behavior features and forward 91 days for outcomes. Only
# MAGIC lines before the snapshot date feed the features. The last snapshot (2011-12-10) has no outcomes;
# MAGIC it's "today" for the dataset. The SQL is in `segmentation/feature_sql.py`.

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

from segmentation.feature_sql import FEATURE_TABLE, create_features_sql

SNAPSHOTS = ["2010-12-10", "2011-03-10", "2011-06-10", "2011-09-10", "2011-12-10"]
LAST_DATA_DATE = "2011-12-09"

spark.sql(create_features_sql(SNAPSHOTS, LAST_DATA_DATE))
display(spark.sql(f"""
    SELECT snapshot_date, COUNT(*) AS customers,
           ROUND(AVG(CAST(next_active AS INT)), 3) AS next_active_rate,
           ROUND(MEDIAN(net_revenue)) AS median_net_revenue,
           ROUND(MEDIAN(orders), 1) AS median_orders
    FROM {FEATURE_TABLE} GROUP BY 1 ORDER BY 1
"""))

# COMMAND ----------

import json

rows = spark.sql(f"SELECT CAST(snapshot_date AS STRING) AS d, COUNT(*) AS n FROM {FEATURE_TABLE} GROUP BY 1 ORDER BY 1").collect()
dbutils.notebook.exit(json.dumps({r["d"]: r["n"] for r in rows}))
