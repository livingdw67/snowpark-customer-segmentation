# Databricks notebook source
# MAGIC %md
# MAGIC # 05 Production: fit on the latest snapshot with known outcomes, score "today"
# MAGIC
# MAGIC Builds the hybrid segments, the `CUSTOMER_SEGMENTS_CURRENT` view, the segment map and the
# MAGIC quarter-to-quarter migration table.

# COMMAND ----------

# MAGIC %pip install scikit-learn==1.9.1 numpy==2.4.6 scipy==1.17.1 pandas==2.3.3 joblib==1.5.3 --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "segmentation")
dbutils.widgets.text("code_root", "")
fq = f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}"
code_root = dbutils.widgets.get("code_root")

import json
import os
import sys

os.environ["SEGMENTATION_SCHEMA"] = fq
sys.path.insert(0, code_root if code_root.startswith("/Workspace") else f"/Workspace{code_root}")

# COMMAND ----------

from segmentation import procedures as p
from segmentation.databricks_session import DatabricksSession
from segmentation.production import (ALL_SNAPSHOTS, FIT, MIGRATION_SQL, RUN_ID, TODAY, check_behavior_names,
                                     current_view_sql)

session = DatabricksSession(spark, fq)

two_axis = json.loads(p.train_two_axis(session, RUN_ID, FIT, ALL_SNAPSHOTS))
k_value, k_grid = two_axis["k_value"], two_axis["k_value"] * two_axis["k_behavior"]
print(two_axis)
print(p.train_outcome_guided(session, RUN_ID, "outcome_guided", FIT, TODAY, k_value))
print(p.train_outcome_guided(session, RUN_ID, "outcome_guided_grid_size", FIT, TODAY, k_grid))
print(p.build_hybrid(session, RUN_ID, f"{FIT},{TODAY}"))
check_behavior_names(session)
spark.sql(current_view_sql())
print(p.build_segment_map(session, RUN_ID, TODAY, 2000))
spark.sql(MIGRATION_SQL)

# COMMAND ----------

display(spark.sql(f"""
    SELECT segment_code, segment_name, COUNT(*) AS customers,
           ROUND(SUM(expected_revenue_next_quarter)) AS expected_revenue
    FROM {fq}.CUSTOMER_SEGMENTS_CURRENT GROUP BY 1, 2 ORDER BY 1
"""))
