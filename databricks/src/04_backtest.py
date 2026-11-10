# Databricks notebook source
# MAGIC %md
# MAGIC # 04 Backtest: two out-of-time folds
# MAGIC
# MAGIC Each fold fits both methods on one snapshot and judges them on the next quarter's snapshot against
# MAGIC what customers actually did over the following 91 days. The modeling code is `segmentation/`,
# MAGIC unchanged from the Snowflake build; `DatabricksSession` stands in for the Snowpark session.

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

import sklearn

from segmentation import procedures as p
from segmentation.databricks_session import DatabricksSession
from segmentation.production import FOLDS

print("scikit-learn", sklearn.__version__)
session = DatabricksSession(spark, fq)

# Derived output: rebuilt from scratch so its columns always match the current code.
spark.sql(f"DROP TABLE IF EXISTS {fq}.METHOD_COMPARISON")

for run_id, (train, eval_) in FOLDS.items():
    two_axis = json.loads(p.train_two_axis(session, run_id, train, eval_))
    k_value, k_grid = two_axis["k_value"], two_axis["k_value"] * two_axis["k_behavior"]
    print(run_id, two_axis)
    print(p.train_outcome_guided(session, run_id, "outcome_guided", train, eval_, k_value))
    print(p.train_outcome_guided(session, run_id, "outcome_guided_grid_size", train, eval_, k_grid))
    print(p.build_hybrid(session, run_id, f"{train},{eval_}"))
    print(p.compare_methods(session, run_id, train, eval_))

# COMMAND ----------

display(spark.sql(f"""
    SELECT run_id, scheme, n_segments, next_active_auc, top20_revenue_capture, behavior_eta2, persistence
    FROM {fq}.METHOD_COMPARISON ORDER BY scheme, run_id
"""))
