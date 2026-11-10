# Databricks notebook source
# MAGIC %md
# MAGIC # 08 Monthly re-score with the champion model
# MAGIC
# MAGIC The Databricks version of the Snowflake `RESCORE_SEGMENTS_MONTHLY` Task. It rebuilds the customer features for the
# MAGIC snapshot after the latest transaction, scores them with the `champion` version of `<schema>.hybrid_segmenter`, and
# MAGIC writes hybrid rows under `run_id = 'rescore'`, which `CUSTOMER_SEGMENTS_CURRENT` prefers. The data is static, so
# MAGIC every run re-derives 2011-12-10, and the result reports how many customers got the same segment as the
# MAGIC production run: all of them should.

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

import os
import sys

os.environ["SEGMENTATION_SCHEMA"] = fq
sys.path.insert(0, code_root if code_root.startswith("/Workspace") else f"/Workspace{code_root}")

# COMMAND ----------

import json

import mlflow
from mlflow import MlflowClient

from segmentation import procedures as p
from segmentation.databricks_session import DatabricksSession
from segmentation.tables import REGISTERED_MODEL

mlflow.set_registry_uri("databricks-uc")
version = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL, "champion").version
champion = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@champion").unwrap_python_model()
result = json.loads(p.rescore_customers(DatabricksSession(spark, fq), "production",
                                        models=(champion.two_axis, champion.outcome)))
result["model"] = f"{REGISTERED_MODEL} v{version} (champion)"
print(json.dumps(result, indent=1))

# COMMAND ----------

dbutils.notebook.exit(json.dumps(result))
