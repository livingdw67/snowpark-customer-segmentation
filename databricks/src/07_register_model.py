# Databricks notebook source
# MAGIC %md
# MAGIC # 07 Register: the production segmenter in Unity Catalog, as challenger
# MAGIC
# MAGIC Logs the production run's two fitted models as one MLflow pyfunc, `<schema>.hybrid_segmenter`, with the settings
# MAGIC the run chose and the hybrid's backtest results, and gives the new version the alias `challenger`. Task 06 is the
# MAGIC gate: it moves `champion` to this version only if the run reproduces the committed baseline.

# COMMAND ----------

# MAGIC %pip install scikit-learn==1.9.1 numpy==2.4.6 scipy==1.17.1 pandas==2.3.3 joblib==1.5.3 --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "segmentation")
dbutils.widgets.text("code_root", "")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
fq = f"{catalog}.{schema}"
code_root = dbutils.widgets.get("code_root")

import os
import sys

os.environ["SEGMENTATION_SCHEMA"] = fq
root = code_root if code_root.startswith("/Workspace") else f"/Workspace{code_root}"
sys.path.insert(0, root)

# COMMAND ----------

import joblib
import mlflow
import pandas as pd
import sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

from segmentation.databricks_session import DatabricksSession
from segmentation.features import load_snapshots
from segmentation.production import FIT, RUN_ID, TODAY
from segmentation.registry import HybridSegmenter
from segmentation.tables import REGISTERED_MODEL

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/customer-segmentation")

models = f"/Volumes/{catalog}/{schema}/models/{RUN_ID}"
artifacts = {"two_axis": f"{models}/two_axis.joblib", "outcome_guided": f"{models}/outcome_guided.joblib"}
local = HybridSegmenter()  # the same models in memory, for the signature and the logged settings
local.two_axis, local.outcome = (joblib.load(path) for path in artifacts.values())
sample = load_snapshots(DatabricksSession(spark, fq), [TODAY]).head(20)
backtest = spark.sql(f"""
    SELECT AVG(TOP20_REVENUE_CAPTURE) AS capture, AVG(NEXT_ACTIVE_AUC) AS auc, AVG(BEHAVIOR_ETA2) AS eta2
    FROM {fq}.METHOD_COMPARISON WHERE SCHEME = 'hybrid'""").first()

with mlflow.start_run(run_name="production segmenter"):
    mlflow.log_params({"fit_snapshot": FIT, "scored_snapshot": TODAY, "value_tiers": len(local.outcome.rules_),
                       "behavior_segments": local.two_axis.k_behavior_, "revenue_rounds": local.outcome.revenue_rounds_,
                       "active_rounds": local.outcome.active_rounds_, "scikit_learn": sklearn.__version__})
    mlflow.log_metrics({"backtest_top20_revenue_capture": backtest.capture, "backtest_next_active_auc": backtest.auc,
                        "backtest_behavior_eta2": backtest.eta2})
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=HybridSegmenter(),
        artifacts=artifacts,
        code_paths=[f"{root}/segmentation"],
        signature=infer_signature(sample, local.predict(None, sample)),
        pip_requirements=[f"scikit-learn=={sklearn.__version__}", f"pandas=={pd.__version__}", "numpy==2.4.6",
                          "scipy==1.17.1", "joblib==1.5.3"],
        registered_model_name=REGISTERED_MODEL,
    )

version = info.registered_model_version
MlflowClient().set_registered_model_alias(REGISTERED_MODEL, "challenger", version)
print(f"Registered {REGISTERED_MODEL} v{version} as challenger")
