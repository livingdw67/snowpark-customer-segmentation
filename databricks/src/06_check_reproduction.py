# Databricks notebook source
# MAGIC %md
# MAGIC # 06 Check: does this run reproduce the committed baseline?
# MAGIC
# MAGIC Compares every table with the results committed in `results/`. What depends only on the
# MAGIC data and SQL must match exactly; fitted-model outputs get the tolerances in `segmentation/reproduce.py`,
# MAGIC which were fixed before the first Databricks run. The task fails if any check fails.
# MAGIC
# MAGIC It is also the gate before `champion`: only when every check passes does the `champion` alias of
# MAGIC `<schema>.hybrid_segmenter` move to the version task 07 registered as `challenger`.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "segmentation")
dbutils.widgets.text("code_root", "")
fq = f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}"
code_root = dbutils.widgets.get("code_root")

import os
import sys

os.environ["SEGMENTATION_SCHEMA"] = fq
root = code_root if code_root.startswith("/Workspace") else f"/Workspace{code_root}"
sys.path.insert(0, root)

# COMMAND ----------

import pandas as pd

from segmentation.reproduce import compare_cleaning, compare_k, compare_schemes, compare_segments


def table(name, where="TRUE"):
    out = spark.sql(f"SELECT * FROM {fq}.{name} WHERE {where}").toPandas()
    out.columns = [c.upper() for c in out.columns]
    return out


def committed(name):
    return pd.read_csv(f"{root}/results/{name}.csv")


checks = (compare_cleaning(table("cleaning_audit"), committed("cleaning_audit"))
          + compare_k(table("K_SELECTION", "RUN_ID = 'production'"), committed("k_selection"))
          + compare_schemes(table("METHOD_COMPARISON"), committed("method_comparison"))
          + compare_segments(table("HYBRID_PROFILES", "RUN_ID = 'production'"), committed("hybrid_profiles")))
result = pd.DataFrame(checks)
spark.createDataFrame(result).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{fq}.reproduction_check")

print(result.groupby("AREA").PASSED.agg(["sum", "count"]).to_string())
failed = result[~result.PASSED]
print(failed.to_string(index=False) if len(failed) else "Every check passed.")
summary = {area: f"{int(g.PASSED.sum())}/{len(g)}" for area, g in result.groupby("AREA")}

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from segmentation.tables import REGISTERED_MODEL

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
try:
    challenger = client.get_model_version_by_alias(REGISTERED_MODEL, "challenger").version
except MlflowException:  # task 07 hasn't registered one yet
    challenger = None
if challenger is None:
    summary["gate"] = "no challenger registered"
elif failed.empty:
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", challenger)
    summary["gate"] = f"v{challenger} promoted to champion"
else:
    summary["gate"] = f"v{challenger} stays challenger"
print(summary["gate"])
assert failed.empty, (f"{len(failed)} of {len(result)} reproduction checks failed ({summary}); first failures: "
                      + "; ".join(f"{r.ITEM}: {r.BASELINE} vs {r.THIS_RUN}" for r in failed.head(30).itertuples()))

# COMMAND ----------

import json

dbutils.notebook.exit(json.dumps(summary))
