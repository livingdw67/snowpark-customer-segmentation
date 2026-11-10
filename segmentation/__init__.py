"""Customer segmentation code that runs in the data platform, on Databricks or Snowflake.

On Databricks the notebooks in databricks/src import it directly; on Snowflake,
05_deploy_procedures.py deploys it as stored procedures. The modeling modules are only imported
there, so scikit-learn is only ever loaded in the platform's Python runtime; the SQL and naming
modules are plain Python and are also imported locally.
"""
