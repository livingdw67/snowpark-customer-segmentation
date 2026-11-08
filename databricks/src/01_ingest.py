# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Ingest: raw transactions into a bronze Delta table
# MAGIC
# MAGIC Reads the parquet that `databricks/upload_raw.py` put in the `raw_files` volume. Rows are kept
# MAGIC exactly as published, typed as they were on Snowflake (`02_load_raw.py`), so money stays a decimal.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "segmentation")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
fq = f"{catalog}.{schema}"

# COMMAND ----------

source = f"/Volumes/{catalog}/{schema}/raw_files/online_retail_ii.parquet"
(spark.read.parquet(source)
 .selectExpr("INVOICE", "STOCKCODE", "DESCRIPTION", "CAST(QUANTITY AS BIGINT) AS QUANTITY",
             "CAST(INVOICEDATE AS TIMESTAMP_NTZ) AS INVOICEDATE", "CAST(PRICE AS DECIMAL(12, 2)) AS PRICE",
             "CAST(CUSTOMERID AS BIGINT) AS CUSTOMERID", "COUNTRY", "SOURCESHEET")
 .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{fq}.raw_transactions"))

display(spark.sql(f"SELECT sourcesheet, COUNT(*) AS rows, MIN(invoicedate) AS first, MAX(invoicedate) AS last "
                  f"FROM {fq}.raw_transactions GROUP BY 1 ORDER BY 1"))
