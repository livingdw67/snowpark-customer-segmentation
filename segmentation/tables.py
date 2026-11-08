"""Where the pipeline's tables live, on either platform.

Snowflake uses one database with a schema per layer. Databricks uses one Unity Catalog schema, so
the layer goes in the table name. Setting SEGMENTATION_SCHEMA (catalog.schema) selects Databricks;
it must be set before anything in this package is imported.
"""
import os

DATABASE = "CUSTOMER_SEGMENTATION"
SCHEMA = os.getenv("SEGMENTATION_SCHEMA")
IS_DATABRICKS = bool(SCHEMA)

if IS_DATABRICKS:
    ML = SCHEMA
    RAW_TABLE = f"{SCHEMA}.raw_transactions"
    STAGING = f"{SCHEMA}.stg_transactions"
    CLEANING_AUDIT = f"{SCHEMA}.cleaning_audit"
    FEATURE_TABLE = f"{SCHEMA}.customer_features"
else:
    ML = f"{DATABASE}.ML"
    RAW_TABLE = f"{DATABASE}.RAW.TRANSACTIONS"
    STAGING = f"{DATABASE}.STAGING.TRANSACTIONS"
    CLEANING_AUDIT = f"{DATABASE}.STAGING.CLEANING_AUDIT"
    FEATURE_TABLE = f"{DATABASE}.MARTS.CUSTOMER_FEATURES"
