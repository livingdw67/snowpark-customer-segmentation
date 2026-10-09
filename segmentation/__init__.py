"""Customer segmentation code that runs inside Snowflake stored procedures.

Deployed by 05_deploy_procedures.py as a zip on a stage; nothing here is imported locally,
so scikit-learn is only ever loaded in Snowflake's Python runtime.
"""
