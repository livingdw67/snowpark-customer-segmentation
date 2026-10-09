"""Customer segmentation code that runs inside Snowflake stored procedures.

Deployed by 05_deploy_procedures.py as a zip on a stage. The modeling modules are only
imported there, so scikit-learn is only ever loaded in Snowflake's Python runtime; naming.py
and feature_sql.py are plain Python and are also imported locally.
"""
