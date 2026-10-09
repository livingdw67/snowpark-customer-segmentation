"""Shared Snowflake session and naming for the pipeline scripts."""
import os

from dotenv import load_dotenv
from snowflake.snowpark import Session

DATABASE = "CUSTOMER_SEGMENTATION"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Training and scoring run inside Snowflake on these exact versions (see requirements.txt).
PYTHON_RUNTIME = "3.11"
SPROC_PACKAGES = [
    "snowflake-snowpark-python[pandas]==1.55.0",
    "scikit-learn==1.9.1",
    "pandas==2.3.3",
    "numpy==2.4.6",
    "scipy==1.17.1",
    "joblib==1.5.3",
]


def get_session(schema: str = "ML") -> Session:
    """Open a Snowpark session with key-pair auth, defaulting to DATABASE.<schema>.

    Credentials come from .env in the project folder, or from the file named by
    SNOWFLAKE_ENV_FILE when the same credentials are shared with another project.
    """
    load_dotenv(os.getenv("SNOWFLAKE_ENV_FILE", os.path.join(PROJECT_DIR, ".env")))
    session = Session.builder.configs({
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "private_key_file": os.environ["SNOWFLAKE_PRIVATE_KEY_FILE"],
        "role": os.environ["SNOWFLAKE_ROLE"],
        "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
    }).create()
    # The database may not exist yet on the first run of 02_load_raw.py, which creates it.
    if session.sql(f"SHOW DATABASES LIKE '{DATABASE}'").collect():
        session.use_database(DATABASE)
        session.use_schema(schema)
    return session


def run_sql(session: Session, sql: str) -> list:
    return session.sql(sql).collect()
