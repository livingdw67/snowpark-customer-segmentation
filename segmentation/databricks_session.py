"""A Snowpark-shaped session over Spark, so procedures.py runs unchanged on Databricks.

It covers only what the procedures call: sql(), table(), write_pandas() and
file.put_stream() / get_stream(). Column names come back upper-case, as they do from Snowflake.
Model files go to a Unity Catalog volume in place of the @ML.MODELS stage.
"""
from __future__ import annotations

import os
import re
import shutil

import pandas as pd

STAGE_PATH = re.compile(r"^@.*?\.MODELS/(?P<rest>.+)$", re.IGNORECASE)


def _spark_type(series: pd.Series) -> str:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    return "STRING"


def spark_ddl(df: pd.DataFrame) -> str:
    """The frame's schema, stated explicitly: Spark can't infer a type for an all-null column."""
    return ", ".join(f"`{c}` {_spark_type(df[c])}" for c in df.columns)


def records(df: pd.DataFrame) -> list[list]:
    """Rows of plain Python values, with every missing value (NaN, pd.NA, None) as None."""
    return df.astype(object).where(df.notna(), None).values.tolist()


def volume_path(root: str, stage_path: str) -> str:
    """'@db.ML.MODELS/production/two_axis.joblib' -> '<root>/production/two_axis.joblib'."""
    match = STAGE_PATH.match(stage_path)
    if not match:
        raise ValueError(f"not a model stage path: {stage_path}")
    return f"{root.rstrip('/')}/{match['rest']}"


class _Frame:
    def __init__(self, df):
        self._df = df

    def collect(self) -> list[dict]:
        return [{k.upper(): v for k, v in row.asDict().items()} for row in self._df.collect()]

    def to_pandas(self) -> pd.DataFrame:
        """Decimals come back as floats, as Snowflake's NUMBER columns do; Spark would give Python Decimals."""
        columns = [f"CAST(`{f.name}` AS DOUBLE) AS `{f.name}`" if type(f.dataType).__name__ == "DecimalType" else f"`{f.name}`"
                   for f in self._df.schema.fields]
        out = self._df.selectExpr(*columns).toPandas()
        out.columns = [c.upper() for c in out.columns]
        return out

    def filter(self, condition: str) -> "_Frame":
        return _Frame(self._df.filter(condition))


class _Files:
    def __init__(self, root: str):
        self.root = root

    def put_stream(self, stream, stage_path: str, **_) -> None:
        path = volume_path(self.root, stage_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as out:
            shutil.copyfileobj(stream, out)

    def get_stream(self, stage_path: str):
        return open(volume_path(self.root, stage_path), "rb")


class DatabricksSession:
    def __init__(self, spark, schema: str, volume: str = "models"):
        self.spark = spark
        self.schema = schema
        catalog, name = schema.split(".")
        self.file = _Files(f"/Volumes/{catalog}/{name}/{volume}")

    def sql(self, text: str) -> _Frame:
        return _Frame(self.spark.sql(text))

    def table(self, name: str) -> _Frame:
        return _Frame(self.spark.table(name))

    def table_exists(self, table: str) -> bool:
        return self.spark.catalog.tableExists(f"{self.schema}.{table}")

    def write_pandas(self, df: pd.DataFrame, table: str, database=None, schema=None,
                     auto_create_table: bool = False, use_logical_type: bool = False) -> None:
        """Append to the table, creating it if needed. Columns match by name; values take the
        existing column's type, because a method can produce nulls where another produced numbers."""
        from pyspark.sql import functions as F

        name = f"{self.schema}.{table}"
        frame = self.spark.createDataFrame(records(df), schema=spark_ddl(df))
        if self.spark.catalog.tableExists(name):
            existing = {f.name.upper(): f.dataType for f in self.spark.table(name).schema.fields}
            frame = frame.select([F.col(c).cast(existing[c.upper()]).alias(c) if c.upper() in existing else F.col(c)
                                  for c in frame.columns])
        frame.write.mode("append").option("mergeSchema", "true").saveAsTable(name)
