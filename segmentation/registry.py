"""The production segmenter in Unity Catalog: one MLflow model, a challenger alias, and a gate before champion.

Each production run registers its two fitted models (option B's behavior axis and option C's value tiers) as one
pyfunc version with the alias `challenger`. The reproduction check is the gate: only a run that reproduces the
committed baseline moves `champion` to its version. The monthly re-score loads `champion`, so a model that fails
the check never scores customers.

Imported only on Databricks: it needs mlflow, which the Snowflake build doesn't use.
"""
import joblib
import mlflow
import pandas as pd

from .procedures import hybrid_assignments


class HybridSegmenter(mlflow.pyfunc.PythonModel):
    """C's value tier crossed with B's buying style, plus the outcome model's scores, for one snapshot of features."""

    def load_context(self, context):
        self.two_axis = joblib.load(context.artifacts["two_axis"])
        self.outcome = joblib.load(context.artifacts["outcome_guided"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        return hybrid_assignments(self.two_axis, self.outcome, model_input).reset_index(drop=True)
