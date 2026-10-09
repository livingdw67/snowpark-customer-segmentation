"""Feature lists and snapshot loading shared by both segmentation methods."""
import numpy as np
import pandas as pd

DATABASE = "CUSTOMER_SEGMENTATION"
FEATURE_TABLE = f"{DATABASE}.MARTS.CUSTOMER_FEATURES"

# How much a customer is worth and how engaged they are right now.
VALUE = ["recency_days", "orders", "net_revenue"]

# How a customer buys, independent of how much: basket shape, price point, returns, seasonality.
# All are per-order or ratio features so they don't just restate volume.
BEHAVIOR = ["avg_units_per_order", "avg_unit_price", "avg_lines_per_order",
            "return_rate", "q4_revenue_share"]

# Everything the outcome model may use. avg_days_between_purchases is NULL for one-time
# buyers; HistGradientBoosting handles that natively. snapshot_month lets a model pooled over
# several quarters learn that a December-to-March quarter is quieter than September-to-December.
MODEL = VALUE + BEHAVIOR + ["purchase_days", "active_months", "avg_order_value",
                            "distinct_products", "tenure_days", "avg_days_between_purchases",
                            "is_uk", "snapshot_month"]

# Features the outcome-guided segments may split on: no NULLs, so every rule reads cleanly.
RULE = VALUE + BEHAVIOR + ["avg_order_value", "tenure_days", "active_months", "is_uk"]

# Heavy right tails; log1p before scaling.
SKEWED = {"recency_days", "orders", "net_revenue", "avg_units_per_order", "avg_unit_price",
          "avg_lines_per_order", "purchase_days", "active_months", "avg_order_value",
          "distinct_products", "tenure_days", "avg_days_between_purchases"}


def load_snapshots(session, dates) -> pd.DataFrame:
    date_list = ", ".join(f"'{d}'" for d in dates)
    df = session.sql(
        f"SELECT * FROM {FEATURE_TABLE} WHERE snapshot_date IN ({date_list})"
    ).to_pandas()
    df.columns = df.columns.str.lower()
    df["snapshot_date"] = df["snapshot_date"].astype(str)
    # A few customers returned more than they bought in the window (returns of older orders).
    df["net_revenue"] = df["net_revenue"].clip(lower=0)
    df["is_uk"] = df["is_uk"].astype(int)
    df["snapshot_month"] = pd.to_datetime(df["snapshot_date"]).dt.month
    for c in MODEL:
        df[c] = df[c].astype(float)
    return df


def log_outcome(revenue) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(revenue, dtype=float), 0, None))
