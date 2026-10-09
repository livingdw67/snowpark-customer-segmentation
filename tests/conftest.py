"""Synthetic customers with known structure, so tests can check the methods recover it."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_customers(n=600, snapshot_date="2011-09-10", seed=0) -> pd.DataFrame:
    """Three buying styles (bulk / holiday / small-basket) crossed with a value level.

    Value drives the outcomes; buying style does not. That mirrors the real data closely
    enough to test that each method separates what it should.
    """
    rng = np.random.default_rng(seed)
    style = rng.integers(0, 3, n)
    value = rng.gamma(2.0, 1.0, n)  # latent customer value

    orders = np.maximum(1, np.round(value * 3 + rng.normal(0, 0.5, n))).astype(int)
    df = pd.DataFrame({
        "customer_id": [f"c{seed}_{i}" for i in range(n)],
        "snapshot_date": snapshot_date,
        "recency_days": np.clip(300 / (1 + value) + rng.normal(0, 10, n), 1, 365),
        "orders": orders,
        "purchase_days": orders,
        "active_months": np.minimum(orders, 12),
        "avg_units_per_order": np.where(style == 0, 300, np.where(style == 1, 180, 60))
                               * rng.lognormal(0, 0.15, n),
        "avg_unit_price": np.where(style == 2, 2.6, 1.6) * rng.lognormal(0, 0.1, n),
        "avg_lines_per_order": np.where(style == 2, 8, 20) * rng.lognormal(0, 0.2, n),
        "return_rate": np.clip(rng.normal(0.01, 0.01, n), 0, 1),
        "q4_revenue_share": np.where(style == 1, 0.9, 0.2) + rng.normal(0, 0.05, n),
        "tenure_days": rng.uniform(100, 700, n),
        "avg_days_between_purchases": np.where(orders > 1, 365 / orders, np.nan),
        "is_uk": (rng.random(n) < 0.9).astype(int),
        "style": style,
    })
    df["q4_revenue_share"] = df["q4_revenue_share"].clip(0, 1)
    df["avg_order_value"] = df["avg_units_per_order"] * df["avg_unit_price"]
    df["net_revenue"] = df["avg_order_value"] * df["orders"]
    df["distinct_products"] = np.round(df["avg_lines_per_order"] * np.sqrt(df["orders"]))
    df["snapshot_month"] = pd.to_datetime(df["snapshot_date"]).dt.month

    p_active = 1 / (1 + np.exp(-(value - 2)))
    df["next_active"] = rng.random(n) < p_active
    df["next_orders"] = np.where(df["next_active"], np.maximum(1, np.round(value)), 0)
    df["next_net_revenue"] = np.where(df["next_active"], value * 400 * rng.lognormal(0, 0.3, n), 0.0)
    return df


@pytest.fixture(scope="session")
def customers():
    return make_customers()


@pytest.fixture(scope="session")
def pooled_customers():
    """Two quarterly snapshots of overlapping customers, as the outcome model is trained."""
    a = make_customers(500, "2011-06-10", seed=1)
    b = make_customers(500, "2011-09-10", seed=2)
    b["customer_id"] = a["customer_id"]  # same people, next quarter
    return pd.concat([a, b], ignore_index=True)
