import numpy as np
import pandas as pd
import pytest

from segmentation.evaluation import (capture_at, eta_squared, forecast_bias, persistence,
                                     rfm_baseline)


def test_capture_at_perfect_ranking_takes_top_revenue():
    revenue = np.array([100, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    assert capture_at(revenue, revenue, frac=0.2) == pytest.approx(1.0)


def test_capture_at_splits_ties_proportionally():
    # One segment holding everyone can only capture its proportional share.
    revenue = np.arange(10, dtype=float)
    assert capture_at(np.ones(10), revenue, frac=0.2) == pytest.approx(0.2)


def test_capture_at_ignores_negative_revenue():
    score = np.array([3, 2, 1, 0])
    revenue = np.array([10, 10, -50, 0], dtype=float)
    assert capture_at(score, revenue, frac=0.5) == pytest.approx(1.0)


def test_eta_squared_bounds():
    Z = pd.DataFrame({"x": [0.0, 0.0, 10.0, 10.0]})
    assert eta_squared(Z, ["a", "a", "b", "b"], ["x"]) == pytest.approx(1.0)
    assert eta_squared(Z, ["a", "b", "a", "b"], ["x"]) == pytest.approx(0.0)


def test_rfm_baseline_names_extremes(customers):
    best = customers.nsmallest(1, "recency_days").assign(orders=1000, net_revenue=1e7)
    worst = customers.nlargest(1, "recency_days").assign(orders=1, net_revenue=0.0)
    labels = rfm_baseline(customers, pd.concat([best, worst]))
    assert list(labels) == ["champions", "hibernating"]


def test_rfm_baseline_uses_train_edges(customers):
    # Scoring a frame of identical customers must not re-derive quintiles from that frame.
    same = pd.concat([customers.iloc[[0]]] * 5, ignore_index=True)
    assert len(set(rfm_baseline(customers, same))) == 1


def test_persistence_counts_only_shared_customers():
    assert persistence(["a", "b", "c"], ["X", "Y", "Z"], ["b", "c", "d"], ["Y", "Q", "Z"]) == 0.5


def test_forecast_bias():
    ev = pd.DataFrame({"next_active": [True, False], "next_net_revenue": [100.0, 0.0]})
    out = forecast_bias(ev, np.array([0.5, 0.5]), np.array([60.0, 60.0]))
    assert out["actual_active_rate"] == 0.5
    assert out["revenue_forecast_bias"] == pytest.approx(0.2)
