"""Out-of-time comparison of segmentation schemes.

Every scheme is fit on the TRAIN snapshot. Each segment's outcome rates (buy-again rate, mean
revenue) are learned from TRAIN-snapshot outcomes, then applied to customers' segments on the
EVAL snapshot and scored against what those customers actually did in the following 91 days.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .features import BEHAVIOR, VALUE, log_outcome
from .profiles import z_space


def rfm_baseline(train: pd.DataFrame, other: pd.DataFrame) -> np.ndarray:
    """Classic RFM scores (quintile edges from TRAIN) mapped to the standard named grid."""
    def score(col, higher_is_better=True):
        edges = np.unique(np.quantile(train[col], [0.2, 0.4, 0.6, 0.8]))
        s = 1 + np.digitize(other[col], edges, right=True)
        return s if higher_is_better else 6 - s
    r = score("recency_days", higher_is_better=False)
    fm = np.ceil((score("orders") + score("net_revenue")) / 2)
    names = np.full(len(other), "hibernating", dtype=object)
    names[(r <= 2) & (fm == 3)] = "at_risk"
    names[(r <= 2) & (fm >= 4)] = "cant_lose"
    names[(r == 3) & (fm <= 2)] = "needs_attention"
    names[(r >= 4) & (fm <= 2)] = "new_or_promising"
    names[(r >= 3) & (fm >= 3)] = "loyal"
    names[(r >= 4) & (fm >= 4)] = "champions"
    return names


def capture_at(score, revenue, frac=0.2) -> float:
    """Share of next-quarter revenue held by the top `frac` of customers ranked by score.
    Customers tied on score (same segment) share the cut-off proportionally."""
    d = pd.DataFrame({"s": np.asarray(score), "r": np.clip(np.asarray(revenue, float), 0, None)})
    g = d.groupby("s").agg(n=("r", "size"), r=("r", "sum")).sort_index(ascending=False)
    target, taken, got = frac * len(d), 0.0, 0.0
    for n, r in zip(g.n, g.r):
        take = min(n, target - taken)
        got += r * take / n
        taken += take
        if taken >= target:
            break
    return float(got / d.r.sum())


def eta_squared(Z: pd.DataFrame, labels, cols) -> float:
    """Mean share of each feature's variance explained by segment membership."""
    out = []
    for c in cols:
        x = Z[c].to_numpy()
        means = pd.Series(x).groupby(np.asarray(labels)).transform("mean").to_numpy()
        out.append(((means - x.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum())
    return float(np.mean(out))


def score_scheme(name, train, ev, train_labels, eval_labels, fidelity=None) -> dict:
    y_tr_active = train["next_active"].astype(int).to_numpy()
    y_tr_rev = log_outcome(train["next_net_revenue"])
    rates = pd.DataFrame({"seg": train_labels, "a": y_tr_active, "r": y_tr_rev}).groupby("seg").mean()
    p_active = pd.Series(eval_labels).map(rates["a"]).fillna(y_tr_active.mean()).to_numpy()
    p_rev = pd.Series(eval_labels).map(rates["r"]).fillna(y_tr_rev.mean()).to_numpy()
    return metrics(name, ev, p_active, p_rev, eval_labels, fidelity)


def metrics(name, ev, p_active, p_rev, labels=None, fidelity=None) -> dict:
    y_active = ev["next_active"].astype(int).to_numpy()
    revenue = ev["next_net_revenue"].to_numpy()
    row = {
        "scheme": name,
        "next_active_auc": float(roc_auc_score(y_active, p_active)),
        "revenue_spearman": float(spearmanr(p_rev, log_outcome(revenue)).statistic),
        "top20_revenue_capture": capture_at(p_rev, revenue),
        "rule_fidelity": fidelity,
    }
    if labels is not None:
        Z = z_space(ev)
        counts = pd.Series(labels).value_counts()
        row.update({
            "n_segments": int(len(counts)),
            "min_segment_share": float(counts.min() / len(labels)),
            "value_eta2": eta_squared(Z, labels, VALUE),
            "behavior_eta2": eta_squared(Z, labels, BEHAVIOR),
        })
    return row


def forecast_bias(ev, p_active, expected_revenue) -> dict:
    """Level check that rank metrics miss: predicted vs. actual buy-again rate and revenue."""
    actual_rev = ev["next_net_revenue"].clip(lower=0).sum()
    return {"predicted_active_rate": float(np.mean(p_active)),
            "actual_active_rate": float(ev["next_active"].astype(int).mean()),
            "revenue_forecast_bias": float(np.sum(expected_revenue) / actual_rev - 1)}


def persistence(train_ids, train_labels, eval_ids, eval_labels) -> float:
    """Of customers present in both snapshots, the share still in the same segment."""
    a = pd.Series(np.asarray(train_labels), index=np.asarray(train_ids))
    b = pd.Series(np.asarray(eval_labels), index=np.asarray(eval_ids))
    both = a.index.intersection(b.index)
    return float((a[both] == b[both]).mean())
