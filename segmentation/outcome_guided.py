"""Option C: segments defined by what customers are predicted to do next.

Trained on every past snapshot whose outcomes are known, pooled, with the snapshot month as a
feature, so forecasts for a quiet quarter aren't learned only from the holiday quarter. The same
customer appears in several snapshots, so every split is grouped by customer.

1. P(buys again in 91 days): HistGradientBoostingClassifier, then temperature-scaled on a
   held-out calibration split. FrozenEstimator keeps the calibration step from refitting the
   booster, so calibration never sees the data the booster was trained on.
2. Expected 91-day revenue: HistGradientBoostingRegressor with Poisson loss (most customers
   spend 0 next quarter, and spend is non-negative and right-skewed).
3. Segments: a shallow regression tree on readable features, trained to reproduce the
   out-of-fold expected revenue. Each leaf is a segment and its path is the rule.
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.tree import DecisionTreeRegressor

from .features import MODEL, RULE
from .profiles import describe


def _booster(cls, **kw):
    return cls(learning_rate=0.05, max_iter=500, early_stopping=True,
               validation_fraction=0.15, random_state=0, **kw)


def leaf_rules(tree: DecisionTreeRegressor, feature_names) -> dict[int, str]:
    """Readable rule for every leaf, merging repeated splits on the same feature."""
    t = tree.tree_
    rules = {}

    def walk(node, bounds):
        if t.children_left[node] == -1:
            parts = []
            for f, (lo, hi) in bounds.items():
                if f == "is_uk":
                    parts.append("UK" if lo is not None else "outside UK")
                elif lo is not None and hi is not None:
                    parts.append(f"{lo:g} < {f} <= {hi:g}")
                elif lo is not None:
                    parts.append(f"{f} > {lo:g}")
                else:
                    parts.append(f"{f} <= {hi:g}")
            rules[node] = " AND ".join(parts) or "all customers"
            return
        f = feature_names[t.feature[node]]
        thr = round(float(t.threshold[node]), 2)
        lo, hi = bounds.get(f, (None, None))
        walk(t.children_left[node], {**bounds, f: (lo, thr if hi is None else min(hi, thr))})
        walk(t.children_right[node], {**bounds, f: (thr if lo is None else max(lo, thr), hi)})

    walk(0, {})
    return rules


class OutcomeGuidedSegmenter:
    def __init__(self, n_segments: int):
        self.n_segments = n_segments

    def fit(self, df: pd.DataFrame):
        df = df.reset_index(drop=True)
        X = df[MODEL]
        groups = df["customer_id"].to_numpy()
        y_active = df["next_active"].astype(int).to_numpy()
        y_revenue = df["next_net_revenue"].clip(lower=0).to_numpy()

        fit_idx, cal_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
                                .split(X, y_active, groups))
        booster = _booster(HistGradientBoostingClassifier).fit(X.iloc[fit_idx], y_active[fit_idx])
        self.active_model_ = CalibratedClassifierCV(
            FrozenEstimator(booster), method="temperature").fit(X.iloc[cal_idx], y_active[cal_idx])

        reg = _booster(HistGradientBoostingRegressor, loss="poisson")
        oof_value = cross_val_predict(reg, X, y_revenue, groups=groups, cv=GroupKFold(5))
        self.revenue_model_ = reg.fit(X, y_revenue)

        # The tree reproduces the model's view of value in log space, using rules people can read.
        target = np.log1p(oof_value)
        self.tree_ = DecisionTreeRegressor(max_leaf_nodes=self.n_segments, min_samples_leaf=0.05,
                                           random_state=0).fit(df[RULE], target)
        self.tree_fidelity_r2_ = float(r2_score(target, self.tree_.predict(df[RULE])))

        leaves = self.tree_.apply(df[RULE])
        leaf_value = pd.Series(oof_value).groupby(leaves).mean()
        rank = leaf_value.rank(ascending=False, method="first")
        self.leaf_order_ = {int(leaf): f"C{int(r)}" for leaf, r in rank.items()}
        rules = leaf_rules(self.tree_, RULE)
        self.rules_ = {self.leaf_order_[leaf]: rules[leaf] for leaf in self.leaf_order_}

        # Profiles describe the most recent training snapshot, the one segments are read against.
        latest = df[df.snapshot_date == df.snapshot_date.max()]
        self.profiles_ = describe(latest, self.predict(latest).segment).assign(axis="outcome")
        self.profiles_["rule"] = self.profiles_.segment.map(self.rules_)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        leaves = self.tree_.apply(df[RULE])
        return pd.DataFrame({
            "segment": pd.Series(leaves).map(self.leaf_order_).to_numpy(),
            "p_active": self.active_model_.predict_proba(df[MODEL])[:, 1],
            "expected_revenue": self.revenue_model_.predict(df[MODEL]),
        }, index=df.index)
