"""Option B: two independent segmentations crossed into a grid.

Value axis    - who is worth what right now (recency, frequency, spend).
Behavior axis - how they buy (basket shape, price point, returns, seasonality).

Each axis is a BisectingKMeans pipeline. Customers sit on a continuum (HDBSCAN finds no dense
clusters, see exploration/), so k is chosen for usefulness and stability rather than "natural"
structure:
  value axis    -> best out-of-fold R^2 for next-91-day log revenue
  behavior axis -> best silhouette
both subject to bootstrap stability (ARI >= 0.85) and every segment holding >= 5% of customers.
BisectingKMeans with bisecting_strategy="largest_cluster" keeps splitting the biggest group,
which avoids carving out tiny outlier segments.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import BisectingKMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import adjusted_rand_score, r2_score, silhouette_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from .features import BEHAVIOR, SKEWED, VALUE, log_outcome
from .profiles import describe

MIN_STABILITY = 0.85
MIN_SHARE = 0.05
TIE_TOLERANCE = 0.01
K_RANGE = range(3, 8)


def preprocessor(cols) -> Pipeline:
    log_cols = [c for c in cols if c in SKEWED]
    raw_cols = [c for c in cols if c not in SKEWED]
    columns = ColumnTransformer(
        [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), log_cols),
         ("raw", "passthrough", raw_cols)],
        verbose_feature_names_out=False,
    )
    # Clip at +/-3 SD so a handful of extreme wholesale orders can't own a centroid.
    clip = FunctionTransformer(np.clip, kw_args={"a_min": -3, "a_max": 3},
                               feature_names_out="one-to-one")
    return make_pipeline(columns, StandardScaler(), clip).set_output(transform="pandas")


def axis_pipeline(cols, k) -> Pipeline:
    return Pipeline([
        ("prep", preprocessor(cols)),
        ("cluster", BisectingKMeans(n_clusters=k, bisecting_strategy="largest_cluster",
                                    n_init=5, random_state=0)),
    ])


def _stability(cols, k, X, labels, n=10) -> float:
    rng = np.random.default_rng(0)
    scores = []
    for _ in range(n):
        idx = rng.choice(len(X), int(0.8 * len(X)), replace=False)
        sub = axis_pipeline(cols, k).fit(X.iloc[idx])
        scores.append(adjusted_rand_score(labels[idx], sub.predict(X.iloc[idx])))
    return float(np.mean(scores))


def _oof_r2(labels, y) -> float:
    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(labels):
        means = pd.Series(y[tr]).groupby(labels[tr]).mean()
        pred[te] = pd.Series(labels[te]).map(means).fillna(y[tr].mean()).to_numpy()
    return float(r2_score(y, pred))


def select_k(df, cols, axis: str) -> tuple[int, pd.DataFrame]:
    X = df[cols]
    y = log_outcome(df["next_net_revenue"])
    rows = []
    for k in K_RANGE:
        pipe = axis_pipeline(cols, k).fit(X)
        labels = pipe.predict(X)
        Z = pipe[:-1].transform(X)
        rows.append({
            "axis": axis, "k": k,
            "stability_ari": _stability(cols, k, X, labels),
            "min_share": float(np.bincount(labels).min() / len(labels)),
            "silhouette": float(silhouette_score(Z, labels)),
            "next_revenue_r2": _oof_r2(labels, y),
        })
    table = pd.DataFrame(rows)
    table["eligible"] = (table.stability_ari >= MIN_STABILITY) & (table.min_share >= MIN_SHARE)
    criterion = "next_revenue_r2" if axis == "value" else "silhouette"
    pool = table[table.eligible] if table.eligible.any() else table
    # Prefer the smaller k unless a larger one is better by more than TIE_TOLERANCE:
    # fewer segments are easier to act on and their rules stay readable.
    best = pool[criterion].max()
    k = int(pool[pool[criterion] >= best - TIE_TOLERANCE].k.min())
    table["chosen"] = table.k == k
    table["criterion"] = criterion
    return k, table


class TwoAxisSegmenter:
    """Fit both axes, order segments so codes are stable, and explain them with rules."""

    def fit(self, df: pd.DataFrame):
        self.k_value_, value_sel = select_k(df, VALUE, "value")
        self.k_behavior_, behavior_sel = select_k(df, BEHAVIOR, "behavior")
        self.k_selection_ = pd.concat([value_sel, behavior_sel], ignore_index=True)

        self.value_pipe_ = axis_pipeline(VALUE, self.k_value_).fit(df[VALUE])
        self.behavior_pipe_ = axis_pipeline(BEHAVIOR, self.k_behavior_).fit(df[BEHAVIOR])

        # Number segments by meaning, not by k-means' arbitrary ids:
        # V1 = highest median spend, B1 = largest median basket.
        v = self.value_pipe_.predict(df[VALUE])
        b = self.behavior_pipe_.predict(df[BEHAVIOR])
        v_rank = df.groupby(v)["net_revenue"].median().rank(ascending=False, method="first")
        b_rank = df.groupby(b)["avg_units_per_order"].median().rank(ascending=False, method="first")
        self.value_order_ = {int(c): f"V{int(r)}" for c, r in v_rank.items()}
        self.behavior_order_ = {int(c): f"B{int(r)}" for c, r in b_rank.items()}

        labels = self.predict(df)
        self.profiles_ = pd.concat([
            describe(df, labels.value_segment).assign(axis="value"),
            describe(df, labels.behavior_segment).assign(axis="behavior"),
            describe(df, labels.segment).assign(axis="grid"),
        ], ignore_index=True)

        # Plain-English assignment rules: a shallow tree trained to mimic each axis.
        self.rules_ = {}
        for axis, cols, y in (("value", VALUE, labels.value_segment),
                              ("behavior", BEHAVIOR, labels.behavior_segment)):
            tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.02, random_state=0)
            fidelity = cross_val_score(tree, df[cols], y, cv=5).mean()
            tree.fit(df[cols], y)
            self.rules_[axis] = {"fidelity": float(fidelity),
                                 "rules": export_text(tree, feature_names=cols, decimals=2)}
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        v = pd.Series(self.value_pipe_.predict(df[VALUE])).map(self.value_order_)
        b = pd.Series(self.behavior_pipe_.predict(df[BEHAVIOR])).map(self.behavior_order_)
        return pd.DataFrame({
            "value_segment": v.to_numpy(),
            "behavior_segment": b.to_numpy(),
            "segment": (v + "-" + b).to_numpy(),
        }, index=df.index)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Both axes' scaled features side by side (used for the segment map)."""
        return pd.concat([self.value_pipe_[:-1].transform(df[VALUE]),
                          self.behavior_pipe_[:-1].transform(df[BEHAVIOR])], axis=1)
