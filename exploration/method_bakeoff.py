"""Compare segmentation methods inside Snowflake (temporary stored procedure).

Fits every candidate on the 2011-09-10 snapshot, then scores each one on:
  - geometry: silhouette and Davies-Bouldin in the transformed feature space
  - stability: mean adjusted Rand index between the full fit and fits on 80% subsamples
  - usefulness: how well segment membership alone predicts the next 91 days
    (AUC for "buys again", R^2 for log revenue), using segment rates learned on one half of
    customers and scored on the other half
against a classic RFM-score segmentation as the baseline.
Results land in ML.METHOD_BAKEOFF; nothing leaves Snowflake.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snowflake.snowpark import Session
from snowflake.snowpark.types import StringType

from common import DATABASE, SPROC_PACKAGES, get_session

SNAPSHOT = "2011-09-10"

RFM = ["recency_days", "orders", "net_revenue"]
BEHAVIOR = RFM + [
    "avg_order_value", "avg_units_per_order", "avg_unit_price", "distinct_products",
    "return_rate", "q4_revenue_share", "tenure_days", "active_months",
]


def bakeoff(session: Session) -> str:
    import numpy as np
    import pandas as pd
    from sklearn.cluster import HDBSCAN, AgglomerativeClustering, BisectingKMeans, KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score, r2_score,
                                 roc_auc_score, silhouette_score)
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, QuantileTransformer, StandardScaler

    df = session.sql(f"""
        SELECT * FROM {DATABASE}.MARTS.CUSTOMER_FEATURES WHERE snapshot_date = '{SNAPSHOT}'
    """).to_pandas()
    df.columns = df.columns.str.lower()
    df["net_revenue"] = df["net_revenue"].clip(lower=0)
    y_active = df["next_active"].astype(int).to_numpy()
    y_rev = np.log1p(df["next_net_revenue"].clip(lower=0).to_numpy())

    skewed = {"recency_days", "orders", "net_revenue", "avg_order_value", "avg_units_per_order",
              "avg_unit_price", "distinct_products", "tenure_days", "active_months"}

    def preprocessor(cols, kind):
        if kind == "quantile":
            return QuantileTransformer(output_distribution="normal", n_quantiles=500, random_state=0)
        log_cols = [c for c in cols if c in skewed]
        rest = [c for c in cols if c not in skewed]
        ct = ColumnTransformer([
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), log_cols),
            ("pass", "passthrough", rest),
        ])
        return make_pipeline(ct, StandardScaler())

    def model(name, k):
        return {
            "kmeans": lambda: KMeans(n_clusters=k, n_init=10, random_state=0),
            "bisecting_kmeans": lambda: BisectingKMeans(n_clusters=k, n_init=5, random_state=0),
            "gmm": lambda: GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=0),
            "ward": lambda: AgglomerativeClustering(n_clusters=k, linkage="ward"),
            "hdbscan": lambda: HDBSCAN(min_cluster_size=k, min_samples=10),
        }[name]()

    def outcome_scores(labels):
        """Out-of-fold: segment rates learned on train customers, applied to held-out ones."""
        p_active = np.zeros(len(labels)); p_rev = np.zeros(len(labels))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(labels, y_active):
            s = pd.DataFrame({"seg": labels[tr], "a": y_active[tr], "r": y_rev[tr]}).groupby("seg").mean()
            p_active[te] = pd.Series(labels[te]).map(s["a"]).fillna(y_active[tr].mean()).to_numpy()
            p_rev[te] = pd.Series(labels[te]).map(s["r"]).fillna(y_rev[tr].mean()).to_numpy()
        return roc_auc_score(y_active, p_active), r2_score(y_rev, p_rev)

    def stability(make, X, ref, n=10):
        rng = np.random.default_rng(0); scores = []
        for _ in range(n):
            idx = rng.choice(len(X), int(0.8 * len(X)), replace=False)
            scores.append(adjusted_rand_score(ref[idx], make().fit_predict(X[idx])))
        return float(np.mean(scores)), float(np.min(scores))

    rows = []

    # Baseline: classic RFM scoring (quintiles 1-5) mapped to the standard named R x F grid.
    r = pd.qcut(-df["recency_days"].rank(method="first"), 5, labels=False) + 1
    f = pd.qcut(df["orders"].rank(method="first"), 5, labels=False) + 1
    m = pd.qcut(df["net_revenue"].rank(method="first"), 5, labels=False) + 1
    fm = np.ceil((f + m) / 2).astype(int)
    def rfm_name(ri, fi):
        if ri >= 4 and fi >= 4: return "champions"
        if ri >= 3 and fi >= 3: return "loyal"
        if ri >= 4 and fi <= 2: return "new_or_promising"
        if ri == 3 and fi <= 2: return "needs_attention"
        if ri <= 2 and fi >= 4: return "cant_lose"
        if ri <= 2 and fi == 3: return "at_risk"
        return "hibernating"
    rfm_labels = pd.factorize(pd.Series([rfm_name(a, b) for a, b in zip(r, fm)]))[0]
    auc, r2 = outcome_scores(rfm_labels)
    rows.append(dict(method="rfm_score_baseline", features="rfm", prep="quintiles",
                     k=len(set(rfm_labels)), silhouette=None, davies_bouldin=None,
                     stability_ari=None, stability_ari_min=None,
                     min_segment_share=float(np.bincount(rfm_labels).min() / len(df)),
                     noise_share=0.0, next_active_auc=auc, next_revenue_r2=r2))

    for feat_name, cols in (("rfm", RFM), ("behavior", BEHAVIOR)):
        for prep in ("log_standard", "quantile"):
            X = preprocessor(cols, prep).fit_transform(df[cols])
            X = np.asarray(X, dtype=float)
            grid = [(name, k) for name in ("kmeans", "bisecting_kmeans", "gmm", "ward") for k in range(3, 9)]
            grid += [("hdbscan", mcs) for mcs in (40, 80, 150)]
            for name, k in grid:
                make = lambda name=name, k=k: model(name, k)
                labels = make().fit_predict(X)
                clustered = labels >= 0
                n_seg = len(set(labels[clustered]))
                if n_seg < 2:
                    continue
                mean_ari, min_ari = stability(make, X, labels)
                auc, r2 = outcome_scores(labels)
                rows.append(dict(
                    method=name, features=feat_name, prep=prep, k=n_seg,
                    silhouette=float(silhouette_score(X[clustered], labels[clustered])),
                    davies_bouldin=float(davies_bouldin_score(X[clustered], labels[clustered])),
                    stability_ari=mean_ari, stability_ari_min=min_ari,
                    min_segment_share=float(np.bincount(labels[clustered]).min() / len(df)),
                    noise_share=float((~clustered).mean()),
                    next_active_auc=auc, next_revenue_r2=r2,
                    min_cluster_size=k if name == "hdbscan" else None,
                ))

    out = pd.DataFrame(rows)
    out.columns = out.columns.str.upper()
    session.write_pandas(out, "METHOD_BAKEOFF", database=DATABASE, schema="ML",
                         auto_create_table=True, overwrite=True)
    return f"{len(out)} configurations scored on {len(df)} customers"


if __name__ == "__main__":
    session = get_session()
    proc = session.sproc.register(
        bakeoff, return_type=StringType(), input_types=[],
        packages=SPROC_PACKAGES, is_permanent=False, replace=True,
    )
    print(proc())
    res = session.table(f"{DATABASE}.ML.METHOD_BAKEOFF").to_pandas()
    res.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "method_bakeoff.csv"), index=False)
    session.close()
