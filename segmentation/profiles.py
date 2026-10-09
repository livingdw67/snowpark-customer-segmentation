"""Describe segments in plain terms: size, typical customer, and what sets each one apart."""
import numpy as np
import pandas as pd

from .features import BEHAVIOR, SKEWED, VALUE

PROFILE_COLS = VALUE + BEHAVIOR + ["avg_order_value", "tenure_days", "is_uk"]

READABLE = {
    "recency_days": ("recent", "lapsed"),
    "orders": ("frequent", "infrequent"),
    "net_revenue": ("high spend", "low spend"),
    "avg_units_per_order": ("bulk baskets", "small baskets"),
    "avg_unit_price": ("premium price point", "low price point"),
    "avg_lines_per_order": ("wide baskets", "narrow baskets"),
    "return_rate": ("high returns", "low returns"),
    "q4_revenue_share": ("holiday-heavy", "year-round"),
    "avg_order_value": ("large orders", "small orders"),
    "tenure_days": ("long tenure", "new"),
    "is_uk": ("UK", "international"),
}


def z_space(df: pd.DataFrame) -> pd.DataFrame:
    """log1p the skewed columns, then standardize: the space traits are judged in."""
    out = df[PROFILE_COLS].astype(float).copy()
    for c in PROFILE_COLS:
        if c in SKEWED:
            out[c] = np.log1p(out[c].clip(lower=0))
    return (out - out.mean()) / out.std(ddof=0).replace(0, 1)


def describe(df: pd.DataFrame, labels, top_n: int = 3) -> pd.DataFrame:
    """One row per segment with medians and the top distinguishing traits.

    Traits are the segment's largest mean z-scores, so "recent" means a *low* recency_days.
    """
    transformed = z_space(df)
    labels = pd.Series(np.asarray(labels), index=df.index, name="segment")
    rows = []
    for seg, idx in labels.groupby(labels).groups.items():
        part = df.loc[idx]
        z = transformed.loc[idx].mean()
        traits = []
        for feat in z.abs().sort_values(ascending=False).index[:top_n]:
            if abs(z[feat]) < 0.25:
                break
            high, low = READABLE.get(feat, (f"high {feat}", f"low {feat}"))
            # recency is "good" when small
            is_high = z[feat] > 0 if feat != "recency_days" else z[feat] < 0
            traits.append(high if is_high else low)
        row = {"segment": seg, "customers": len(part), "share": len(part) / len(df),
               "traits": ", ".join(traits) or "average on every feature"}
        for c in PROFILE_COLS:
            row[f"median_{c}" if c != "is_uk" else "uk_share"] = (
                float(part[c].mean()) if c == "is_uk" else float(part[c].median()))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)
