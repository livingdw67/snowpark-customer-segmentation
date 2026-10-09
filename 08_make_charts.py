"""Draw the README charts from the aggregated result tables in Snowflake.

Only summary tables (comparison metrics, segment counts, map coordinates) are read; customer-
level modeling stays in the stored procedures.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from common import DATABASE, PROJECT_DIR, get_session  # noqa: E402

ML = f"{DATABASE}.ML"
OUT = os.path.join(PROJECT_DIR, "docs", "img")

SURFACE, INK, INK_2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0", "#9a9893"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference palette slots 1-3 (validated all-pairs)
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

SCHEMES = {
    "rfm_score_baseline": "Classic RFM scores",
    "two_axis_value": "B: value axis only",
    "two_axis_grid": "B: value × behavior grid",
    "outcome_guided": "C: outcome tiers (3–4)",
    "outcome_guided_grid_size": "C: outcome tiers (9–12)",
    "hybrid": "Hybrid: C tiers × B behavior",
}
TIER_NAMES = {"C1": "Top spenders", "C2": "Core", "C3": "Occasional", "C4": "Low value"}
BEHAVIOR_NAMES = {"B1": "Bulk buyers", "B2": "Holiday buyers", "B3": "Small-basket buyers"}


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, PROJECT_DIR)}")


def tradeoff(session):
    """Who-to-target accuracy vs. how-they-buy separation, one point per scheme."""
    d = session.table(f"{ML}.METHOD_COMPARISON").to_pandas()
    d.columns = d.columns.str.lower()
    ceiling = d[d.scheme == "outcome_model_continuous"].top20_revenue_capture.mean()
    d = d[d.scheme.isin(SCHEMES)]
    g = d.groupby("scheme").agg(x=("behavior_eta2", "mean"), y=("top20_revenue_capture", "mean"),
                                y_lo=("top20_revenue_capture", "min"),
                                y_hi=("top20_revenue_capture", "max"))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(ceiling, color=MUTED, linewidth=1)
    ax.text(0.495, ceiling + 0.006, "ceiling: unsegmented model", color=INK_2,
            fontsize=8.5, ha="right")
    offsets = {"hybrid": (8, 6), "two_axis_grid": (8, -12), "two_axis_value": (8, -4),
               "outcome_guided": (8, -12), "outcome_guided_grid_size": (8, 4),
               "rfm_score_baseline": (8, -4)}
    for scheme, r in g.iterrows():
        color = SERIES[0] if scheme == "hybrid" else MUTED
        ax.plot([r.x, r.x], [r.y_lo, r.y_hi], color=color, linewidth=2, solid_capstyle="round")
        ax.scatter(r.x, r.y, s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(SCHEMES[scheme], (r.x, r.y), xytext=offsets[scheme], textcoords="offset points",
                    fontsize=9, color=INK if scheme == "hybrid" else INK_2,
                    fontweight="bold" if scheme == "hybrid" else "normal", va="center")
    ax.set_xlabel("Buying-style separation (η², share of behavior variance explained)")
    ax.set_ylabel("Next-quarter revenue held by top 20% of customers")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0.45, 0.82)
    ax.set_title("Only the hybrid does both jobs")
    fig.text(0.0, -0.04, "Dots: mean of two out-of-time backtests (fit Mar→score Jun, fit Jun→score Sep 2011). "
             "Lines: range across the two.", fontsize=8, color=INK_2)
    save(fig, "method_tradeoff.png")


def heat(ax, values, annotations, row_labels, col_labels, vmax=None):
    im = ax.imshow(values, cmap=BLUES, vmin=0, vmax=vmax or np.nanmax(values), aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            dark = v > 0.55 * (vmax or np.nanmax(values))
            ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=9,
                    color="#ffffff" if dark else INK)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # 2px surface gaps between cells
    ax.set_xticks(np.arange(-0.5, len(col_labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)
    return im


def hybrid_grid(session):
    d = session.sql(f"""
        SELECT value_tier, behavior_segment, COUNT(*) AS customers,
               SUM(expected_revenue_next_quarter) AS value
        FROM {ML}.CUSTOMER_SEGMENTS_CURRENT GROUP BY 1, 2
    """).to_pandas()
    d["value_share"] = d.VALUE / d.VALUE.sum()
    tiers, styles = list(TIER_NAMES), list(BEHAVIOR_NAMES)
    share = d.pivot(index="VALUE_TIER", columns="BEHAVIOR_SEGMENT", values="value_share").loc[tiers, styles]
    n = d.pivot(index="VALUE_TIER", columns="BEHAVIOR_SEGMENT", values="CUSTOMERS").loc[tiers, styles]
    ann = [[f"{share.iloc[i, j]:.0%} of value\n{int(n.iloc[i, j]):,} customers"
            for j in range(len(styles))] for i in range(len(tiers))]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    heat(ax, share.to_numpy(), ann, [f"{TIER_NAMES[t]} ({t})" for t in tiers],
         [f"{BEHAVIOR_NAMES[s]}\n({s})" for s in styles])
    ax.xaxis.tick_top()
    ax.set_title("Where next quarter's predicted value sits (Dec 2011, 4,261 customers)", pad=34)
    save(fig, "hybrid_grid.png")


def migration(session):
    d = session.table(f"{ML}.SEGMENT_MIGRATION").filter("TO_SNAPSHOT = '2011-12-10'").to_pandas()
    rows = ["NEW OR RETURNING", "V1", "V2", "V3", "V4"]
    cols = ["V1", "V2", "V3", "V4", "LAPSED (12M)"]
    m = d.pivot_table(index="FROM_SEGMENT", columns="TO_SEGMENT", values="CUSTOMERS",
                      aggfunc="sum", fill_value=0).reindex(index=rows, columns=cols, fill_value=0)
    pct = m.div(m.sum(axis=1), axis=0)
    ann = [[f"{pct.iloc[i, j]:.0%}" if m.iloc[i, j] else "–" for j in range(len(cols))]
           for i in range(len(rows))]
    names = {"V1": "V1 highest value", "V2": "V2", "V3": "V3", "V4": "V4 lowest value",
             "NEW OR RETURNING": "New / returning", "LAPSED (12M)": "Lapsed\n(no order in 12m)"}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    heat(ax, pct.to_numpy(), ann, [f"{names[r]}  (n={int(m.loc[r].sum()):,})" for r in rows],
         [names[c] for c in cols], vmax=1)
    ax.xaxis.tick_top()
    ax.set_ylabel("September 2011 segment")
    ax.set_title("Where each September value segment was in December", pad=34)
    save(fig, "migration.png")


def segment_map(session):
    d = session.table(f"{ML}.SEGMENT_MAP").filter("RUN_ID = 'production'").to_pandas()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for color, (seg, name) in zip(SERIES, BEHAVIOR_NAMES.items()):
        part = d[d.BEHAVIOR_SEGMENT == seg]
        ax.scatter(part.X, part.Y, s=16, color=color, edgecolor=SURFACE, linewidth=0.5,
                   alpha=0.85, label=f"{name} ({seg})")
        # Label each group at its outer edge, away from the overlap: the overlap is the point.
        cx, cy = d.X.median(), d.Y.median()
        far = part.assign(dist=np.hypot(part.X - cx, part.Y - cy)).nlargest(len(part) // 10, "dist")
        ax.annotate(name, (far.X.median(), far.Y.median()), fontsize=9, fontweight="bold",
                    color=INK, ha="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.85))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1), labelcolor=INK_2)
    ax.set_title("Customers on a continuum: ClassicalMDS map of the two-axis feature space")
    fig.text(0.0, -0.02, "2,000 customers sampled at Dec 2011. ClassicalMDS (scikit-learn 1.8+) "
             "on scaled value + behavior features; color = behavior segment. Heavy overlap: no natural clusters.",
             fontsize=8, color=INK_2)
    save(fig, "segment_map.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    session = get_session()
    tradeoff(session)
    hybrid_grid(session)
    migration(session)
    segment_map(session)
    session.close()
