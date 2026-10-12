"""Draw the README charts, in a light and a dark version each.

Only summary tables (comparison metrics, segment totals, migration counts, map coordinates) are
read; customer-level modeling stays in the stored procedures.

    python 08_make_charts.py                  # read ML.* in Snowflake (all four charts)
    python 08_make_charts.py --from-results   # read results/*.csv (no Snowflake needed)

The Snowflake run also writes results/segment_migration.csv, so later --from-results runs can draw
the migration chart too. The segment map plots sampled customers, so it is only drawn from
Snowflake and its coordinates are never exported.

Each chart is written as <name>.png (light) and <name>-dark.png. The README shows the dark one
under GitHub's dark theme with <picture><source media="(prefers-color-scheme: dark)" ...>.
Colors are the brand chart palettes (validated for contrast and color-vision deficiency).
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PROJECT_DIR, "docs", "img")
RESULTS = os.path.join(PROJECT_DIR, "results")

THEMES = {
    "light": dict(surface="#f7f9fb", ink="#0e1a2b", ink_2="#3c5a7d", muted="#8a9bb0", grid="#dde4ec",
                  series=["#00948c", "#c2760a", "#5b6fd8"], highlight="#c2760a", accent="#00948c",
                  ramp=["#e3f3f1", "#58b0aa", "#107f7a", "#00514d"]),
    "dark": dict(surface="#0e1a2b", ink="#f2f5f9", ink_2="#9fb1c7", muted="#6f86a3", grid="#2b3f58",
                 series=["#26a59f", "#c8820f", "#7c80ea"], highlight="#f0a742", accent="#26a59f",
                 ramp=["#16283f", "#075957", "#248b88", "#5ebdb9", "#79d7d3"]),
}

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
# Brand headings are Segoe UI at weight 600, which matplotlib only finds as its own family.
SEMIBOLD = ({"fontfamily": "Segoe UI Semibold", "fontweight": "normal"}
            if any(f.name == "Segoe UI Semibold" for f in font_manager.fontManager.ttflist)
            else {"fontweight": "bold"})


def style(t):
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "axes.edgecolor": t["grid"], "axes.labelcolor": t["ink_2"], "text.color": t["ink"],
        "xtick.color": t["ink_2"], "ytick.color": t["ink_2"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 13, "axes.titlelocation": "left",
        "axes.titlecolor": t["ink"],
    })


def save(fig, name, mode):
    path = os.path.join(OUT, f"{name}.png" if mode == "light" else f"{name}-dark.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, PROJECT_DIR)}")


def luminance(color):
    c = np.array(to_rgb(color))
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def text_on(fill, t):
    """The theme ink (dark or light) with the higher contrast on this fill."""
    inks = [t["ink"], t["surface"]]
    contrast = [(max(luminance(fill), luminance(i)) + 0.05) / (min(luminance(fill), luminance(i)) + 0.05)
                for i in inks]
    return inks[int(np.argmax(contrast))]


# ---- charts ---------------------------------------------------------------------------------

def tradeoff(d, t):
    """Revenue concentration vs buying-style separation, one point per scheme."""
    ceiling = d[d.SCHEME == "outcome_model_continuous"].TOP20_REVENUE_CAPTURE.mean()
    g = d[d.SCHEME.isin(SCHEMES)].groupby("SCHEME").agg(
        x=("BEHAVIOR_ETA2", "mean"), y=("TOP20_REVENUE_CAPTURE", "mean"),
        lo=("TOP20_REVENUE_CAPTURE", "min"), hi=("TOP20_REVENUE_CAPTURE", "max"))
    color = {"hybrid": t["highlight"], "two_axis_grid": t["accent"]}

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.grid(True, axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(ceiling, color=t["muted"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(0.545, ceiling + 0.007, f"Unsegmented model (ceiling) {ceiling:.0%}", color=t["ink_2"],
            fontsize=8.5, ha="right")
    offsets = {"hybrid": (9, -1), "two_axis_grid": (9, -1), "two_axis_value": (9, 4),
               "outcome_guided": (9, -10), "outcome_guided_grid_size": (9, -12),
               "rfm_score_baseline": (9, -4)}
    for scheme, r in g.iterrows():
        c = color.get(scheme, t["muted"])
        lead = scheme in color
        ax.plot([r.x, r.x], [r.lo, r.hi], color=c, linewidth=2, solid_capstyle="round")
        ax.scatter(r.x, r.y, s=70 if lead else 50, color=c, edgecolor=t["surface"], linewidth=2, zorder=3)
        label = SCHEMES[scheme] + (f"  {r.y:.0%}" if lead else "")
        ax.annotate(label, (r.x, r.y), xytext=offsets[scheme], textcoords="offset points", va="center",
                    fontsize=9, color=t["ink"] if lead else t["ink_2"], **(SEMIBOLD if lead else {}))
    ax.set_xlabel("Buying-style separation (η², share of behavior variance explained)")
    ax.set_ylabel("Next-quarter revenue held by the top 20%")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlim(0, 0.55)
    ax.set_ylim(0.45, 0.80)
    ax.tick_params(length=0)
    ax.set_title("Two schemes separate buying styles; the hybrid also holds more revenue", pad=12, **SEMIBOLD)
    fig.text(0.0, -0.03, "Dot: mean of two out-of-time backtests (fit Mar → score Jun, fit Jun → score Sep 2011). "
             "Line: the range across the two.", fontsize=8, color=t["ink_2"])
    return fig


def heat(ax, values, labels, rows, cols, t, vmax=None):
    cmap = LinearSegmentedColormap.from_list("ramp", t["ramp"])
    vmax = vmax or np.nanmax(values)
    ax.imshow(values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            fill = cmap(values[i, j] / vmax)
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=9, color=text_on(fill, t))
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelcolor=t["ink_2"])
    for s in ax.spines.values():
        s.set_visible(False)
    # surface gaps between cells
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color=t["surface"], linewidth=3)
    ax.tick_params(which="minor", length=0)


def hybrid_grid(d, t):
    """d: one row per hybrid segment (SEGMENT like 'C1-B2', CUSTOMERS, EXPECTED_REVENUE)."""
    d = d.assign(tier=d.SEGMENT.str[:2], style=d.SEGMENT.str[3:])
    d["share"] = d.EXPECTED_REVENUE / d.EXPECTED_REVENUE.sum()
    tiers, styles = list(TIER_NAMES), list(BEHAVIOR_NAMES)
    share = d.pivot(index="tier", columns="style", values="share").loc[tiers, styles]
    n = d.pivot(index="tier", columns="style", values="CUSTOMERS").loc[tiers, styles]
    labels = [[f"{share.iloc[i, j]:.0%} of value\n{int(n.iloc[i, j]):,} customers" for j in range(len(styles))]
              for i in range(len(tiers))]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    heat(ax, share.to_numpy(), labels, [f"{TIER_NAMES[x]} ({x})" for x in tiers],
         [f"{BEHAVIOR_NAMES[x]} ({x})" for x in styles], t)
    ax.set_title(f"Where next quarter's predicted value sits ({int(d.CUSTOMERS.sum()):,} customers, Dec 2011)",
                 pad=28, **SEMIBOLD)
    return fig


def migration(d, t):
    """d: FROM_SEGMENT, TO_SEGMENT, CUSTOMERS for September -> December 2011 (option B value axis)."""
    rows = ["NEW OR RETURNING", "V1", "V2", "V3", "V4"]
    cols = ["V1", "V2", "V3", "V4", "LAPSED (12M)"]
    m = d.pivot_table(index="FROM_SEGMENT", columns="TO_SEGMENT", values="CUSTOMERS",
                      aggfunc="sum", fill_value=0).reindex(index=rows, columns=cols, fill_value=0)
    pct = m.div(m.sum(axis=1), axis=0)
    labels = [[f"{pct.iloc[i, j]:.0%}" if m.iloc[i, j] else "none" for j in range(len(cols))]
              for i in range(len(rows))]
    names = {"V1": "Highest value (V1)", "V2": "V2", "V3": "V3", "V4": "Lowest value (V4)",
             "NEW OR RETURNING": "New or returning", "LAPSED (12M)": "Lapsed (no order\nin 12 months)"}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    heat(ax, pct.to_numpy(), labels, [f"{names[r]}  n={int(m.loc[r].sum()):,}" for r in rows],
         [names[c].replace(" value ", "\nvalue ") for c in cols], t, vmax=1)
    ax.set_ylabel("September 2011 value segment")
    ax.set_title("Where each September value segment was in December", pad=34, **SEMIBOLD)
    fig.text(0.0, 0.0, "Option B's value axis. Each row sums to 100%.", fontsize=8, color=t["ink_2"])
    return fig


def segment_map(d, t):
    """d: X, Y, BEHAVIOR_SEGMENT for a sample of customers (never exported)."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cx, cy = d.X.median(), d.Y.median()
    for color, (seg, name) in zip(t["series"], BEHAVIOR_NAMES.items()):
        part = d[d.BEHAVIOR_SEGMENT == seg]
        ax.scatter(part.X, part.Y, s=16, color=color, edgecolor=t["surface"], linewidth=0.5,
                   alpha=0.85, label=f"{name} ({seg})")
        # Label each group at its outer edge, away from the overlap: the overlap is the point.
        far = part.assign(dist=np.hypot(part.X - cx, part.Y - cy)).nlargest(max(len(part) // 10, 1), "dist")
        ax.annotate(name, (far.X.median(), far.Y.median()), fontsize=9, color=t["ink"], ha="center", **SEMIBOLD,
                    bbox=dict(boxstyle="round,pad=0.25", fc=t["surface"], ec="none", alpha=0.85))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, labelcolor=t["ink_2"])
    ax.set_title("Customers sit on a continuum, not in clusters", **SEMIBOLD)
    fig.text(0.0, -0.06, "2,000 customers sampled at Dec 2011. ClassicalMDS on scaled value and behavior "
             "features; color = buying style.", fontsize=8, color=t["ink_2"])
    return fig


# ---- data -----------------------------------------------------------------------------------

def from_results():
    path = os.path.join(RESULTS, "segment_migration.csv")
    return {
        "method_tradeoff": pd.read_csv(os.path.join(RESULTS, "method_comparison.csv")),
        "hybrid_grid": pd.read_csv(os.path.join(RESULTS, "hybrid_profiles.csv")),
        "migration": pd.read_csv(path) if os.path.exists(path) else None,
    }


def from_snowflake():
    from common import DATABASE, get_session

    ml = f"{DATABASE}.ML"
    session = get_session()
    try:
        grid = session.sql(f"""
            SELECT value_tier || '-' || behavior_segment AS segment, COUNT(*) AS customers,
                   SUM(expected_revenue_next_quarter) AS expected_revenue
            FROM {ml}.CUSTOMER_SEGMENTS_CURRENT GROUP BY 1
        """).to_pandas()
        moves = session.sql(f"""
            SELECT from_segment, to_segment, SUM(customers) AS customers
            FROM {ml}.SEGMENT_MIGRATION WHERE to_snapshot = '2011-12-10' GROUP BY 1, 2 ORDER BY 1, 2
        """).to_pandas()
        moves.to_csv(os.path.join(RESULTS, "segment_migration.csv"), index=False)
        print("wrote results/segment_migration.csv")
        return {
            "method_tradeoff": session.table(f"{ml}.METHOD_COMPARISON").to_pandas(),
            "hybrid_grid": grid,
            "migration": moves,
            "segment_map": session.table(f"{ml}.SEGMENT_MAP").filter("RUN_ID = 'production'").to_pandas(),
        }
    finally:
        session.close()


CHARTS = {"method_tradeoff": tradeoff, "hybrid_grid": hybrid_grid, "migration": migration,
          "segment_map": segment_map}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-results", action="store_true", help="read results/*.csv instead of Snowflake")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    data = from_results() if args.from_results else from_snowflake()
    for name, draw in CHARTS.items():
        if data.get(name) is None:
            print(f"skipped {name}: needs a Snowflake run")
            continue
        for mode, theme in THEMES.items():
            style(theme)
            save(draw(data[name], theme), name, mode)
