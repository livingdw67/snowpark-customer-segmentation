"""How big a fair Segment Assistant vs Cortex Analyst test has to be, and what each round found.

    streamlit run streamlit_app.py

A preview page for the portfolio app. Needs streamlit (altair, numpy and pandas come with it) and
rounds.json next to this file (written by power_page_data.py in the eval folder).
"""
import json
import math
from pathlib import Path
from statistics import NormalDist

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

Z = NormalDist()
CREDITS_PER_QUESTION = 0.067  # one Cortex Analyst message
PILOT_SPLIT = 27 / 175  # held-out questions where exactly one system was right (both fixed)
EFFECTS = [2.0, 3.0, 3.4, 4.0, 5.0]  # points of accuracy
DESIGNS = [  # tests actually run: (questions, short label, what it was)
    (175, "175", "Held-out test: 175 questions"),
    (1050, "1,050", "First plan: 80% power at 3.4 points"),
    (1400, "1,400", "This run: 90% power at 3.4 points"),
]
INK, SUB, MUTED, GRID, SURFACE = "#f2f5f9", "#9fb1c7", "#6f86a3", "#2b3f58", "#0e1a2b"
TEAL, TEAL_LIGHT, AMBER, SLATE = "#5fc4c0", "#79d7d3", "#f0a742", "#6f86a3"
ROUNDS = json.loads((Path(__file__).parent / "rounds.json").read_text(encoding="utf-8"))
RUN = next((x for x in ROUNDS if x["short"] == "Power run"), None)  # present once the run is graded
TYPE_NAMES = {"lookup": "Look up one figure", "total_share": "Totals and shares", "ranking": "Rankings",
              "comparison": "Comparisons", "breakdown": "Breakdowns", "filter": "Filters",
              "model_eval": "Model scores", "followup": "Follow-ups", "ambiguous": "Ambiguous (should ask)",
              "unanswerable": "Can't be answered (should decline)"}


def fmt_p(p):
    """p-values for reading: tiny ones as '< 0.0001' rather than a rounded 0.000."""
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


# ----------------------------------------------------------------------------- the statistics
def questions_needed(power, effect, split, alpha):
    """Connor (1987): paired questions for McNemar's test to detect `effect`, a difference in accuracy."""
    za, zb = Z.inv_cdf(1 - alpha / 2), Z.inv_cdf(power)
    return (za * math.sqrt(split) + zb * math.sqrt(split - effect**2)) ** 2 / effect**2


def power_of(n, effect, split, alpha):
    za = Z.inv_cdf(1 - alpha / 2)
    return Z.cdf((math.sqrt(n) * effect - za * math.sqrt(split)) / math.sqrt(split - effect**2))


def log_choose(n, lf):
    k = np.arange(n + 1)
    return lf[n] - lf[k] - lf[n - k]


@st.cache_data(show_spinner=False, max_entries=64)
def exact_power(effect, split, alpha, n_max, points=160):
    """Power of the exact (binomial) McNemar test at n_max/points steps up to n_max questions.

    Given d split decisions, the test rejects when the split is lopsided enough under a fair coin;
    under the alternative each split goes Analyst's way with chance 1/2 + effect / (2 split).
    Averaging over d ~ Binomial(n, split) gives the power at n questions."""
    theta = 0.5 + effect / (2 * split)
    lf = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n_max + 1)))])
    d_max = min(n_max, int(split * n_max + 8 * math.sqrt(n_max * split) + 20))
    reject_power = np.zeros(d_max + 1)
    for d in range(6, d_max + 1):  # fewer than 6 split decisions can never reach 5%
        lc = log_choose(d, lf)
        null = np.exp(lc + d * math.log(0.5))
        lower, upper = np.cumsum(null), np.cumsum(null[::-1])[::-1]
        reject = np.minimum(1, 2 * np.minimum(lower, upper)) <= alpha
        k = np.arange(d + 1)
        alt_pmf = np.exp(lc + k * math.log(theta) + (d - k) * math.log(1 - theta))
        reject_power[d] = alt_pmf[reject].sum()
    rows = []
    for n in np.unique(np.linspace(20, n_max, points).astype(int)):
        d = np.arange(min(n, d_max) + 1)
        weights = np.exp(log_choose(n, lf)[: d.size] + d * math.log(split) + (n - d) * math.log(1 - split))
        rows.append((float(weights @ reject_power[: d.size]), int(n)))
    return pd.DataFrame(rows, columns=["power", "questions"])


# ----------------------------------------------------------------------------- page
st.set_page_config(page_title="How big a test?", page_icon=":material/query_stats:", layout="centered")
st.title("How many questions does a fair test need?")
st.caption(
    "Segment Assistant, a semantic layer I built on an open-weight model, against Snowflake's Cortex Analyst. "
    "Both answer the same questions, so every question gives a pair of right-or-wrong results, and "
    "McNemar's test compares them. Only the split decisions, where exactly one system is right, can tell "
    "the two apart; the rest cancel out."
)

power_pct = st.slider("Power", min_value=10, max_value=99, value=80, step=1, format="%d%%",
                      help="The chance the test detects a real difference of the size below.")
effect_pts = st.segmented_control("Difference to detect, in points of accuracy", EFFECTS, default=3.4,
                                  format_func=lambda x: f"{x:g} pts", required=True,
                                  help="Power always needs a difference to aim at. 3.4 points is the gap the first "
                                       "round of testing showed. It also sets the equivalence margin below.")
with st.expander("Assumptions", icon=":material/tune:"):
    split_pct = st.slider("Questions where exactly one system is right", 5.0, 30.0, round(PILOT_SPLIT * 100, 1), 0.1,
                          format="%.1f%%", help="Measured on the 175 held-out questions after both fixes: 27 split decisions.")
    alpha = st.segmented_control("Significance level", [0.05, 0.01], default=0.05, required=True,
                                 format_func=lambda x: f"{x:.0%}")
    show_exact = st.toggle("Overlay the exact test's power", value=True,
                           help="The curve uses Connor's normal approximation; the dotted line is the exact binomial test.")

power, effect, split = power_pct / 100, effect_pts / 100, split_pct / 100
if effect >= split:
    st.warning("The difference can't be larger than the share of split decisions. Raise that share or pick a smaller "
               "difference.", icon=":material/warning:")
    st.stop()

needed = math.ceil(questions_needed(power, effect, split, alpha))
run_n = DESIGNS[-1][0]
run_power = power_of(run_n, effect, split, alpha)
with st.container(horizontal=True):
    st.metric("Questions needed", f"{needed:,}", border=True)
    st.metric("Split decisions among them", f"{needed * split:,.0f}", border=True)
    st.metric("Cortex Analyst cost", f"{needed * CREDITS_PER_QUESTION:,.0f} credits", border=True)
    st.metric(f"Power of this run ({run_n:,} questions)", f"{run_power:.0%}", border=True)

# ----------------------------------------------------------------------------- chart 1: the curves
st.subheader("Questions needed, by power")


def axis_config(chart):
    return chart.configure(background="transparent", font="Segoe UI, Open Sans, system-ui, sans-serif").configure_view(
        stroke=None).configure_axis(domainColor=GRID, tickColor=GRID, gridColor=GRID, labelColor=SUB, titleColor=SUB,
                                    labelFontSize=11, titleFontSize=12, titleFontWeight="normal", labelPadding=6)


grid = [p / 1000 for p in range(50, 991, 5)]
curves = pd.DataFrame([dict(power=p, questions=questions_needed(p, e / 100, split, alpha), effect=e,
                            label=f"{e:g} pts", chosen=e == effect_pts)
                       for e in EFFECTS if e / 100 < split for p in grid])
curves["credits"] = curves["questions"] * CREDITS_PER_QUESTION
x = alt.X("power:Q", title="Power", scale=alt.Scale(domain=[0.05, 0.99]), axis=alt.Axis(format="%", grid=False))
Y_MIN = 50  # lines below this are clipped at the axis rather than drawn past it
y = alt.Y("questions:Q", title="Paired questions needed (log scale)",
          scale=alt.Scale(type="log", domain=[Y_MIN, max(10_000, curves["questions"].max())]),
          axis=alt.Axis(values=[50, 100, 200, 500, 1000, 2000, 5000, 10000], format=",d"))
others = alt.Chart(curves[~curves["chosen"]]).mark_line(color=SLATE, strokeWidth=1.5, strokeDash=[5, 4], clip=True).encode(
    x=x, y=y, detail="label:N")
other_labels = alt.Chart(curves[~curves["chosen"] & (curves["power"] == grid[-1])]).mark_text(
    align="left", dx=6, color=SUB, fontSize=11).encode(x=x, y=y, text="label:N")
chosen_curve = curves[curves["chosen"]]
main = alt.Chart(chosen_curve).mark_line(color=TEAL, strokeWidth=3, strokeCap="round", clip=True).encode(x=x, y=y)
main_label = alt.Chart(chosen_curve[chosen_curve["power"] == grid[-1]]).mark_text(
    align="left", dx=6, color=INK, fontSize=11, fontWeight=600).encode(x=x, y=y, text="label:N")
layers = [others, other_labels, main, main_label]
if show_exact:
    exact = exact_power(effect, split, alpha, int(min(20_000, questions_needed(0.99, effect, split, alpha) * 1.1)))
    exact = exact[(exact["power"] >= 0.05) & (exact["power"] <= 0.99)]
    layers.append(alt.Chart(exact).mark_line(color=TEAL_LIGHT, strokeWidth=2, strokeDash=[1, 3], strokeCap="round", clip=True)
                  .encode(x=x, y=y, order="questions:Q"))
hover = alt.selection_point(fields=["power"], encodings=["x"], nearest=True, on="pointerover", clear="pointerout",
                            empty=False)
catcher = alt.Chart(chosen_curve[chosen_curve["questions"] >= Y_MIN]).mark_point(size=400, opacity=0).encode(
    x=x, y=y, tooltip=[alt.Tooltip("power:Q", title="Power", format=".1%"),
                       alt.Tooltip("questions:Q", title="Questions needed", format=",.0f"),
                       alt.Tooltip("credits:Q", title="Analyst credits", format=",.0f")]).add_params(hover)
crosshair = alt.Chart(chosen_curve).mark_rule(color=SUB, strokeWidth=1).encode(x=x).transform_filter(hover)
designs = pd.DataFrame([dict(power=power_of(n, effect, split, alpha), questions=n, short=s, what=w,
                             colour=AMBER if n == run_n else SUB) for n, s, w in DESIGNS])
designs = designs[(designs["power"] >= 0.05) & (designs["power"] <= 0.99)]
design_dots = alt.Chart(designs).mark_point(filled=True, size=110, opacity=1, stroke=SURFACE, strokeWidth=2).encode(
    x=x, y=y, color=alt.Color("colour:N", scale=None),
    tooltip=[alt.Tooltip("what:N", title="Test"), alt.Tooltip("questions:Q", title="Questions", format=","),
             alt.Tooltip("power:Q", title=f"Power at {effect_pts:g} pts", format=".0%")])
design_text = alt.Chart(designs).mark_text(align="right", dx=-9, dy=-9, fontSize=11, stroke=SURFACE, strokeWidth=4,
                                           strokeJoin="round").encode(x=x, y=y, text="short:N")
design_text_fill = alt.Chart(designs).mark_text(align="right", dx=-9, dy=-9, fontSize=11).encode(
    x=x, y=y, text="short:N", color=alt.Color("colour:N", scale=None))
pick = pd.DataFrame([dict(power=power, questions=needed)])
pick_rule = alt.Chart(pick).mark_rule(color=INK, strokeWidth=1, opacity=0.35).encode(x=x)
pick_dot = alt.Chart(pick).mark_point(filled=True, size=150, opacity=1, color=TEAL, stroke=SURFACE,
                                      strokeWidth=2).encode(x=x, y=y)
chart1 = alt.layer(*layers, pick_rule, crosshair, catcher, design_dots, design_text, design_text_fill, pick_dot)
st.altair_chart(axis_config(chart1.properties(height=380)), theme=None, width="stretch",
                alt=f"Questions needed rise steeply with power and as the difference shrinks; {needed:,} questions "
                    f"give {power:.0%} power for {effect_pts:g} points")
legend = (f":material/horizontal_rule: bold teal: {effect_pts:g} points, the difference you picked · dashed grey: the "
          "other differences" + (" · dotted: the exact test" if show_exact else "") +
          " · dots: tests I ran (amber: this run)")
st.caption(legend)
half_width = Z.inv_cdf(1 - alpha / 2) * math.sqrt(split / run_n) * 100
st.caption(
    f"To catch a real {effect_pts:g}-point difference {power:.0%} of the time, the test needs {needed:,} paired "
    f"questions. The held-out test's 175 questions had only {power_of(175, effect, split, alpha):.0%} power for that "
    f"difference, which is why its \"no significant difference\" settled little. This run's {run_n:,} questions "
    f"have {run_power:.0%}, and pin the difference down to within about ±{half_width:.1f} points whatever it is."
    + (f" Measured afterwards, {RUN['split_share']:.1%} of the run's questions were split decisions, which gives it "
       f"{RUN['connor_power']:.1%} power at 3.4 points by this formula and {RUN['exact_power']:.1%} by the exact test."
       if RUN else "")
)

# ----------------------------------------------------------------------------- chart 2: each round's answer
st.subheader("What each round found")
rounds = pd.DataFrame(ROUNDS)
rounds["order"] = range(len(rounds))
rounds["name"] = rounds["label"] + " (" + rounds["n"].map("{:,}".format) + ")"
for col in ("diff", "lo95", "hi95", "lo90", "hi90"):
    rounds[col + "_pts"] = rounds[col] * 100


def verdict(r):
    if r.lo90_pts > -effect_pts and r.hi90_pts < effect_pts:
        return f"Equivalent within ±{effect_pts:g}"
    if r.lo95_pts > 0 or r.hi95_pts < 0:
        return "Different"
    return "Inconclusive"


rounds["verdict"] = rounds.apply(verdict, axis=1)
rounds["p_text"] = rounds["exact_p"].map(fmt_p)
rounds["colour"] = [AMBER if s == "Power run" else TEAL for s in rounds["short"]]
span = max(12.0, float(rounds[["lo95_pts", "hi95_pts"]].abs().max().max()) + 1)
XT = "Analyst minus Assistant, points of accuracy (right favours Analyst)"
fscale = alt.Scale(domain=[-span, span])
fx = alt.X("diff_pts:Q", title=XT, scale=fscale)
fy = alt.Y("name:N", title=None, sort=alt.SortField("order"), axis=alt.Axis(labelLimit=260, ticks=False, domain=False))
band = alt.Chart(pd.DataFrame([dict(lo=-effect_pts, hi=effect_pts)])).mark_rect(color=TEAL, opacity=0.10).encode(
    x=alt.X("lo:Q", title=XT, scale=fscale), x2="hi:Q")
zero = alt.Chart(pd.DataFrame([dict(z=0)])).mark_rule(color=SUB, strokeWidth=1).encode(x=alt.X("z:Q", title=XT, scale=fscale))
base = alt.Chart(rounds).encode(y=fy)
ci95 = base.mark_rule(color=SLATE, strokeWidth=1.5).encode(x=alt.X("lo95_pts:Q", title=XT, scale=fscale), x2="hi95_pts:Q")
ci90 = base.mark_rule(strokeWidth=5, strokeCap="round").encode(x=alt.X("lo90_pts:Q", title=XT, scale=fscale), x2="hi90_pts:Q",
                                                                color=alt.Color("colour:N", scale=None))
dots = base.mark_point(filled=True, size=90, opacity=1, stroke=SURFACE, strokeWidth=2).encode(
    x=fx, color=alt.Color("colour:N", scale=None),
    tooltip=[alt.Tooltip("label:N", title="Round"), alt.Tooltip("note:N", title="What"),
             alt.Tooltip("n:Q", title="Questions", format=","),
             alt.Tooltip("assistant_accuracy:Q", title="Assistant", format=".1%"),
             alt.Tooltip("analyst_accuracy:Q", title="Analyst", format=".1%"),
             alt.Tooltip("diff_pts:Q", title="Analyst minus Assistant", format="+.1f"),
             alt.Tooltip("lo95_pts:Q", title="95% interval from", format="+.1f"),
             alt.Tooltip("hi95_pts:Q", title="95% interval to", format="+.1f"),
             alt.Tooltip("p_text:N", title="McNemar p"), alt.Tooltip("verdict:N", title="Verdict")])
st.altair_chart(axis_config(alt.layer(band, zero, ci95, ci90, dots).properties(height=52 * len(rounds))),
                theme=None, width="stretch",
                alt="Analyst-minus-Assistant difference for each round with 95% and 90% intervals over the equivalence band")
st.dataframe(rounds[["label", "n", "diff_pts", "lo95_pts", "hi95_pts", "p_text", "verdict"]], hide_index=True,
             alt="Each round's difference, interval, McNemar p-value and verdict",
             column_config={"label": "Round", "n": st.column_config.NumberColumn("Questions", format="localized"),
                            "diff_pts": st.column_config.NumberColumn("Analyst − Assistant", format="%+.1f pts"),
                            "lo95_pts": st.column_config.NumberColumn("95% from", format="%+.1f"),
                            "hi95_pts": st.column_config.NumberColumn("95% to", format="%+.1f"),
                            "p_text": "McNemar p", "verdict": "Verdict"})
st.caption(
    f"Thin line: 95% interval. Thick bar: 90% interval, the one the equivalence test uses. Shaded: the ±{effect_pts:g}-point "
    "margin, set by the difference picker above. A round counts as **equivalent** when its 90% interval sits inside the "
    "margin (two one-sided tests), **different** when its 95% interval misses zero, and **inconclusive** otherwise. "
    "Intervals are Newcombe's method for paired proportions; right of zero favours Analyst."
)
if RUN:
    hand = next(c for c in RUN["checks"] if c["check"].startswith("Hand-written"))
    st.markdown(
        f"**The power run finds a difference.** Analyst got {RUN['analyst_accuracy']:.1%} of the {RUN['n']:,} questions "
        f"right and the Assistant {RUN['assistant_accuracy']:.1%}. Analyst won {RUN['analyst_only']} of the "
        f"{RUN['assistant_only'] + RUN['analyst_only']} split decisions, a lead of {RUN['diff'] * 100:.1f} points (95% "
        f"interval {RUN['lo95'] * 100:.1f} to {RUN['hi95'] * 100:.1f}, McNemar p {fmt_p(RUN['exact_p'])}). On the 175 "
        f"hand-written held-out questions alone the two were level, {hand['analyst_only']} to {hand['assistant_only']}: "
        f"{hand['analyst_only'] + hand['assistant_only']} split decisions were too few to tell them apart, which is what "
        "the bigger run was for."
    )

# ----------------------------------------------------------------------------- chart 2b: by question type
if RUN and RUN.get("types"):
    st.subheader("Where the gap comes from")
    types = pd.DataFrame(RUN["types"])
    types["name"] = types["type"].map(TYPE_NAMES) + " (" + types["n"].map("{:,}".format) + ")"
    for col in ("diff", "lo95", "hi95"):
        types[col + "_pts"] = types[col] * 100
    types["p_text"], types["holm_text"] = types["exact_p"].map(fmt_p), types["holm_p"].map(fmt_p)
    types["colour"] = np.where(types["holm_p"] < 0.05, TEAL, SLATE)
    types["order"] = types["diff_pts"].rank(ascending=False, method="first")
    tspan = max(12.0, float(types[["lo95_pts", "hi95_pts"]].abs().max().max()) + 2)
    tscale = alt.Scale(domain=[-tspan, tspan])
    TT = "Analyst minus Assistant, points of accuracy"
    ty = alt.Y("name:N", title=None, sort=alt.SortField("order"), axis=alt.Axis(labelLimit=280, ticks=False, domain=False))
    tbase = alt.Chart(types).encode(y=ty)
    tzero = alt.Chart(pd.DataFrame([dict(z=0)])).mark_rule(color=SUB, strokeWidth=1).encode(
        x=alt.X("z:Q", title=TT, scale=tscale))
    tci = tbase.mark_rule(strokeWidth=3, strokeCap="round").encode(
        x=alt.X("lo95_pts:Q", title=TT, scale=tscale), x2="hi95_pts:Q", color=alt.Color("colour:N", scale=None))
    tdots = tbase.mark_point(filled=True, size=80, opacity=1, stroke=SURFACE, strokeWidth=2).encode(
        x=alt.X("diff_pts:Q", title=TT, scale=tscale), color=alt.Color("colour:N", scale=None),
        tooltip=[alt.Tooltip("name:N", title="Question type"),
                 alt.Tooltip("assistant_accuracy:Q", title="Assistant", format=".1%"),
                 alt.Tooltip("analyst_accuracy:Q", title="Analyst", format=".1%"),
                 alt.Tooltip("assistant_only:Q", title="Only the Assistant right"),
                 alt.Tooltip("analyst_only:Q", title="Only Analyst right"),
                 alt.Tooltip("diff_pts:Q", title="Analyst minus Assistant", format="+.1f"),
                 alt.Tooltip("lo95_pts:Q", title="95% interval from", format="+.1f"),
                 alt.Tooltip("hi95_pts:Q", title="95% interval to", format="+.1f"),
                 alt.Tooltip("p_text:N", title="McNemar p"), alt.Tooltip("holm_text:N", title="Holm-adjusted p")])
    st.altair_chart(axis_config(alt.layer(tzero, tci, tdots).properties(height=34 * len(types))), theme=None,
                    width="stretch", alt="Analyst-minus-Assistant difference by question type with 95% intervals")
    st.caption(
        "Each type of question on its own, largest gap first, with 95% intervals. Ten types are ten tests, so the "
        "colour uses Holm-adjusted p-values: **teal** where the gap survives the adjustment, **grey** where it doesn't. "
        "The intervals themselves aren't adjusted, which is why lookups and rankings sit just clear of zero yet stay grey."
    )
    rest = types[types["type"] != "unanswerable"]
    st.markdown(
        "The biggest gap is questions the data can't answer, such as a customer count for Lithuania or an email open "
        "rate. Analyst says it doesn't hold that; the Assistant usually answers with the nearest figure it has. Analyst "
        "also leads on breakdowns (every segment's value at once) and on the model's backtest scores. Leave out the "
        f"unanswerable questions and Analyst still leads, {rest['analyst_only'].sum()} split decisions to "
        f"{rest['assistant_only'].sum()}."
    )
    with st.expander("Table view", icon=":material/table:"):
        st.dataframe(types.sort_values("order")[["name", "assistant_accuracy", "analyst_accuracy", "assistant_only",
                                                 "analyst_only", "diff_pts", "p_text", "holm_text"]],
                     hide_index=True, alt="Each question type's accuracies, split decisions and p-values",
                     column_config={"name": "Question type",
                                    "assistant_accuracy": st.column_config.NumberColumn("Assistant", format="percent"),
                                    "analyst_accuracy": st.column_config.NumberColumn("Analyst", format="percent"),
                                    "assistant_only": "Only the Assistant right", "analyst_only": "Only Analyst right",
                                    "diff_pts": st.column_config.NumberColumn("Analyst − Assistant", format="%+.1f pts"),
                                    "p_text": "McNemar p", "holm_text": "Holm-adjusted p"})

# ----------------------------------------------------------------------------- chart 3: the test itself
st.subheader("How McNemar's test reads a round")
pick_round = st.segmented_control("Round", list(rounds["short"]), default=rounds["short"].iloc[-1], required=True)
r = rounds[rounds["short"] == pick_round].iloc[0]
d, k = int(r.assistant_only + r.analyst_only), int(r.analyst_only)
with st.container(horizontal=True):
    st.metric("Both right", f"{r.both_right:,}", border=True)
    st.metric("Only the Assistant right", f"{r.assistant_only:,}", border=True)
    st.metric("Only Analyst right", f"{r.analyst_only:,}", border=True)
    st.metric("Both wrong", f"{r.both_wrong:,}", border=True)
ks = np.arange(d + 1)
lf = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, d + 1)))]) if d else np.array([0.0])
null = np.exp(log_choose(d, lf) + d * math.log(0.5))
lo_k, hi_k = min(k, d - k), max(k, d - k)
coin = pd.DataFrame(dict(wins=ks, chance=null,
                         kind=np.where(ks == k, "Observed", np.where((ks <= lo_k) | (ks >= hi_k), "As lopsided", "Less lopsided"))))
coin_chart = alt.Chart(coin).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
    x=alt.X("wins:Q", title=f"Split decisions Analyst wins, out of {d}", scale=alt.Scale(domain=[-0.5, d + 0.5])),
    y=alt.Y("chance:Q", title="Chance with a fair coin", axis=alt.Axis(format="%", tickCount=4)),
    color=alt.Color("kind:N", scale=alt.Scale(domain=["Observed", "As lopsided", "Less lopsided"],
                                              range=[AMBER, TEAL, "#3c5a7d"]),
                    legend=alt.Legend(title=None, orient="top", labelColor=SUB)),
    tooltip=[alt.Tooltip("wins:Q", title="Analyst wins"), alt.Tooltip("chance:Q", title="Chance", format=".2%"),
             alt.Tooltip("kind:N", title="")])
st.altair_chart(axis_config(coin_chart.properties(height=240)), theme=None, width="stretch",
                alt=f"Fair-coin distribution of {d} split decisions with the observed {k} to {d - k} marked")
st.caption(
    f"The {r.both_right:,} questions both got right and {r.both_wrong:,} both got wrong say nothing about which is better, so "
    f"the test sets them aside. Of the {d} split decisions Analyst won {k} and the Assistant {d - k}. If the systems were "
    f"equally good each would be a coin flip; a split at least this lopsided, either way, comes up with probability "
    f"**{fmt_p(r.exact_p)}** (the coloured bars"
    + (", too thin to see at this size)." if r.exact_p < 0.001 else ").")
)

with st.expander("How the numbers are worked out", icon=":material/functions:"):
    st.markdown(
        "- **Questions needed** (Connor 1987), for a difference *δ* in accuracy and a share *ψ* of split decisions: "
        "$n = \\left(z_{1-\\alpha/2}\\sqrt{\\psi} + z_{1-\\beta}\\sqrt{\\psi-\\delta^2}\\right)^2 / \\delta^2$.\n"
        "- **Exact power** averages, over the number of split decisions *d* ~ Binomial(*n*, *ψ*), the chance the exact "
        "binomial test rejects when each split goes Analyst's way with probability ½ + *δ*/(2*ψ*).\n"
        "- **Intervals** for the paired difference use Newcombe's method 10; the equivalence test is two one-sided tests "
        "at 5%, read off the 90% interval.\n"
        f"- **Split-decision share** *ψ* = {PILOT_SPLIT:.1%}: 27 of the 175 held-out questions after both systems' fixes."
        + (f"\n- **After the run:** {RUN['assistant_only'] + RUN['analyst_only']} of the {RUN['n']:,} questions "
           f"({RUN['split_share']:.1%}) were split decisions. At that share the run's power to detect 3.4 points was "
           f"{RUN['connor_power']:.1%} by Connor's formula and {RUN['exact_power']:.1%} by the exact test, just under the "
           "planned 90% because the exact test is a little cautious." if RUN else "")
    )
    table = curves[curves["chosen"] & curves["power"].isin([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99])][
        ["power", "questions", "credits"]]
    st.dataframe(table, hide_index=True, alt="Questions and Analyst credits needed at selected power levels",
                 column_config={"power": st.column_config.NumberColumn("Power", format="percent"),
                                "questions": st.column_config.NumberColumn("Questions needed", format="%.0f"),
                                "credits": st.column_config.NumberColumn("Analyst credits", format="%.0f")})


# ----------------------------------------------------------------------------- more statistics
def binom_ge(m, n, p):
    """P(X >= m) for X ~ Binomial(n, p)."""
    if m <= 0:
        return 1.0
    if m > n or p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    return sum(math.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * math.log(p)
                        + (n - j) * math.log(1 - p)) for j in range(m, n + 1))


def clopper_pearson(k, n, alpha=0.05):
    """Exact interval for a binomial share: the shares the exact test wouldn't reject, found by bisection."""
    def solve(f, target):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: binom_ge(k, n, p), alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: binom_ge(k + 1, n, p), 1 - alpha / 2)
    return lower, upper


st.subheader("More statistics")
st.caption(f"For the round picked above: {r.label}, {d} split decisions, Analyst {k} and the Assistant {d - k}.")

m_wins = max(k, d - k)
point = math.exp(math.lgamma(d + 1) - math.lgamma(m_wins + 1) - math.lgamma(d - m_wins + 1) + d * math.log(0.5)) if d else 1.0
mid_p = min(1.0, 2 * (binom_ge(m_wins, d, 0.5) - point / 2))
b_, c_ = int(r.assistant_only), int(r.analyst_only)
chi_cc = (abs(b_ - c_) - 1) ** 2 / (b_ + c_) if b_ + c_ else 0.0
chi_raw = (b_ - c_) ** 2 / (b_ + c_) if b_ + c_ else 0.0
versions = pd.DataFrame([
    dict(version="Exact binomial (the one used)", statistic=None, p=r.exact_p,
         note="Adds up the coin-flip probabilities directly. Never too optimistic; slightly cautious."),
    dict(version="Mid-p", statistic=None, p=mid_p,
         note="Counts only half of the observed split's own probability. Less cautious, close to the nominal 5%."),
    dict(version="Chi-square, continuity-corrected", statistic=chi_cc, p=math.erfc(math.sqrt(chi_cc / 2)) if chi_cc else 1.0,
         note="McNemar's original shortcut. Trustworthy once there are about 25 split decisions."),
    dict(version="Chi-square, uncorrected", statistic=chi_raw, p=math.erfc(math.sqrt(chi_raw / 2)) if chi_raw else 1.0,
         note="The same shortcut without the correction; runs optimistic when split decisions are few."),
])
versions["p"] = versions["p"].map(fmt_p)
st.markdown("**Four versions of the same test.** They differ only in how they turn the split into a p-value.")
st.dataframe(versions, hide_index=True, alt="Exact, mid-p and chi-square McNemar p-values for the picked round",
             column_config={"version": "Version", "statistic": st.column_config.NumberColumn("Statistic", format="%.2f"),
                            "p": "p", "note": "What it does"})
st.caption(f"This round has {d} split decisions, so the shortcuts are "
           f"{'reliable here' if d >= 25 else 'shaky here, and the exact version is the one to trust'}.")

half = d / 2
folded = coin.assign(distance=(coin["wins"] - half).abs()).groupby("distance", as_index=False)["chance"].sum()
obs_dist = abs(k - half)
folded["kind"] = np.where(np.isclose(folded["distance"], obs_dist), "Observed",
                          np.where(folded["distance"] > obs_dist, "Further from even", "Closer to even"))
fold_chart = alt.Chart(folded).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
    x=alt.X("distance:Q", title=f"Distance from an even split ({half:g} each), in split decisions",
            scale=alt.Scale(domain=[-0.5, folded["distance"].max() + 0.5])),
    y=alt.Y("chance:Q", title="Chance with a fair coin", axis=alt.Axis(format="%", tickCount=4)),
    color=alt.Color("kind:N", scale=alt.Scale(domain=["Observed", "Further from even", "Closer to even"],
                                              range=[AMBER, TEAL, "#3c5a7d"]),
                    legend=alt.Legend(title=None, orient="top", labelColor=SUB)),
    tooltip=[alt.Tooltip("distance:Q", title="Distance from even"), alt.Tooltip("chance:Q", title="Chance", format=".2%")])
st.markdown(
    "**Two-sided, as one tail.** A two-sided test asks how far the split is from even, in either direction. Fold "
    "the coin-flip chart at its middle and each bar joins its mirror image; the p-value is then a single tail, the "
    f"amber and teal bars here, adding up to {fmt_p(r.exact_p)}. The chi-square versions do the same to a normal curve: "
    "their statistic is a squared z-score, and squaring throws away the sign."
)
st.altair_chart(axis_config(fold_chart.properties(height=200)), theme=None, width="stretch",
                alt="The coin-flip distribution folded at an even split, with the observed distance marked")

cp_lo, cp_hi = clopper_pearson(k, d)
with st.container(horizontal=True):
    st.metric("Analyst's share of split decisions", f"{k / d:.1%}" if d else "—", border=True)
    st.metric("Exact 95% interval for that share", f"{cp_lo:.1%} to {cp_hi:.1%}", border=True)
st.markdown(
    "**The same test as an interval.** Every share of split decisions that the exact test would *not* reject forms "
    "the Clopper–Pearson interval. The test finds a difference exactly when that interval leaves out 50%"
    f"{': here it does.' if cp_lo > 0.5 or cp_hi < 0.5 else ': here it includes 50%, so no difference is found.'}"
)

checks = ROUNDS[[x["short"] for x in ROUNDS].index(pick_round)].get("checks")
if checks:
    st.markdown("**Does the result survive its assumptions?** The same comparison, redone each way the analysis "
                "could be wrong.")
    table = pd.DataFrame(checks)
    table["p_text"] = table["exact_p"].map(fmt_p)
    st.dataframe(table[["check", "n", "assistant_only", "analyst_only", "p_text", "why"]], hide_index=True,
                 alt="The power run's result under each robustness check",
                 column_config={"check": "Check", "n": st.column_config.NumberColumn("Units", format="localized"),
                                "assistant_only": "Only Assistant right", "analyst_only": "Only Analyst right",
                                "p_text": "p", "why": "Why it matters"})
    kappa = {k["system"]: k for k in ROUNDS[[x["short"] for x in ROUNDS].index(pick_round)].get("kappa", [])}
    if kappa:
        st.caption(
            f"The machine first pass agreed with the read on {kappa['assistant']['agreement']:.1%} of the Assistant's "
            f"answers and {kappa['analyst']['agreement']:.1%} of Analyst's. Cohen's kappa, which discounts agreement "
            f"expected by chance, is {kappa['assistant']['kappa']:.2f} and {kappa['analyst']['kappa']:.2f}: substantial "
            "and moderate. The read is the grade that counts. It was done knowing which system wrote each answer (the "
            "held-out old-vs-fixed comparison was graded blind; these 2,800 answers weren't), with the same rules for "
            "both systems, checked by script."
        )
