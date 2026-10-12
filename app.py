"""Segment Assistant: ask questions about the segments or have it draft persona cards and campaigns.

    streamlit run app.py

Offline by default: DuckDB over results/*.csv and a local Ollama model (qwen2.5:7b). With
Snowflake credentials in .env it queries the ML schema live and uses Snowflake Cortex.
See assistant/data.py and assistant/llm.py for the environment variables. Colors and fonts are in
.streamlit/config.toml.

Two pages: Ask (the assistant, chat box pinned to the bottom) and Segments (charts and table).
"""
import time
import uuid
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from assistant import llm as llm_factory
from assistant.data import open_data
from assistant.graph import build_graph, turn_input
from segmentation.naming import BEHAVIOR_NAMES, TIER_NAMES

REPO_URL = "https://github.com/livingdw67/snowpark-customer-segmentation"
IMAGES = Path(__file__).resolve().parent / "docs" / "img"
HEADLINE = "Segments tied to what a customer is worth next quarter."
AUTHOR = "Daniel Livingston"
ROLE = "Senior Data Scientist · Finance & Energy"

# Grouped by agent, so the chips also show what the assistant can do.
ASK_THE_DATA = {
    "Most valuable segment": "Which segment holds the most expected revenue next quarter?",
    "Hybrid vs. RFM": "How does the hybrid compare with the RFM baseline in the backtests?",
    "Holiday buyers": "How many customers are holiday buyers in total?",
}
CREATE = {
    "Persona card": "Write a persona card for Top spenders / Holiday buyers",
    "Win-back email": "Draft a win-back email for Low value / Bulk buyers",
    "Q4 campaign": "Plan a Q4 campaign for Core / Holiday buyers",
}
CHIP_ICONS = {"Most valuable segment": "query_stats", "Hybrid vs. RFM": "compare_arrows",
              "Holiday buyers": "redeem", "Persona card": "badge", "Win-back email": "mail",
              "Q4 campaign": "campaign"}
CHIP_KEYS = ("chips_ask", "chips_create")

# Brand colors for charts (the theme sets everything else).
NAVY, RAISED, LINE, TEAL, AMBER, SLATE, MUTED, WHITE = (
    "#0e1a2b", "#16283f", "#2b3f58", "#5fc4c0", "#f0a742", "#9fb1c7", "#6f86a3", "#f2f5f9")

st.set_page_config(page_title="Segment Assistant", page_icon=":material/insights:",
                   layout="centered", initial_sidebar_state="collapsed")


@st.cache_resource(show_spinner=False)
def load():
    data = open_data()
    llm = llm_factory.make_llm(getattr(data, "session", None))
    llm_factory.warm_up(llm)
    return data, llm, build_graph(llm, data)


def new_conversation() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []
    for key in CHIP_KEYS:  # otherwise a previously picked chip would fire again
        st.session_state.pop(key, None)


# --------------------------------------------------------------------------------------
# Agent steps
# --------------------------------------------------------------------------------------

def step_label(s: dict) -> str:
    if s["title"] == "Intent":
        intent = s["detail"].split(":")[0]
        return {"analytical": "Routed to the calculation agent", "methodology": "Routed to the documentation agent",
                "generative": "Routed to the writing agent"}.get(intent, "Outside what this assistant covers")
    return s["title"] + (f" · {s['detail']}" if s["detail"] and s["kind"] == "table" else "")


def render_step(s: dict) -> None:
    failed = s["title"].endswith("failed")
    with st.status(step_label(s), type="step", state="error" if failed else "complete"):
        if s["kind"] == "text":
            st.caption(s["detail"])
        if s["kind"] in ("sql", "plan") and s["data"]:
            if failed or s["detail"]:
                st.caption(s["detail"])
            st.code(s["data"], language="sql" if s["kind"] == "sql" else "json")
        elif s["kind"] == "table" and s["data"] is not None:
            st.dataframe(pd.DataFrame(s["data"]), hide_index=True, width="stretch")


def render_trace(trace: list[dict], seconds: float | None = None) -> None:
    label = f"Worked for {seconds:.0f}s" if seconds else "How I got this"
    with st.status(label, type="compact", state="complete", expanded=False):
        for s in trace:
            render_step(s)


def run_turn(graph, question: str) -> tuple[str, list[dict], float]:
    """Stream the graph, adding each agent step to the timeline as it completes."""
    trace, answer, start = [], "", time.time()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.status(":shimmer[Reading your question]", type="compact", expanded=True) as status:
        for update in graph.stream(turn_input(question), config, stream_mode="updates"):
            for node, out in update.items():
                for s in out.get("trace", [])[len(trace):]:
                    render_step(s)
                trace = out.get("trace", trace)
                if out.get("messages"):
                    answer = out["messages"][-1].content
                if node == "router":
                    verb = {"analytical": "Calculating", "methodology": "Reading the documentation",
                            "generative": "Writing"}.get(out["intent"], "Checking")
                    status.update(label=f":shimmer[{verb}]")
        seconds = time.time() - start
        status.update(label=f"Worked for {seconds:.0f}s", state="complete", expanded=False)
    return answer, trace, seconds


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------

def value_grid(seg: pd.DataFrame, height: int = 360) -> alt.Chart:
    """Where next quarter's predicted value sits: value tier x buying style."""
    df = seg.assign(
        VALUE_SHARE=seg.EXPECTED_REVENUE / seg.EXPECTED_REVENUE.sum(),
        TIER=seg.VALUE_TIER.map(TIER_NAMES) + " (" + seg.VALUE_TIER + ")",
        STYLE=seg.BEHAVIOR_SEGMENT.map(BEHAVIOR_NAMES),
    )
    df["LABEL"] = (df.VALUE_SHARE * 100).round().astype(int).astype(str) + "% of value"
    df["SUB"] = df.CUSTOMERS.map("{:,} customers".format)
    df["TOP"] = df.VALUE_SHARE == df.VALUE_SHARE.max()
    tiers = [f"{v} ({k})" for k, v in TIER_NAMES.items()]
    styles = list(BEHAVIOR_NAMES.values())

    base = alt.Chart(df).encode(
        x=alt.X("STYLE:N", sort=styles, title=None,
                axis=alt.Axis(orient="top", labelColor=SLATE, labelFontSize=13, labelAngle=0, ticks=False, domain=False)),
        y=alt.Y("TIER:N", sort=tiers, title=None,
                axis=alt.Axis(labelColor=SLATE, labelFontSize=13, ticks=False, domain=False, labelPadding=10)),
    )
    cells = base.mark_rect(cornerRadius=6, stroke=NAVY, strokeWidth=4).encode(
        color=alt.Color("VALUE_SHARE:Q", legend=None,
                        scale=alt.Scale(range=[RAISED, "#1f4361", "#2f718d", TEAL], domain=[0, 0.12, 0.3, 0.52])),
        tooltip=[alt.Tooltip("SEGMENT:N", title="Segment"), alt.Tooltip("SEGMENT_NAME:N", title="Name"),
                 alt.Tooltip("CUSTOMERS:Q", title="Customers", format=","),
                 alt.Tooltip("VALUE_SHARE:Q", title="Share of value", format=".0%"),
                 alt.Tooltip("EXPECTED_REVENUE:Q", title="Expected revenue (£)", format=",.0f")],
    )
    highlight = base.transform_filter("datum.TOP").mark_rect(
        cornerRadius=6, fill=None, stroke=AMBER, strokeWidth=2.5)
    on_bright = "datum.VALUE_SHARE > 0.3"
    label = base.mark_text(dy=-8, fontSize=15, fontWeight=600).encode(
        text="LABEL:N", color=alt.condition(on_bright, alt.value(NAVY), alt.value(WHITE)))
    sub = base.mark_text(dy=11, fontSize=12).encode(
        text="SUB:N", color=alt.condition(on_bright, alt.value(NAVY), alt.value(SLATE)))
    return (cells + highlight + label + sub).properties(height=height).configure_view(stroke=None)


def method_tradeoff(backtest: pd.DataFrame) -> alt.Chart:
    """Revenue concentration vs. buying-style separation, mean of the two backtest folds."""
    names = {"rfm_score_baseline": "RFM scores", "two_axis_value": "B: value axis",
             "two_axis_grid": "B: value × behavior", "outcome_guided": "C: outcome tiers (3–4)",
             "outcome_guided_grid_size": "C: outcome tiers (9–12)", "hybrid": "Hybrid"}
    df = (backtest[backtest.SCHEME.isin(names)]
          .groupby("SCHEME", as_index=False)[["TOP20_REVENUE_CAPTURE", "BEHAVIOR_ETA2", "NEXT_ACTIVE_AUC"]].mean())
    df["NAME"] = df.SCHEME.map(names)
    df["HYBRID"] = df.SCHEME == "hybrid"

    axis = dict(labelColor=MUTED, titleColor=SLATE, gridColor=LINE, domainColor=LINE, tickColor=LINE,
                titleFontWeight=400, titleFontSize=13, format=".0%")
    base = alt.Chart(df).encode(
        x=alt.X("TOP20_REVENUE_CAPTURE:Q", title="Next-quarter revenue held by the top 20% →",
                scale=alt.Scale(domain=[0.5, 0.75]), axis=alt.Axis(**axis)),
        y=alt.Y("BEHAVIOR_ETA2:Q", title="Buying-style separation (η²) →",
                scale=alt.Scale(domain=[0, 0.4]), axis=alt.Axis(**axis)),
        tooltip=[alt.Tooltip("NAME:N", title="Scheme"),
                 alt.Tooltip("TOP20_REVENUE_CAPTURE:Q", title="Top-20% capture", format=".0%"),
                 alt.Tooltip("BEHAVIOR_ETA2:Q", title="Behavior η²", format=".2f"),
                 alt.Tooltip("NEXT_ACTIVE_AUC:Q", title="Buys-again AUC", format=".3f")],
    )
    halo = base.transform_filter("datum.HYBRID").mark_circle(size=520, color=AMBER, opacity=0.18)
    dots = base.mark_circle(size=130, opacity=1).encode(
        color=alt.condition("datum.HYBRID", alt.value(AMBER), alt.value(TEAL)))
    labels = base.mark_text(align="left", dx=12, fontSize=12).encode(
        text="NAME:N", color=alt.condition("datum.HYBRID", alt.value(WHITE), alt.value(SLATE)))
    return (halo + dots + labels).properties(height=340).configure_view(stroke=None)


def key_numbers(seg: pd.DataFrame) -> None:
    value = seg.EXPECTED_REVENUE / seg.EXPECTED_REVENUE.sum()
    top = seg.loc[value.idxmax()]
    cols = st.columns(3)
    cols[0].metric("Active customers", f"{seg.CUSTOMERS.sum():,}", border=True)
    cols[1].metric("Predicted next-quarter revenue", f"£{seg.EXPECTED_REVENUE.sum() / 1e6:.2f}M", border=True,
                   help="Ranks segments reliably, but under-forecasts holiday quarters: not a budget figure.")
    cols[2].metric(f"Held by {top.SEGMENT_NAME}", f"{value.max():.0%}", border=True,
                   delta=f"{top.SHARE:.0%} of customers", delta_color="off", delta_arrow="off")


# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------

def chips() -> str | None:
    """Starter prompts in two rows, one per agent. Returns the picked prompt, if any."""
    picked = None
    for label, options, key in (("Ask the data", ASK_THE_DATA, "chips_ask"), ("Create", CREATE, "chips_create")):
        left, right = st.columns([1, 5], vertical_alignment="center")
        left.caption(label)
        choice = right.pills(label, list(options), key=key, label_visibility="collapsed",
                             format_func=lambda o: f":material/{CHIP_ICONS[o]}: {o}")
        picked = picked or options.get(choice)
    return picked


def ask_page() -> None:
    data, graph = st.session_state.data, st.session_state.graph
    history = st.session_state.history

    if not history:
        st.title(HEADLINE)
        st.markdown(":orange[━━━━]  \n:gray[Segment Assistant · ask the data or create from it · UCI Online Retail II]")
        seg = data.segments()
        key_numbers(seg)
        question = chips()
        st.space("small")
        st.caption("Where next quarter's predicted value sits, by value tier and buying style")
        st.altair_chart(value_grid(seg, height=300), width="stretch")
    else:
        with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute"):
            st.markdown(f"**{HEADLINE}**", width="content")
            st.button("New chat", icon=":material/add_comment:", on_click=new_conversation)
        question = None
        for msg in history:
            avatar = ":material/person:" if msg["role"] == "user" else ":material/insights:"
            with st.chat_message(msg["role"], avatar=avatar):
                if msg.get("trace"):
                    render_trace(msg["trace"], msg.get("seconds"))
                st.markdown(msg["content"])

    # Outside any container, so Streamlit pins it to the bottom of the page.
    typed = st.chat_input("Ask about the segments, or ask for a persona card or campaign", submit_mode="disable")
    question = typed or question
    if question:
        # Save it and rerun: the next run shows it in the conversation view, then answers it.
        history.append({"role": "user", "content": question})
        st.rerun()


def answer_last(graph, data, question: str) -> None:
    with st.chat_message("assistant", avatar=":material/insights:"):
        try:
            answer, trace, seconds = run_turn(graph, question)
        except Exception as exc:
            hint = " Check that Ollama is running (`ollama ps`)." if data.mode == "local" else ""
            answer, trace, seconds = f"Something went wrong: `{exc}`.{hint}", [], None
        st.markdown(answer)
    st.session_state.history.append({"role": "assistant", "content": answer, "trace": trace, "seconds": seconds})
    st.rerun()


def segments_page() -> None:
    data = st.session_state.data
    seg = data.segments()
    st.header("Where next quarter's value sits")
    st.caption("Predicted revenue for the 91 days after 10 Dec 2011, by value tier and buying style.")
    st.altair_chart(value_grid(seg), width="stretch")
    if "BACKTEST" in data.tables:
        st.header("Why the hybrid")
        st.caption("Mean of two out-of-time backtests. Up and right is better: the hybrid keeps most "
                   "of the revenue concentration and adds the buying-style split.")
        st.altair_chart(method_tradeoff(data.query("SELECT * FROM BACKTEST")), width="stretch")
    st.header("All segments")
    st.dataframe(
        seg[["SEGMENT", "SEGMENT_NAME", "CUSTOMERS", "SHARE", "TRAITS", "MEAN_P_ACTIVE", "EXPECTED_REVENUE"]],
        hide_index=True, width="stretch",
        column_config={
            "SEGMENT": "Code", "SEGMENT_NAME": "Segment",
            "CUSTOMERS": st.column_config.NumberColumn("Customers", format="localized"),
            "SHARE": st.column_config.NumberColumn("Share", format="percent"), "TRAITS": "Traits",
            "MEAN_P_ACTIVE": st.column_config.ProgressColumn("Buys again", format="percent", min_value=0, max_value=1),
            "EXPECTED_REVENUE": st.column_config.NumberColumn("Expected revenue", format="£%,.0f")})


def sidebar(data, llm) -> None:
    with st.sidebar:
        st.markdown(f"#### {AUTHOR}")
        st.caption(ROLE)
        st.markdown(f":material/link: [github.com/livingdw67]({REPO_URL})")
        st.space("medium")
        st.markdown("**Under the hood**")
        st.markdown(f":material/database: {data.mode.title()} data · {data.dialect}  \n"
                    f":material/memory: {llm_factory.describe(llm)}  \n"
                    f":material/table: {len(data.tables)} tables: " + ", ".join(data.tables).lower())
        st.caption("A router sends each request to a calculation, documentation or writing agent. "
                   "Numbers are computed in code, never by the model, and every answer shows its steps.")
        if data.mode == "local":
            st.caption("Offline mode reads the aggregated results in `results/`.")


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------

with st.spinner("Loading the model. The first start can take a minute while it moves onto the GPU."):
    try:
        data, llm, graph = load()
    except Exception as exc:
        st.error(f"Couldn't start the assistant: {exc}. If you're offline, check that Ollama is running "
                 "and the model is pulled (`ollama pull qwen2.5:7b`).")
        st.stop()

if "thread_id" not in st.session_state:
    new_conversation()
st.session_state.data, st.session_state.graph = data, graph

sidebar(data, llm)
# Wordmark in the top bar (next to Ask / Segments), so the app stays anchored while scrolling.
st.logo(str(IMAGES / "wordmark.svg"), size="large", link=REPO_URL, icon_image=str(IMAGES / "mark.svg"))
page = st.navigation([st.Page(ask_page, title="Ask", icon=":material/forum:", default=True),
                      st.Page(segments_page, title="Segments", icon=":material/grid_view:")],
                     position="top")
page.run()

# Answer a question saved on the previous run, below the conversation.
history = st.session_state.history
if page.title == "Ask" and history and history[-1]["role"] == "user":
    answer_last(graph, data, history[-1]["content"])
