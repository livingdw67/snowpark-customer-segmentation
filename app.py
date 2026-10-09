"""Segment Copilot: ask questions about the segments or have it draft persona cards and campaigns.

    streamlit run app.py

Offline by default: DuckDB over results/*.csv and a local Ollama model (qwen2.5:7b). With
Snowflake credentials in .env it queries the ML schema live and uses Snowflake Cortex.
See copilot/data.py and copilot/llm.py for the environment variables.
"""
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from copilot import llm as llm_factory
from copilot.data import open_data
from copilot.graph import build_graph, turn_input

ROOT = Path(__file__).resolve().parent
EXAMPLES = [
    "Which segment holds the most expected revenue next quarter?",
    "How does the hybrid compare with the RFM baseline in the backtests?",
    "How many customers are holiday buyers, and how much do they spend?",
    "Write a persona card for Top spenders / Holiday buyers",
    "Draft a win-back email for Low value / Bulk buyers",
]

st.set_page_config(page_title="Segment Copilot", page_icon="🧭", layout="wide")


@st.cache_resource(show_spinner="Connecting...")
def load():
    data = open_data()
    llm = llm_factory.make_llm(getattr(data, "session", None))
    return data, llm, build_graph(llm, data)


def render_step(s: dict) -> None:
    label = f"**{s['title']}**" + (f": {s['detail']}" if s["detail"] else "")
    st.markdown(label)
    if s["kind"] == "sql" and s["data"]:
        st.code(s["data"], language="sql")
    elif s["kind"] == "table" and s["data"] is not None:
        st.dataframe(pd.DataFrame(s["data"]), hide_index=True, width="stretch")


def run_turn(graph, question: str) -> tuple[str, list[dict]]:
    """Stream the graph, showing each agent step as it completes, then return the answer."""
    trace, answer = [], ""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.status("Routing your request...", expanded=True) as status:
        for update in graph.stream(turn_input(question), config, stream_mode="updates"):
            for node, out in update.items():
                for s in out.get("trace", [])[len(trace):]:
                    render_step(s)
                trace = out.get("trace", trace)
                if out.get("messages"):
                    answer = out["messages"][-1].content
                if node == "router":
                    status.update(label=f"Running the {out['intent']} agent...")
        status.update(label="How I got this", state="complete", expanded=False)
    return answer, trace


try:
    data, llm, graph = load()
except Exception as exc:
    st.error(f"Couldn't start the copilot: {exc}")
    st.stop()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []

with st.sidebar:
    st.header("Segment Copilot")
    st.caption(f"Data: **{data.mode}** ({data.dialect})  \nModel: **{llm_factory.describe(llm)}**")
    st.markdown("Tables: " + ", ".join(f"`{t}`" for t in data.tables))
    if data.mode == "local":
        st.caption("Offline mode reads the aggregated results committed in `results/`. "
                   "Customer-level questions need live mode.")
    st.subheader("Try")
    for example in EXAMPLES:
        if st.button(example, width="stretch"):
            st.session_state.pending = example
    if st.button("New conversation", type="primary", width="stretch"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

chat, segments_tab = st.tabs(["Copilot", "Segments"])

with segments_tab:
    seg = data.segments()
    st.dataframe(seg[["SEGMENT", "SEGMENT_NAME", "CUSTOMERS", "SHARE", "TRAITS",
                      "MEAN_P_ACTIVE", "EXPECTED_REVENUE"]],
                 hide_index=True, width="stretch",
                 column_config={"SHARE": st.column_config.NumberColumn(format="percent"),
                                "MEAN_P_ACTIVE": st.column_config.NumberColumn("P(BUYS AGAIN)", format="percent"),
                                "EXPECTED_REVENUE": st.column_config.NumberColumn(format="£%.0f")})
    left, right = st.columns(2)
    left.image(str(ROOT / "docs/img/hybrid_grid.png"))
    right.image(str(ROOT / "docs/img/method_tradeoff.png"))

with chat:
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            if msg.get("trace"):
                with st.expander("How I got this"):
                    for s in msg["trace"]:
                        render_step(s)
            st.markdown(msg["content"])

    question = st.chat_input("Ask about the segments, or ask for a persona card or campaign")
    question = question or st.session_state.pop("pending", None)
    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                answer, trace = run_turn(graph, question)
            except Exception as exc:
                hint = (" Is Ollama running with the model pulled (`ollama pull qwen2.5:7b`)?"
                        if data.mode == "local" else "")
                answer, trace = f"Something went wrong: `{exc}`.{hint}", []
            st.markdown(answer)
        st.session_state.history.append({"role": "assistant", "content": answer, "trace": trace})
