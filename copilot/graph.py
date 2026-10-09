"""LangGraph workflow: an intent router in front of an analytical and a generative agent.

    START -> router -> analytical -> END     questions answered by querying the data
                    -> generative -> END     persona cards, campaigns, copy

Conversation state (messages) persists per thread through the checkpointer, so follow-ups
like "now draft an email for that segment" work. `trace` is rebuilt every turn and records
what each agent did (intent, SQL attempts, rows, retrieved segment stats) for the UI.
"""
from __future__ import annotations

import re
from typing import Annotated, Any, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from segmentation.naming import BEHAVIOR_NAMES, TIER_NAMES

from .data import schema_text

MAX_SQL_ATTEMPTS = 2
HISTORY = 6  # earlier messages passed to each agent

GENERATIVE_WORDS = re.compile(
    r"\b(draft|write|persona|card|campaign|email|e-mail|message|copy|slogan|tagline|pitch|"
    r"brainstorm|ideas?|subject line|newsletter|offer)\b", re.IGNORECASE)


class CopilotState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: str
    sql: str | None
    rows: list[dict] | None
    context: list[dict] | None
    trace: list[dict]


def step(title: str, detail: str = "", kind: str = "text", data: Any = None) -> dict:
    return {"title": title, "detail": detail, "kind": kind, "data": data}


def _question(state) -> str:
    return state["messages"][-1].content


def _history(state) -> list[AnyMessage]:
    return state["messages"][-HISTORY - 1:-1]


def _text(msg) -> str:
    content = msg.content
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content.strip()


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else text).strip()


ROUTER_PROMPT = """You route requests for a customer-segmentation copilot.
analytical: a question answered by looking up or computing numbers in the data (sizes, \
rankings, comparisons, metrics, backtest results, how the segments were chosen).
generative: a request to create content about segments (persona cards, campaign plans, \
emails, messages, offers, ideas).
Reply with exactly one word: analytical or generative."""

SQL_PROMPT = """You write one {dialect} SELECT query that answers the user's latest question.

Tables (use only these, with these exact names):

{schema}

Rules:
- One read-only SELECT (or WITH ... SELECT). No DDL or DML.
- For "top"/"most"/"least" questions, ORDER BY and LIMIT.
- Include the SEGMENT code and SEGMENT_NAME when listing segments.
- Fractions are 0-1 (0.07 = 7%); money is GBP.
Return only the query, in a ```sql block."""

ANSWER_PROMPT = """You answer questions about a customer segmentation using a query result.
Use only figures from the result. Write money as £ with thousands separators and fractions \
as percentages. Name segments as "Top spenders / Bulk buyers (C1-B1)". If the result doesn't \
answer the question, say what's missing. Two to four sentences."""

GENERATIVE_PROMPT = """You are a B2B marketing strategist for a UK online giftware \
wholesaler. Most customers are retailers buying in bulk (median baskets of 60-340 units).

Ground every number you use in the segment statistics below and cite the segment code. \
Don't invent statistics. Timing: a high MEDIAN_Q4_REVENUE_SHARE means the segment buys for \
the October-December season. EXPECTED_REVENUE ranks segments reliably but is under-forecast \
for holiday quarters, so don't present it as a budget.

For a persona card use: Name and code; Who they are (2 sentences); Key numbers (4-6 \
bullets); What they respond to; Risk; Next best action.
For a campaign use: Audience; Goal; Timing; Offer; Channel; Draft (subject line + body under \
120 words); Success metric.

Segment statistics:
{stats}"""

STAT_COLUMNS = ["SEGMENT", "SEGMENT_NAME", "CUSTOMERS", "SHARE", "TRAITS", "MEDIAN_RECENCY_DAYS",
                "MEDIAN_ORDERS", "MEDIAN_NET_REVENUE", "MEDIAN_AVG_UNITS_PER_ORDER",
                "MEDIAN_AVG_UNIT_PRICE", "MEDIAN_RETURN_RATE", "MEDIAN_Q4_REVENUE_SHARE",
                "MEDIAN_AVG_ORDER_VALUE", "UK_SHARE", "MEAN_P_ACTIVE", "EXPECTED_REVENUE"]


def match_segments(text: str, segments: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Segments named in `text` by code, full name, tier name and/or behavior name."""
    codes = {c.upper() for c in re.findall(r"\bC\d-B\d\b", text, re.IGNORECASE)}
    lower = text.lower()
    codes |= {s for s, n in zip(segments.SEGMENT, segments.SEGMENT_NAME) if n.lower() in lower}
    if codes:
        hit = segments[segments.SEGMENT.isin(codes)]
        return hit, f"named segments {', '.join(sorted(codes))}"
    tiers = {k for k, v in TIER_NAMES.items() if re.search(rf"\b{re.escape(v.lower())}\b", lower)}
    styles = {k for k, v in BEHAVIOR_NAMES.items() if re.search(rf"\b{re.escape(v.lower())}\b", lower)}
    mask = pd.Series(True, index=segments.index)
    if tiers:
        mask &= segments.VALUE_TIER.isin(tiers)
    if styles:
        mask &= segments.BEHAVIOR_SEGMENT.isin(styles)
    if tiers or styles:
        names = [TIER_NAMES[t] for t in sorted(tiers)] + [BEHAVIOR_NAMES[b] for b in sorted(styles)]
        return segments[mask], "mentioned " + " + ".join(names)
    return segments.iloc[0:0], ""


def build_graph(llm, data, checkpointer=None):
    def router(state: CopilotState) -> dict:
        question = _question(state)
        reply = _text(llm.invoke([SystemMessage(ROUTER_PROMPT), *_history(state), HumanMessage(question)]))
        word = reply.lower()
        if "generative" in word and "analytical" not in word:
            intent, how = "generative", f"model said '{reply[:40]}'"
        elif "analytical" in word and "generative" not in word:
            intent, how = "analytical", f"model said '{reply[:40]}'"
        else:  # unparseable reply: fall back to keywords
            intent = "generative" if GENERATIVE_WORDS.search(question) else "analytical"
            how = f"keyword fallback (model said '{reply[:40]}')"
        return {"intent": intent, "trace": [step("Intent", f"{intent}: {how}")]}

    def analytical(state: CopilotState) -> dict:
        trace = list(state["trace"])
        question = _question(state)
        prompt = [SystemMessage(SQL_PROMPT.format(dialect=data.dialect, schema=schema_text(data))),
                  *_history(state), HumanMessage(question)]
        sql, rows, error = None, None, None
        for attempt in range(1, MAX_SQL_ATTEMPTS + 1):
            reply = llm.invoke(prompt)
            sql = extract_sql(_text(reply))
            try:
                rows = data.query(sql)
                trace.append(step(f"SQL (attempt {attempt})", kind="sql", data=sql))
                break
            except Exception as exc:  # validation or database error: show it and let the model repair
                error = str(exc).strip().splitlines()[0][:300]
                trace.append(step(f"SQL (attempt {attempt}) failed", error, kind="sql", data=sql))
                prompt += [AIMessage(f"```sql\n{sql}\n```"),
                           HumanMessage(f"That query failed: {error}\nReturn a corrected query.")]

        if rows is None:
            answer = f"I couldn't answer that from the data. The last query failed with: {error}"
            return {"sql": sql, "rows": None, "trace": trace, "messages": [AIMessage(answer)]}

        trace.append(step("Result", f"{len(rows)} row(s)", kind="table", data=rows.to_dict("records")))
        result = rows.to_string(index=False, max_rows=50) if len(rows) else "(no rows)"
        answer = _text(llm.invoke([
            SystemMessage(ANSWER_PROMPT),
            HumanMessage(f"Question: {question}\n\nSQL:\n{sql}\n\nResult:\n{result}"),
        ]))
        return {"sql": sql, "rows": rows.to_dict("records"), "trace": trace,
                "messages": [AIMessage(answer)]}

    def generative(state: CopilotState) -> dict:
        trace = list(state["trace"])
        segments = data.segments()
        hit, how = match_segments(_question(state), segments)
        if hit.empty:  # e.g. "that segment": look at the rest of the conversation
            hit, how = match_segments(" ".join(_text(m) for m in _history(state)), segments)
            how = f"{how} earlier in the conversation" if how else how
        if hit.empty:
            hit, how = segments, "no segment named, using all segments"
        stats = hit[STAT_COLUMNS]
        trace.append(step("Retrieved segment statistics", how, kind="table", data=stats.to_dict("records")))

        answer = _text(llm.invoke([
            SystemMessage(GENERATIVE_PROMPT.format(stats=stats.to_string(index=False))),
            *_history(state), HumanMessage(_question(state)),
        ]))
        return {"context": stats.to_dict("records"), "trace": trace, "messages": [AIMessage(answer)]}

    graph = StateGraph(CopilotState)
    graph.add_node("router", router)
    graph.add_node("analytical", analytical)
    graph.add_node("generative", generative)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", lambda s: s["intent"],
                                {"analytical": "analytical", "generative": "generative"})
    graph.add_edge("analytical", END)
    graph.add_edge("generative", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def turn_input(question: str) -> dict:
    """Input for one user turn: append the message, reset the per-turn fields."""
    return {"messages": [HumanMessage(question)], "trace": [], "sql": None, "rows": None,
            "context": None}
