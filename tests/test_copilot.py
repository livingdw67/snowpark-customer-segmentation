"""Copilot tests with a scripted chat model: routing, SQL repair, retrieval, memory, guard rails.

They run offline on the committed results/*.csv and need neither Ollama nor Snowflake.
"""
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from copilot.data import LocalData, UnsafeQuery, validate_sql
from copilot.graph import build_graph, match_segments, turn_input


@pytest.fixture(scope="module")
def data():
    return LocalData()


def ask(graph, question, thread="t"):
    return graph.invoke(turn_input(question), {"configurable": {"thread_id": thread}})


def test_local_tables(data):
    seg = data.segments()
    assert set(data.tables) == {"SEGMENTS", "BACKTEST", "K_CHOICES", "CLEANING"}
    assert len(seg) == 12 and seg.CUSTOMERS.sum() == 4261
    assert seg.set_index("SEGMENT").loc["C1-B2", "SEGMENT_NAME"] == "Top spenders / Holiday buyers"


@pytest.mark.parametrize("sql", [
    "DROP TABLE SEGMENTS",
    "SELECT 1; DROP TABLE SEGMENTS",
    "WITH x AS (SELECT 1) DELETE FROM SEGMENTS",
    "COPY SEGMENTS TO 'out.csv'",
])
def test_guard_rejects_writes(sql):
    with pytest.raises(UnsafeQuery):
        validate_sql(sql)


def test_guard_allows_keywords_in_strings_and_blocks_files(data):
    assert validate_sql("SELECT 'drop; table' AS s;") == "SELECT 'drop; table' AS s"
    with pytest.raises(Exception):  # external access is disabled in DuckDB
        data.query("SELECT * FROM read_csv('results/k_selection.csv')")


def test_analytical_path(data):
    llm = FakeListChatModel(responses=[
        "analytical",
        "```sql\nSELECT SEGMENT, SEGMENT_NAME, EXPECTED_REVENUE FROM SEGMENTS "
        "ORDER BY EXPECTED_REVENUE DESC LIMIT 1\n```",
        "Top spenders / Bulk buyers (C1-B1) hold the most expected revenue.",
    ])
    out = ask(build_graph(llm, data), "Which segment holds the most expected revenue?")
    assert out["intent"] == "analytical"
    assert out["rows"][0]["SEGMENT"] == "C1-B1"
    assert [s["kind"] for s in out["trace"]] == ["text", "sql", "table"]
    assert "C1-B1" in out["messages"][-1].content


def test_failed_sql_is_repaired(data):
    llm = FakeListChatModel(responses=[
        "analytical",
        "SELECT * FROM NOT_A_TABLE",
        "```sql\nSELECT COUNT(*) AS N FROM SEGMENTS\n```",
        "There are 12 segments.",
    ])
    out = ask(build_graph(llm, data), "How many segments are there?")
    titles = [s["title"] for s in out["trace"]]
    assert titles == ["Intent", "SQL (attempt 1) failed", "SQL (attempt 2)", "Result"]
    assert out["rows"] == [{"N": 12}]


def test_gives_up_after_repeated_failures(data):
    llm = FakeListChatModel(responses=["analytical", "DROP TABLE SEGMENTS", "SELECT * FROM NOPE"])
    out = ask(build_graph(llm, data), "Delete everything")
    assert out["rows"] is None
    assert out["messages"][-1].content.startswith("I couldn't answer that")


def test_generative_path_retrieves_named_segment(data):
    llm = FakeListChatModel(responses=["generative", "**Top spenders / Holiday buyers (C1-B2)** ..."])
    out = ask(build_graph(llm, data), "Write a persona card for Top spenders / Holiday buyers")
    assert out["intent"] == "generative"
    assert [r["SEGMENT"] for r in out["context"]] == ["C1-B2"]
    assert out["trace"][-1]["title"] == "Retrieved segment statistics"


def test_router_falls_back_to_keywords(data):
    llm = FakeListChatModel(responses=["I'm not sure", "Here's a draft."])
    out = ask(build_graph(llm, data), "Draft a win-back email for Low value / Bulk buyers")
    assert out["intent"] == "generative"
    assert [r["SEGMENT"] for r in out["context"]] == ["C4-B1"]


def test_follow_up_uses_conversation(data):
    llm = FakeListChatModel(responses=[
        "analytical",
        "```sql\nSELECT SEGMENT, SEGMENT_NAME FROM SEGMENTS WHERE SEGMENT = 'C2-B2'\n```",
        "That's Core / Holiday buyers (C2-B2).",
        "generative",
        "Subject: Your holiday range is in",
    ])
    graph = build_graph(llm, data)
    ask(graph, "Which segment is C2-B2?", thread="memory")
    out = ask(graph, "Now draft an email for that segment", thread="memory")
    assert len(out["messages"]) == 4
    assert [r["SEGMENT"] for r in out["context"]] == ["C2-B2"]


@pytest.mark.parametrize("text,expected", [
    ("holiday buyers", ["C1-B2", "C2-B2", "C3-B2", "C4-B2"]),
    ("Top spenders", ["C1-B1", "C1-B2", "C1-B3"]),
    ("core small-basket buyers", ["C2-B3"]),
    ("c3-b1 please", ["C3-B1"]),
    ("hello", []),
])
def test_match_segments(data, text, expected):
    hit, _ = match_segments(text, data.segments())
    assert list(hit.SEGMENT) == expected


def test_app_runs_offline(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from copilot import llm as llm_factory

    monkeypatch.setenv("COPILOT_MODE", "local")
    monkeypatch.setattr(llm_factory, "make_llm", lambda session=None: FakeListChatModel(responses=[
        "analytical", "```sql\nSELECT COUNT(*) AS N FROM SEGMENTS\n```", "There are 12 segments."]))
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception
    at.chat_input[0].set_value("How many segments?").run()
    assert not at.exception
    assert "There are 12 segments." in [m.markdown[-1].value for m in at.chat_message]
