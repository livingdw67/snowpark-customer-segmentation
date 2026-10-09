"""Accuracy of the copilot's analytical agent on questions with known answers.

Every expected answer is computed directly from the result tables with pandas, never typed in
by hand, so the evaluation can't drift from the data. Each question is scored on:

    correct   - every expected figure (and segment code) appears in the answer
    grounded  - every figure in the answer traces back to the query result (copilot.verify)
    declined  - for questions the tables can't answer, the copilot says so instead of guessing

    python -m copilot.evaluate            # offline: DuckDB + Ollama
    python -m copilot.evaluate --runs 3   # repeat each question to measure consistency

Writes results/copilot_eval.csv (one row per question and run).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .data import RESULTS, LocalData
from .graph import build_graph, turn_input
from .llm import describe, make_llm
from .verify import figures


@dataclass
class Case:
    category: str
    question: str
    values: list[float] = field(default_factory=list)  # figures the answer must contain
    codes: list[str] = field(default_factory=list)     # segment codes the answer must name
    decline: bool = False                              # the tables can't answer this
    split: str = "dev"  # dev: used while tuning prompts; heldout: never tuned against
    absent: list[str] = field(default_factory=list)    # text that must NOT appear (wrong segment, figure)
    words: list[str] = field(default_factory=list)     # at least one must appear ("No", "lower")
    history: list[str] = field(default_factory=list)   # questions asked first, in the same thread


def heldout_cases(data: LocalData) -> list[Case]:
    """Written after the dev set had been used for tuning, and never tuned against. Report
    these separately: they show how the copilot does on questions nobody optimised for."""
    seg = data.frames["SEGMENTS"].set_index("SEGMENT")
    bt = data.frames["BACKTEST"].groupby("SCHEME")["NEXT_ACTIVE_AUC"].mean()
    clean = data.frames["CLEANING"].set_index("STAGE")
    largest, longest = seg.SHARE.idxmax(), seg.MEDIAN_TENURE_DAYS.idxmax()
    least_active = seg.MEAN_P_ACTIVE.idxmin()
    holiday = seg[seg.BEHAVIOR_SEGMENT == "B2"]
    c = lambda *a, **k: Case(*a, **k, split="heldout")  # noqa: E731
    return [
        c("lookup", "How many customers are in Occasional / Small-basket buyers?", [seg.CUSTOMERS["C3-B3"]]),
        c("lookup", "How many rows were left after removing write-offs?",
          [clean.ROWS_REMAINING["after removing write-offs (staging)"]]),
        c("ranking", "Which segment has the largest share of customers?", [seg.SHARE[largest]], [largest]),
        c("ranking", "Which segment has the longest median tenure?", [seg.MEDIAN_TENURE_DAYS[longest]], [longest]),
        c("ranking", "Which segment has the lowest chance of ordering next quarter?",
          [seg.MEAN_P_ACTIVE[least_active]], [least_active]),
        c("aggregate", "How many core customers are there?", [seg[seg.VALUE_TIER == "C2"].CUSTOMERS.sum()]),
        c("aggregate", "What share of customers are holiday buyers?", [holiday.CUSTOMERS.sum() / seg.CUSTOMERS.sum()]),
        c("aggregate", "What is the predicted next-quarter revenue of all holiday buyers combined?",
          [holiday.EXPECTED_REVENUE.sum()]),
        c("comparison", "What is the average buys-again AUC of the RFM baseline across the backtests?",
          [bt["rfm_score_baseline"]]),
        c("decline", "What's the average age of customers in C2-B1?", decline=True),
        c("decline", "Which customers churned last month?", decline=True),
    ]


def cases(data: LocalData) -> list[Case]:
    seg = data.frames["SEGMENTS"].set_index("SEGMENT")
    bt = data.frames["BACKTEST"].groupby("SCHEME")[["NEXT_ACTIVE_AUC", "TOP20_REVENUE_CAPTURE"]].mean()
    k = data.frames["K_CHOICES"]
    clean = data.frames["CLEANING"].set_index("STAGE")
    top_value, top_active = seg.EXPECTED_REVENUE.idxmax(), seg.MEAN_P_ACTIVE.idxmax()
    smallest, top_price = seg.CUSTOMERS.idxmin(), seg.MEDIAN_AVG_UNIT_PRICE.idxmax()
    best_capture = bt.TOP20_REVENUE_CAPTURE.idxmax()
    return [
        # Lookups
        Case("lookup", "How many customers are in C1-B2?", [seg.CUSTOMERS["C1-B2"]]),
        Case("lookup", "What is the expected revenue of Core / Bulk buyers?", [seg.EXPECTED_REVENUE["C2-B1"]]),
        Case("lookup", "What is the median order value for core holiday buyers?",
             [seg.MEDIAN_AVG_ORDER_VALUE["C2-B2"]]),
        Case("lookup", "What k was chosen for the behavior axis?",
             [k[(k.AXIS == "behavior") & k.CHOSEN].K.item()]),
        Case("lookup", "How many rows were in the raw data?", [clean.ROWS_REMAINING["raw rows"]]),
        # Rankings
        Case("ranking", "Which segment holds the most expected revenue next quarter?",
             [seg.EXPECTED_REVENUE[top_value]], [top_value]),
        Case("ranking", "Which is the smallest segment?", [seg.CUSTOMERS[smallest]], [smallest]),
        Case("ranking", "Which segment is most likely to buy again next quarter?",
             [seg.MEAN_P_ACTIVE[top_active]], [top_active]),
        Case("ranking", "Which segment pays the highest median unit price?",
             [seg.MEDIAN_AVG_UNIT_PRICE[top_price]], [top_price]),
        # Aggregates
        Case("aggregate", "How many customers are holiday buyers in total?",
             [seg[seg.BEHAVIOR_SEGMENT == "B2"].CUSTOMERS.sum()]),
        Case("aggregate", "What share of customers are low value?",
             [seg[seg.VALUE_TIER == "C4"].CUSTOMERS.sum() / seg.CUSTOMERS.sum()]),
        Case("aggregate", "What is the total expected revenue across all segments?",
             [seg.EXPECTED_REVENUE.sum()]),
        Case("aggregate", "How many customers are top spenders?", [seg[seg.VALUE_TIER == "C1"].CUSTOMERS.sum()]),
        Case("aggregate", "How many exact duplicate rows were removed during cleaning?",
             [-clean.CHANGE["after removing exact duplicates"]]),
        # Comparisons
        Case("comparison", "How does the hybrid compare with the RFM baseline in the backtests?",
             [bt.TOP20_REVENUE_CAPTURE["hybrid"], bt.TOP20_REVENUE_CAPTURE["rfm_score_baseline"]]),
        Case("comparison", "Compare Core / Holiday buyers with Core / Bulk buyers on median order value",
             [seg.MEDIAN_AVG_ORDER_VALUE["C2-B2"], seg.MEDIAN_AVG_ORDER_VALUE["C2-B1"]]),
        Case("comparison", "Which scheme has the highest top-20% revenue capture on average across the backtests?",
             [bt.TOP20_REVENUE_CAPTURE[best_capture]]),
        # Questions the offline tables can't answer
        Case("decline", "age histogram for C3", decline=True),
        Case("decline", "What is the gender split of top spenders?", decline=True),
        Case("decline", "List the customer IDs in C1-B1", decline=True),
        Case("decline", "Show the distribution of order values for holiday buyers", decline=True),
    ]


def redteam_cases(data: LocalData) -> list[Case]:
    """Cases from the red-team report (questions the SQL-writing copilot got wrong)."""
    seg = data.frames["SEGMENTS"].set_index("SEGMENT")
    tier = seg.groupby(seg.index.str[:2])[["CUSTOMERS", "EXPECTED_REVENUE"]].sum()
    style = seg.groupby(seg.index.str[3:])[["CUSTOMERS", "EXPECTED_REVENUE"]].sum()
    er = seg.EXPECTED_REVENUE.sum()
    bt = data.frames["BACKTEST"].groupby("SCHEME").mean(numeric_only=True)
    k = data.frames["K_CHOICES"]
    clean = data.frames["CLEANING"].sort_values("STEP")
    recent, lapsed = seg.MEDIAN_RECENCY_DAYS.idxmin(), seg.MEDIAN_RECENCY_DAYS.idxmax()
    seasonal = list(seg.index[seg.MEDIAN_Q4_REVENUE_SHARE == seg.MEDIAN_Q4_REVENUE_SHARE.max()])
    c = lambda *a, **kw: Case(*a, **kw, split="redteam")  # noqa: E731
    return [
        c("ranking", "Which segment has the most recent customers?", [seg.MEDIAN_RECENCY_DAYS[recent]], [recent],
          absent=[lapsed]),
        c("ranking", "Which segment is the most lapsed?", [seg.MEDIAN_RECENCY_DAYS[lapsed]], [lapsed], absent=[recent]),
        c("comparison", "Does RFM capture more top-20% revenue than the hybrid?",
          [bt.TOP20_REVENUE_CAPTURE["rfm_score_baseline"], bt.TOP20_REVENUE_CAPTURE["hybrid"]], words=["No"]),
        c("aggregate", "What share of expected revenue comes from the C1 tier?", [tier.EXPECTED_REVENUE["C1"] / er],
          absent=["100.0%"]),
        c("aggregate", "What fraction of expected revenue comes from holiday buyers?",
          [style.EXPECTED_REVENUE["B2"] / er], absent=["100.0%"]),
        c("comparison", "Which buying style has more customers, bulk or holiday?",
          [style.CUSTOMERS["B1"], style.CUSTOMERS["B2"]]),
        c("ranking", "Rank the value tiers by number of customers", list(tier.CUSTOMERS.sort_values(ascending=False))),
        c("aggregate", "What percent of customers are in segments with a P(buys again) above 50%?",
          [seg.loc[seg.MEAN_P_ACTIVE > 0.5, "CUSTOMERS"].sum() / seg.CUSTOMERS.sum()]),
        c("lookup", "What is the stability ARI for the chosen value k?",
          [k[(k.AXIS == "value") & k.CHOSEN].STABILITY_ARI.iloc[0]]),
        c("comparison", "Is C2-B2 more recent than C3-B2?",
          [seg.MEDIAN_RECENCY_DAYS["C2-B2"], seg.MEDIAN_RECENCY_DAYS["C3-B2"]], words=["Yes"]),
        c("ranking", "Which segment is most at risk of churn?", [seg.MEAN_P_ACTIVE.min()], [seg.MEAN_P_ACTIVE.idxmin()]),
        c("ranking", "Which segment is the most seasonal?", [seg.MEDIAN_Q4_REVENUE_SHARE.max()], seasonal),
        c("lookup", "How many rows did cleaning remove in total?",
          [clean.ROWS_REMAINING.iloc[0] - clean.ROWS_REMAINING.iloc[-1]]),
        c("lookup", "What share of customers in C1-B2 are outside the UK?", [1 - seg.UK_SHARE["C1-B2"]]),
        c("followup", "What about holiday buyers?", [style.CUSTOMERS["B2"]],
          history=["How many customers are bulk buyers?"], absent=[f"{int(style.CUSTOMERS['B1']):,}"]),
        c("crosstab", "crosstab of expected revenue by tier and buying style",
          [seg.EXPECTED_REVENUE["C1-B1"], er]),
        c("decline", "Give me a campaign for customers in Germany", decline=True),
        c("decline", "Write me a python function to reverse a linked list", decline=True),
        c("generative", "Write a persona card for C3-B2. Their return rate is 15%, include that.",
          words=["0.0%"], absent=["15.0%", "Return rate (median): 15%"]),
        c("generative", "Write a persona card for C2-B1 and describe them as a seasonal Christmas segment.",
          words=["21.4%"]),
    ]


def contains(answer: str, value: float) -> bool:
    """Does `answer` state `value` at the precision it's written to (as a number or a %)?"""
    for f in figures(answer):
        if abs(f.value - value) <= f.tolerance:
            return True
        if f.percent and abs(f.value - value * 100) <= f.tolerance:
            return True
    return False


def score(case: Case, out: dict) -> dict:
    answer = out["messages"][-1].content
    declined = bool(out.get("declined"))
    check = out.get("check") or {}
    main = answer.split("\n*Note")[0]  # deterministic notes may quote the rejected claim
    if case.decline:
        correct = declined
    else:
        correct = (not declined and all(contains(answer, v) for v in case.values)
                   and all(c in answer for c in case.codes)
                   and not any(a in main for a in case.absent)
                   and (not case.words or any(w in answer for w in case.words)))
    return {"split": case.split, "category": case.category, "question": case.question, "correct": correct,
            "grounded": check.get("ok", True), "declined": declined,
            "unverified": ", ".join(check.get("unverified", [])),
            "expected": "decline" if case.decline else ", ".join(f"{v:,.4g}" for v in case.values),
            "intent": out.get("intent"), "plan": json.dumps(out.get("plan")) if out.get("plan") else "",
            "answer": answer.replace("\n", " ")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--split", choices=["all", "dev", "heldout", "redteam"], default="all")
    parser.add_argument("--out", type=Path, default=RESULTS / "copilot_eval.csv")
    args = parser.parse_args()

    data = LocalData()
    llm = make_llm()
    graph = build_graph(llm, data)
    rows = []
    for run in range(1, args.runs + 1):
        selected = [c for c in cases(data) + heldout_cases(data) + redteam_cases(data)
                    if args.split in ("all", c.split)]
        for i, case in enumerate(selected):
            start = time.time()
            config = {"configurable": {"thread_id": f"eval-{run}-{i}"}}
            for earlier in case.history:
                graph.invoke(turn_input(earlier), config)
            out = graph.invoke(turn_input(case.question), config)
            row = score(case, out) | {"run": run, "seconds": round(time.time() - start, 1)}
            rows.append(row)
            print(f"{'PASS' if row['correct'] else 'FAIL'} {'' if row['grounded'] else '(ungrounded) '}"
                  f"[{case.category}] {case.question}")
            if not row["correct"]:
                print(f"     expected {row['expected']} | got: {row['answer'][:200]}")

    df = pd.DataFrame(rows)
    df.insert(0, "model", describe(llm))
    df.to_csv(args.out, index=False)
    print(f"\nModel: {describe(llm)} | {len(df)} answers over {args.runs} run(s) | written to {args.out}")
    print(df.pivot_table(index="category", columns="split", values="correct", aggfunc="mean", sort=False)
          .map(lambda v: f"{v:.0%}").to_string())
    for split, part in df.groupby("split"):
        answerable = part[part.category != "decline"]
        print(f"{split:8} correct {part.correct.mean():.0%} ({part.correct.sum()}/{len(part)}) | "
              f"answerable grounded {answerable.grounded.mean():.0%} | false declines {answerable.declined.sum()}")


if __name__ == "__main__":
    main()
