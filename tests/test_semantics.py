"""The copilot's calculations, checked against pandas computed independently from the CSVs.

Each case is a question the copilot used to get wrong when the model wrote its own SQL.
"""
import pandas as pd
import pytest

from copilot.data import LocalData
from copilot.semantics import (SCHEME_METRICS as S, SEGMENT_METRICS as M, Semantics, asked_direction,
                               metrics_named_in, resolve, resolve_one)


@pytest.fixture(scope="module")
def frames():
    return LocalData().frames


@pytest.fixture(scope="module")
def sem(frames):
    return Semantics(frames["SEGMENTS"], frames["BACKTEST"], frames["K_CHOICES"], frames["CLEANING"])


@pytest.fixture(scope="module")
def seg(frames):
    return frames["SEGMENTS"]


def first_fact(result):
    return result.facts[0]


@pytest.mark.parametrize("text,expected", [
    ("core holiday buyers", [("segment", ["C2-B2"])]),
    ("Core / Holiday buyers vs Core / Bulk buyers", [("segment", ["C2-B1"]), ("segment", ["C2-B2"])]),
    ("bulk or holiday", [("style", ["C1-B1", "C2-B1", "C3-B1", "C4-B1"]), ("style", ["C1-B2", "C2-B2", "C3-B2", "C4-B2"])]),
    ("top spenders", [("tier", ["C1-B1", "C1-B2", "C1-B3"])]),
    ("c2-b2 and C3-B2", [("segment", ["C2-B2"]), ("segment", ["C3-B2"])]),
    ("hello", []),
])
def test_resolve(seg, text, expected):
    assert [(g.kind, g.codes) for g in resolve(text, seg)] == expected


@pytest.mark.parametrize("question,key,expected", [
    ("Which segment has the most recent customers?", "recency_days", -1),
    ("Which segment is the most lapsed?", "recency_days", 1),
    ("Which segment is most at risk of churn?", "p_buys_again", -1),
    ("Who is least likely to buy again?", "p_buys_again", -1),
    ("Which is the most seasonal segment?", "q4_share", 1),
    ("Which is the smallest segment?", "customers", -1),
    ("Which is the biggest segment?", "customers", 1),
    ("Is C2-B2 more recent than C3-B2?", "recency_days", -1),
])
def test_direction_follows_the_wording(question, key, expected):
    assert asked_direction(question, M[key]) == expected


def test_metric_named_in_question():
    assert metrics_named_in("does RFM capture more top-20% revenue than the hybrid?", "scheme")[0] == "top20_revenue_capture"
    assert metrics_named_in("which segment is most at risk of churn?")[0] == "p_buys_again"


def test_most_recent_is_fewest_days(sem, seg):
    truth = seg.loc[seg.MEDIAN_RECENCY_DAYS.idxmin()]
    fact = first_fact(sem.rank(M["recency_days"], "segment", -1))
    assert truth.SEGMENT in fact and f"{truth.MEDIAN_RECENCY_DAYS:.0f} days" in fact and "more recent" in fact


def test_ties_are_reported(sem, seg):
    top = seg.MEDIAN_Q4_REVENUE_SHARE.max()
    tied = seg[seg.MEDIAN_Q4_REVENUE_SHARE == top].SEGMENT
    fact = first_fact(sem.rank(M["q4_share"], "segment", +1))
    assert fact.startswith(f"{len(tied)} are tied") and all(c in fact for c in tied)


def test_share_uses_the_unfiltered_total(sem, seg):
    c1 = seg[seg.SEGMENT.str.startswith("C1")].EXPECTED_REVENUE.sum() / seg.EXPECTED_REVENUE.sum()
    assert f"{c1:.1%}" in first_fact(sem.share([resolve_one("C1", seg)], M["expected_revenue"]))
    assert f"{c1:.1%}" != "100.0%"


def test_buying_styles_are_totals_not_one_segment(sem, seg):
    fact = first_fact(sem.compare(resolve_one("bulk", seg), resolve_one("holiday", seg), [M["customers"]]))
    bulk = seg[seg.SEGMENT.str.endswith("B1")].CUSTOMERS.sum()
    holiday = seg[seg.SEGMENT.str.endswith("B2")].CUSTOMERS.sum()
    assert f"{bulk:,}" in fact and f"{holiday:,}" in fact


def test_tier_ranking_includes_every_tier(sem, seg):
    result = sem.rank(M["customers"], "tier", +1, n=4)
    truth = seg.assign(T=seg.SEGMENT.str[:2]).groupby("T").CUSTOMERS.sum().sort_values(ascending=False)
    assert [g.split("(")[1][:2] for g in result.table.Group] == list(truth.index)


def test_medians_are_not_combined(sem, seg):
    result = sem.value([resolve_one("core", seg)], [M["net_revenue_12m"]])
    assert "can't be combined" in result.notes[0] and len(result.table) == 3


def test_yes_no_comparisons(sem, seg):
    recency = sem.compare(resolve_one("C2-B2", seg), resolve_one("C3-B2", seg), [M["recency_days"]], asked=-1)
    assert first_fact(recency).startswith("Yes.")  # 15 days vs 27: C2-B2 is more recent
    rfm = sem.compare_schemes("rfm_score_baseline", "hybrid", [S["top20_revenue_capture"]], asked=+1)
    assert first_fact(rfm).startswith("No.")       # 55.6% vs 71.0%


def test_count_where(sem, seg):
    hit = seg[seg.MEAN_P_ACTIVE > 0.5]
    fact = first_fact(sem.count_where(M["p_buys_again"], ">", 0.5))
    assert f"{hit.CUSTOMERS.sum() / seg.CUSTOMERS.sum():.1%}" in fact and f"{len(hit)} of 12" in fact


def test_k_choice_explains_the_tie_rule(sem, frames):
    k = frames["K_CHOICES"]
    chosen = k[(k.AXIS == "value") & k.CHOSEN].iloc[0]
    fact = first_fact(sem.k_choice("value"))
    assert f"k={int(chosen.K)}" in fact and f"{chosen.STABILITY_ARI:.4f}" in fact and "prefers fewer" in fact


def test_cleaning_totals(sem, frames):
    c = frames["CLEANING"].sort_values("STEP")
    removed = int(c.ROWS_REMAINING.iloc[0] - c.ROWS_REMAINING.iloc[-1])
    assert f"{removed:,}" in first_fact(sem.cleaning())
    assert "11,812" in first_fact(sem.cleaning("duplicates"))


@pytest.mark.parametrize("question,key,expected", [
    ("Which segment's customers are most likely to lapse?", "p_buys_again", -1),
    ("Is two_axis_value's min segment share larger than the hybrid's?", "min_segment_share", 1),
    ("Which segment has the cheapest orders?", "order_value", -1),
])
def test_direction_words_inside_metric_names_and_lapse(question, key, expected):
    m = S[key] if key in S else M[key]
    assert asked_direction(question, m) == expected


@pytest.mark.parametrize("question,first", [
    ("Which Core segment has the most orders per customer?", "orders"),
    ("Which segment has the cheapest orders?", "order_value"),
    ("who pays the least per item?", "unit_price"),
    ("How many times bigger is C4-B3 than C1-B2?", "customers"),
    ("Which segment's customers are most likely to lapse?", "p_buys_again"),
])
def test_odd_wording_names_the_right_metric(question, first):
    assert metrics_named_in(question)[0] == first


def test_plan_group_spellings(seg):
    assert resolve_one("C1-all", seg).codes == ["C1-B1", "C1-B2", "C1-B3"]
    assert resolve_one("all-B2", seg).kind == "style"
    assert [g.codes for g in resolve("Which Top-spender segment is smallest?", seg)] == [["C1-B1", "C1-B2", "C1-B3"]]


def test_rank_within_a_style_is_by_segment(sem, seg):
    # "Among bulk buyers, which tier ...": each tier is one segment, so rows are segments, not tiers.
    result = sem.rank(M["p_buys_again"], "tier", +1, within=[resolve_one("bulk", seg)])
    best = seg[seg.SEGMENT.str.endswith("B1")].set_index("SEGMENT").MEAN_P_ACTIVE.idxmax()
    assert best in first_fact(result) and "all buying styles" not in result.markdown()


def test_rank_tiers_on_a_median_shows_segment_ranges(sem, seg):
    result = sem.rank(M["q4_share"], "style", +1)
    b2 = seg[seg.SEGMENT.str.endswith("B2")].MEDIAN_Q4_REVENUE_SHARE
    assert first_fact(result).startswith("Holiday buyers (B2) has the highest") and f"{b2.min():.1%}" in first_fact(result)
    assert len(result.table) == 12 and "can't be combined" in result.notes[0]


def test_nth_ranked(sem, seg):
    core = seg[seg.SEGMENT.str.startswith("C2")].sort_values("CUSTOMERS")
    fact = first_fact(sem.rank(M["customers"], "segment", -1, within=[resolve_one("core", seg)], nth=2))
    assert core.SEGMENT.iloc[1] in fact and "second lowest" in fact


def test_count_where_totals_a_second_measure(sem, seg):
    hit = seg[seg.MEAN_P_ACTIVE > 0.5]
    result = sem.count_where(M["p_buys_again"], ">", 0.5, report=M["expected_revenue"])
    assert f"£{hit.EXPECTED_REVENUE.sum():,.0f}" in " ".join(result.facts)
    assert f"{hit.EXPECTED_REVENUE.sum() / seg.EXPECTED_REVENUE.sum():.1%}" in " ".join(result.facts)


def test_crosstab_as_shares(sem, seg):
    table = sem.crosstab(M["expected_revenue"], as_share=True).table
    c1b1 = seg.set_index("SEGMENT").EXPECTED_REVENUE["C1-B1"] / seg.EXPECTED_REVENUE.sum()
    assert table.iloc[0]["Bulk buyers (B1)"] == f"{c1b1:.1%}" and table.iloc[-1]["All styles"] == "100.0%"


def test_cleaning_step_by_words_and_survival(sem, frames):
    assert "22.7% of the 1,026,999 rows before it" in first_fact(sem.cleaning("customer-id"))
    c = frames["CLEANING"].sort_values("STEP")
    assert f"{c.ROWS_REMAINING.iloc[-1] / c.ROWS_REMAINING.iloc[0]:.1%} of the raw rows survived" in first_fact(sem.cleaning())


def test_scheme_average(sem, frames):
    bt = frames["BACKTEST"]
    q3 = bt[bt.RUN_ID == "backtest_q3"].groupby("SCHEME").NEXT_ACTIVE_AUC.mean().mean()
    assert f"{q3:.3f}" in first_fact(sem.schemes([S["next_active_auc"]], fold="q3", average=True))


def test_crosstab(sem, seg):
    revenue = sem.crosstab(M["expected_revenue"])
    assert revenue.table.iloc[-1]["All styles"] == f"£{seg.EXPECTED_REVENUE.sum():,.0f}"
    assert len(revenue.table) == 5  # four tiers and a total
    medians = sem.crosstab(M["net_revenue_12m"])
    assert "All styles" not in medians.table.columns and "Medians don't add up" in medians.notes[0]
