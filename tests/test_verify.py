"""The figure checker that runs on every copilot answer (copilot/verify.py)."""
import pytest

from copilot.verify import check, figures

NAMES = {"C2-B1": "Core / Bulk buyers", "C2-B2": "Core / Holiday buyers"}
BULK = {"SEGMENT": "C2-B1", "SEGMENT_NAME": "Core / Bulk buyers", "MEDIAN_AVG_ORDER_VALUE": 410.66}
HOLIDAY = {"SEGMENT": "C2-B2", "SEGMENT_NAME": "Core / Holiday buyers", "MEDIAN_AVG_ORDER_VALUE": 423.79}


def test_parses_money_percent_and_suffixes_but_not_codes():
    found = {f.text: f.value for f in figures("C1-B2 holds £974,087 (52%), Q4 TOP20 k=4, £1.88M on 2011-12-10")}
    assert found == {"£974,087": 974087, "52%": 52, "4": 4, "£1.88M": 1_880_000}


@pytest.mark.parametrize("answer", [
    "Expected revenue is £974,087.",       # rounded to the pound
    "Expected revenue is £974,087.28.",
    "Expected revenue is about £0.97M.",
])
def test_rounding_to_the_written_precision_is_accepted(answer):
    assert check(answer, [{"EXPECTED_REVENUE": 974087.284}]).ok


def test_fractions_match_percentages():
    assert check("71.0% vs 55.6%", [{"A": 0.7097}, {"A": 0.5565}]).ok
    assert not check("72% vs 55.6%", [{"A": 0.7097}, {"A": 0.5565}]).ok


def test_invented_figures_are_flagged():
    result = check("The hybrid captured 20% more, about £974,000.", [{"A": 0.7097}, {"R": 974087.28}])
    assert [f.text for f in result.unverified] == ["20%", "£974,000"]


def test_numbers_in_labels_question_and_counts_are_allowed():
    rows = [{"AVG_TOP20_REVENUE_CAPTURE": 0.7097}, {"AVG_TOP20_REVENUE_CAPTURE": 0.5565}]
    assert check("Top 20 revenue capture: 71.0% and 55.6% across 2 schemes.", rows).ok
    assert check("Over the last 12 months it was 71.0%.", rows[:1], allowed_text="last 12 months").ok


def test_segments_without_rows_are_flagged():
    claim = "Core / Bulk buyers have a median order value of £411, higher than Core / Holiday buyers."
    result = check(claim, [BULK], segment_names=NAMES)
    assert result.segments == ["C2-B2"] and not result.ok
    assert check("C2-B2 £424 vs C2-B1 £411.", [BULK, HOLIDAY], segment_names=NAMES).ok
