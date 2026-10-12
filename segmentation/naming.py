"""Persona names for the hybrid segments, shared by the published view and the assistant.

Codes are ordered by meaning (C1 = highest predicted value, B1 = largest baskets), and
07_run_production.check_behavior_names() confirms each behavior group still matches its
name before the view is built.
"""
TIER_NAMES = {"C1": "Top spenders", "C2": "Core", "C3": "Occasional", "C4": "Low value"}
BEHAVIOR_NAMES = {"B1": "Bulk buyers", "B2": "Holiday buyers", "B3": "Small-basket buyers"}


def segment_name(code: str) -> str:
    """'C1-B2' -> 'Top spenders / Holiday buyers'."""
    tier, behavior = code.split("-")
    return f"{TIER_NAMES.get(tier, tier)} / {BEHAVIOR_NAMES.get(behavior, behavior)}"


def case_sql(column: str, names: dict[str, str]) -> str:
    """SQL CASE expression mapping codes in `column` to names."""
    return f"CASE {column} " + " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in names.items()) + " END"
