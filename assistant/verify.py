"""Check that every figure in an answer comes from the data the agent was given.

A figure counts as verified when it matches a value in the rows within the precision it is
written at: "71.0%" matches 0.7097, "£974,087" matches 974087.28, "£1.88M" matches 1,878,371.
Percentages also match values already stored as percentages, and every number that appears
in a text cell (dates, names) or in the question itself is allowed through.

The check is deterministic, so it runs on every answer, not only in tests.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# A number not glued to a letter or code (C1-B2, TOP20, Q4 stay out), with optional £, a
# thousands separator, decimals, a %, or a K/M/bn suffix.
NUMBER = re.compile(
    r"(?<![\w.\-/])(?P<money>£)?(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?"
    r"(?:\s?(?P<pct>%)|(?P<suffix>[kKmM]\b|bn\b))?(?![\w-])")
SCALE = {"k": 1e3, "K": 1e3, "m": 1e6, "M": 1e6, "bn": 1e9}


@dataclass
class Figure:
    text: str
    value: float
    tolerance: float
    percent: bool


@dataclass
class Check:
    figures: list[Figure] = field(default_factory=list)
    unverified: list[Figure] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)  # segments mentioned but not in the data

    @property
    def ok(self) -> bool:
        return not self.unverified and not self.segments

    def summary(self) -> str:
        parts = []
        if not self.figures:
            parts.append("no figures to check")
        elif not self.unverified:
            parts.append(f"{len(self.figures)} of {len(self.figures)} figures match the data")
        else:
            missing = ", ".join(f.text for f in self.unverified)
            parts.append(f"{len(self.unverified)} of {len(self.figures)} figures not found in the data: {missing}")
        if self.segments:
            parts.append(f"mentions segments with no data in the result: {', '.join(self.segments)}")
        return "; ".join(parts)


def figures(text: str) -> list[Figure]:
    found = []
    for m in NUMBER.finditer(text):
        digits = m.group("num").replace(",", "")
        decimals = m.group("dec") or ""
        value = float(digits + decimals)
        scale = SCALE.get(m.group("suffix") or "", 1.0)
        places = len(decimals) - 1 if decimals else 0
        # Half a unit in the last written place, plus a little slack for rounding twice.
        tolerance = 0.5 * 10 ** -places * scale * 1.02
        found.append(Figure(m.group(0).strip(), value * scale, tolerance, bool(m.group("pct"))))
    return found


def _values(rows: list[dict]) -> tuple[list[float], set[float]]:
    """Numeric cells, and numbers that appear inside text cells (dates, names, codes)."""
    numbers, from_text = [], set()
    for row in rows:
        for key, v in row.items():
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float)):
                if not (isinstance(v, float) and math.isnan(v)):
                    numbers.append(float(v))
            else:
                from_text |= {f.value for f in figures(str(v))}
            # Column names like AVG_TOP20_REVENUE_CAPTURE: "top 20" is a label, not a figure.
            label = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", str(key).replace("_", " "))
            from_text |= {f.value for f in figures(label)}
    return numbers, from_text


def mentioned_segments(text: str, names: dict[str, str]) -> set[str]:
    """Segment codes named in `text`, by code (C1-B2) or full name (Top spenders / Holiday buyers)."""
    found = {c.upper() for c in re.findall(r"\bC\d-B\d\b", text, re.IGNORECASE)}
    lower = text.lower()
    return found | {code for code, name in names.items() if name.lower() in lower}


def check(answer: str, rows: list[dict], allowed_text: str = "",
          segment_names: dict[str, str] | None = None) -> Check:
    """Which figures in `answer` can be traced to `rows` (or appear in `allowed_text`).

    With `segment_names` (code -> name), also flags segments the answer talks about that have
    no row in the result: a comparison with a segment the query never returned is a guess.
    """
    numbers, from_text = _values(rows)
    from_text |= {f.value for f in figures(allowed_text)}
    from_text.add(float(len(rows)))  # "three segments", "12 rows"
    result = Check()
    for fig in figures(answer):
        result.figures.append(fig)
        candidates = numbers + [v * 100 for v in numbers] if fig.percent else numbers
        if any(abs(fig.value - c) <= fig.tolerance for c in candidates):
            continue
        if any(abs(fig.value - t) < 1e-9 for t in from_text):
            continue
        result.unverified.append(fig)

    if segment_names:
        returned = {str(r.get("SEGMENT")) for r in rows if r.get("SEGMENT")}
        returned |= {code for code, name in segment_names.items()
                     if any(r.get("SEGMENT_NAME") == name for r in rows)}
        if returned:  # only meaningful when the rows are segments
            result.segments = sorted(mentioned_segments(answer, segment_names) - returned)
    return result
