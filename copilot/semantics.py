"""The copilot's semantic layer: what each metric means and how it may be calculated.

The language model never writes SQL or arithmetic. It picks one of the calculations below and
names its inputs; this module resolves names to segments, applies each metric's definition
(how it aggregates, which direction is "better"), computes the numbers and writes the answer
sentences. A wrong answer can only come from a wrong choice of calculation or input, which
the graph also cross-checks against the question's wording.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from segmentation.naming import BEHAVIOR_NAMES, TIER_NAMES, segment_name


# --------------------------------------------------------------------------------------
# Metric catalogue
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Metric:
    key: str
    column: str
    label: str        # plain English, used in answers
    unit: str         # count | money | percent | days | number | score
    agg: str          # sum | weighted (customer-weighted mean) | median (can't be combined)
    high: str         # what a high value means
    low: str          # what a low value means
    high_words: tuple[str, ...] = ()  # question words that ask about the high end
    low_words: tuple[str, ...] = ()   # ... and the low end
    synonyms: tuple[str, ...] = ()    # words that name this metric in a question


SEGMENT_METRICS = {m.key: m for m in [
    # Size words (biggest, smallest) are superlatives, handled by asked_direction, not metric words.
    Metric("customers", "CUSTOMERS", "customers", "count", "sum", "more customers", "fewer customers",
           (), (),
           ("customers", "size", "biggest", "largest", "smallest", "how many", "headcount", "segment size",
            "times bigger", "times larger", "times smaller", "times as big", "times as large", "times the size",
            "bigger than", "smaller than")),
    Metric("expected_revenue", "EXPECTED_REVENUE", "predicted revenue next quarter", "money", "sum",
           "more predicted revenue", "less predicted revenue", ("valuable",), (),
           ("expected revenue", "predicted revenue", "forecast revenue", "valuable", "worth", "value next quarter",
            "next-quarter revenue", "next quarter revenue", "revenue next quarter", "predicted next-quarter revenue")),
    Metric("p_buys_again", "MEAN_P_ACTIVE", "chance of ordering next quarter", "percent", "weighted",
           "more likely to order again", "more at risk of churning",
           ("likely to buy", "likely to order", "loyal", "retention", "retain", "active", "engaged"),
           ("churn", "at risk", "risk", "unlikely", "lapse", "to lapse", "stop buying", "stop ordering", "drop off"),
           ("buy again", "buys again", "order again", "p(buys again)", "p(active)", "p_active", "probability",
            "likely", "churn", "retention", "at risk", "loyal", "engaged", "engagement", "active", "returning",
            "return next quarter", "likely to return", "come back", "coming back", "repeat", "chance of buying",
            "buying again", "buy again", "purchase again", "chance of ordering", "chance of buying again",
            "likely to lapse", "lapse")),
    Metric("recency_days", "MEDIAN_RECENCY_DAYS", "days since last order", "days", "median",
           "more lapsed", "more recent",
           ("lapsed", "dormant", "inactive", "longest since", "least recent"), ("recent", "recently", "fresh"),
           ("recent", "recency", "lapsed", "last order", "last purchase", "dormant", "since")),
    Metric("orders", "MEDIAN_ORDERS", "orders in the last 12 months", "number", "median",
           "order more often", "order less often", ("frequent", "often"), ("infrequent", "rarely"),
           ("orders", "frequency", "frequent", "often", "how often")),
    Metric("net_revenue_12m", "MEDIAN_NET_REVENUE", "net revenue in the last 12 months", "money", "median",
           "spent more in the last 12 months", "spent less in the last 12 months", ("spend",), (),
           ("net revenue", "spend", "spent", "revenue last year", "last 12 months", "annual revenue", "sales")),
    Metric("order_value", "MEDIAN_AVG_ORDER_VALUE", "order value", "money", "median",
           "larger orders by value", "smaller orders by value",
           ("most expensive orders", "priciest orders", "pricier orders", "larger orders", "largest orders"),
           ("cheapest orders", "cheaper orders", "cheap orders", "smaller orders", "smallest orders"),
           ("order value", "aov", "basket value", "spend per order", "per order", "spends per order",
            "cheapest orders", "cheaper orders", "cheap orders", "most expensive orders", "priciest orders",
            "pricier orders", "larger orders", "largest orders", "smaller orders", "smallest orders")),
    Metric("units_per_order", "MEDIAN_AVG_UNITS_PER_ORDER", "units per order", "number", "median",
           "bigger baskets", "smaller baskets", (), (),
           ("units", "basket size", "units per order", "baskets", "items per order")),
    Metric("unit_price", "MEDIAN_AVG_UNIT_PRICE", "unit price", "money", "median",
           "pay more per unit", "pay less per unit", ("premium", "expensive", "pricier"), ("cheap", "cheapest", "cheaper"),
           ("unit price", "price", "per unit", "premium", "expensive", "cheap", "cheapest", "per item",
            "each item", "price per item", "item price")),
    Metric("lines_per_order", "MEDIAN_AVG_LINES_PER_ORDER", "product lines per order", "number", "median",
           "wider baskets", "narrower baskets", (), (), ("lines", "lines per order", "variety", "narrow", "narrowest",
                                                         "wide", "widest", "basket width", "product lines",
                                                         "products per order", "number of lines")),
    Metric("return_rate", "MEDIAN_RETURN_RATE", "return rate", "percent", "median",
           "return more", "return less", (), (), ("returns", "return rate", "refund", "refunds", "returned items", "send back")),
    Metric("q4_share", "MEDIAN_Q4_REVENUE_SHARE", "share of revenue bought in October-December",
           "percent", "median", "more seasonal", "more year-round",
           ("seasonal", "holiday-heavy", "christmas"), ("year-round", "steady"),
           ("seasonal", "q4", "holiday season", "christmas", "october", "year-round")),
    Metric("tenure_days", "MEDIAN_TENURE_DAYS", "days since first order (tenure)", "days", "median",
           "longer-standing customers", "newer customers", ("longest-standing",), (), ("tenure", "first order", "how long", "longest-standing", "newest", "been with us", "customer since")),
    Metric("uk_share", "UK_SHARE", "share of customers in the UK", "percent", "weighted",
           "more UK-based", "more international", (), (),
           ("uk", "in the uk", "uk share", "uk-based", "united kingdom", "domestic")),
    Metric("revenue_per_customer", "REV_PER_CUSTOMER", "predicted revenue per customer next quarter",
           "money", "ratio", "more predicted revenue per customer", "less predicted revenue per customer",
           (), (), ("revenue per customer", "value per customer", "each customer worth", "revenue per head",
                    "worth per customer", "revenue per capita")),
    Metric("non_uk_customers", "NON_UK_CUSTOMERS", "customers outside the UK (estimated from the UK share)",
           "count", "sum", "more customers outside the UK", "fewer customers outside the UK", (), (),
           ("non-uk customers", "customers outside the uk", "customers are non-uk", "are non-uk",
            "based outside the uk", "customers based outside",
            "how many international", "customers abroad")),
    Metric("uk_customers", "UK_CUSTOMERS", "customers in the UK (estimated from the UK share)", "count", "sum",
           "more customers in the UK", "fewer customers in the UK", (), (),
           ("uk customers", "customers in the uk", "customers are in the uk", "based in the uk")),
    Metric("expected_orderers", "EXPECTED_ORDERERS",
           "customers expected to order next quarter (customers × chance of ordering)", "count", "sum",
           "more expected orderers", "fewer expected orderers", (), (),
           ("expect to place an order", "expected to order", "expected to place an order", "expected buyers",
            "will order next quarter", "expected orderers", "expected to buy", "would order", "would actually order",
            "will order", "would buy", "would place an order", "expected buyers")),
    Metric("non_uk_share", "NON_UK_SHARE", "share of customers outside the UK", "percent", "weighted",
           "more international", "more UK-based", (), (), ("outside the uk", "international", "abroad", "non-uk")),
]}

SCHEME_METRICS = {m.key: m for m in [
    Metric("next_active_auc", "NEXT_ACTIVE_AUC", "buys-again AUC (how well segments rank who orders next quarter)",
           "score", "mean", "ranks who buys again better", "ranks who buys again worse", (), (),
           ("auc", "buys again", "predict who", "ranking who", "next active")),
    Metric("top20_revenue_capture", "TOP20_REVENUE_CAPTURE",
           "next-quarter revenue held by the top 20% of customers it ranks highest", "percent", "mean",
           "concentrates more revenue in its top 20%", "concentrates less revenue in its top 20%", (), (),
           ("top 20", "top-20", "top20", "revenue capture", "capture", "concentration")),
    Metric("revenue_spearman", "REVENUE_SPEARMAN", "rank correlation with next-quarter revenue", "score", "mean",
           "tracks revenue better", "tracks revenue worse", (), (), ("spearman", "correlation")),
    Metric("behavior_eta2", "BEHAVIOR_ETA2", "buying-style separation (eta squared, 0-1)", "score", "mean",
           "separates buying styles better", "separates buying styles worse", (), (),
           ("behavior", "behaviour", "buying style", "eta")),
    Metric("value_eta2", "VALUE_ETA2", "value separation (eta squared, 0-1)", "score", "mean",
           "separates customer value better", "separates customer value worse", (), (), ("value eta",)),
    Metric("persistence", "PERSISTENCE", "share of customers keeping their segment next quarter", "percent",
           "mean", "more stable segments", "less stable segments", ("stable",), (),
           ("persistence", "stable", "stability", "keep their segment")),
    Metric("rule_fidelity", "RULE_FIDELITY", "how often the plain-English rules reproduce the segments",
           "percent", "mean", "more faithful rules", "less faithful rules", (), (), ("rule", "fidelity")),
    Metric("n_segments", "N_SEGMENTS", "number of segments", "number", "mean", "more segments",
           "fewer segments", (), (), ("number of segments", "how many segments")),
    Metric("min_segment_share", "MIN_SEGMENT_SHARE", "smallest segment's share of customers", "percent", "mean",
           "larger smallest segment", "smaller smallest segment", (), (),
           ("minimum segment share", "min segment share", "smallest segment share", "smallest segment")),
    Metric("predicted_active_rate", "PREDICTED_ACTIVE_RATE", "predicted share of customers ordering next quarter",
           "percent", "mean", "predicts more customers ordering", "predicts fewer customers ordering", (), (),
           ("predicted active rate", "predicted active")),
    Metric("actual_active_rate", "ACTUAL_ACTIVE_RATE", "actual share of customers ordering next quarter",
           "percent", "mean", "more customers actually ordered", "fewer customers actually ordered", (), (),
           ("actual active rate", "actual active")),
    Metric("revenue_forecast_bias", "REVENUE_FORECAST_BIAS",
           "revenue forecast bias (negative = under-forecast)", "percent", "mean",
           "over-forecasts more", "under-forecasts more", (), (), ("bias", "under-forecast", "over-forecast")),
]}

SCHEME_NAMES = {
    "rfm_score_baseline": "Classic RFM scores", "two_axis_value": "B: value axis only",
    "two_axis_grid": "B: value × behavior grid", "outcome_guided": "C: outcome tiers (3-4 segments)",
    "outcome_guided_grid_size": "C: outcome tiers (9-12 segments)", "hybrid": "Hybrid (C tiers × B behavior)",
    "outcome_model_continuous": "Unsegmented outcome model (the ceiling)",
}
SCHEME_ALIASES = {
    "rfm": "rfm_score_baseline", "baseline": "rfm_score_baseline", "hybrid": "hybrid",
    "value axis": "two_axis_value", "two-axis value": "two_axis_value", "grid": "two_axis_grid",
    "two-axis": "two_axis_grid", "two axis": "two_axis_grid", "option b": "two_axis_grid",
    "outcome tiers (9": "outcome_guided_grid_size", "outcome_guided_grid_size": "outcome_guided_grid_size",
    "outcome tiers": "outcome_guided", "outcome-guided": "outcome_guided", "outcome guided": "outcome_guided",
    "option c": "outcome_guided", "continuous": "outcome_model_continuous", "ceiling": "outcome_model_continuous",
    "unsegmented": "outcome_model_continuous",
}


def metric(key: str, level: str = "segment") -> Metric:
    table = SCHEME_METRICS if level == "scheme" else SEGMENT_METRICS
    if key in table:
        return table[key]
    by_column = {m.column.lower(): m for m in table.values()}
    if key.lower() in by_column:
        return by_column[key.lower()]
    raise ValueError(f"unknown {level} metric '{key}'. Known: {', '.join(table)}")


GROUP_WORDS = {"bulk", "holiday", "small-basket", "small basket", "core", "occasional", "top spenders",
               "low value"}


def metrics_named_in(text: str, level: str = "segment") -> list[str]:
    """Metrics whose synonyms appear in `text` as whole words, longest match first.

    Whole words only ("engaged" doesn't contain "age"), and tier or buying-style words never
    count as naming a metric ("bulk buyers" is a group, not units per order).
    """
    lower = text.lower().replace("’", "'")
    table = SCHEME_METRICS if level == "scheme" else SEGMENT_METRICS
    hits = []
    for m in table.values():
        best = max((len(s) for s in m.synonyms
                    if s not in GROUP_WORDS and re.search(rf"(?<![\w]){re.escape(s)}(?![\w])", lower)), default=0)
        if best:
            hits.append((best, m.key))
    return [k for _, k in sorted(hits, reverse=True)]


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------

def fmt(m: Metric, v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    if m.unit == "count":
        return f"{v:,.0f}"
    if m.unit == "money":
        return f"£{v:,.0f}" if abs(v) >= 100 else f"£{v:,.2f}"
    if m.unit == "percent":
        return f"{v:.1%}"
    if m.unit == "days":
        return f"{v:,.0f} days" if float(v).is_integer() else f"{v:,.1f} days"
    if m.unit == "score":
        return f"{v:.3f}"
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.1f}"


def cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def difference(m: Metric, va: float, vb: float) -> str:
    """'Difference: 848 customers (19.8 times as many).' Percent measures differ in points."""
    d = abs(va - vb)
    if m.unit == "percent":
        return f"Difference: {d * 100:.1f} percentage points."
    if m.unit == "score":
        return f"Difference: {d:.3f}."
    text = f"Difference: {fmt(m, d)}"
    lo, hi = sorted([va, vb])
    if lo > 0 and m.unit in ("count", "money", "number", "days"):
        text += f" ({hi / lo:.1f} times as much)"
    return text + "."


def convert_days(text: str, unit: str) -> str:
    """'393 days' -> '393 days (about 56.1 weeks)' for each day count in the text."""
    per = {"week": 7, "weeks": 7, "month": 30.44, "months": 30.44, "year": 365.25, "years": 365.25}[unit]
    name = unit if unit.endswith("s") else unit + "s"
    if "(about" in text:
        return text
    return re.sub(r"(\d[\d,]*(?:\.\d+)?) days",
                  lambda mm: f"{mm.group(0)} (about {float(mm.group(1).replace(',', '')) / per:.1f} {name})", text)


def convert_money(text: str, unit: str) -> str:
    """'£1,290,485' -> '£1,290,485 (about £1,290k)' for each £ amount in the text; unit is
    pence, thousands or millions."""
    if "(about" in text or " pence)" in text:
        return text

    def one(mm):
        v = float(mm.group(1).replace(",", ""))
        if unit == "pence":
            return f"{mm.group(0)} ({v * 100:,.0f} pence)"
        if unit == "thousands":
            return f"{mm.group(0)} (about £{v / 1e3:,.1f}k)" if v < 1e5 else f"{mm.group(0)} (about £{v / 1e3:,.0f}k)"
        return f"{mm.group(0)} (about £{v / 1e6:,.2f} million)"
    return re.sub(r"£(\d[\d,]*(?:\.\d+)?)", one, text)


def combine(groups: list[Group]) -> Group:
    """Several groups as one ("C1-B2 and C1-B3 combined")."""
    codes = list(dict.fromkeys(c for g in groups for c in g.codes))
    label = " + ".join(g.label.split(" (")[0] + (f" ({g.codes[0]})" if g.single else "") for g in groups)
    return Group("combined", codes, f"{label} combined")


def pct(v: float) -> str:
    return f"{v:.1%}"


# --------------------------------------------------------------------------------------
# Groups: resolve what a question or plan refers to
# --------------------------------------------------------------------------------------

TIER_WORDS = {"C1": ("top spenders", "top spender", "top tier", "top-tier", "top-spender", "top-spenders"), "C2": ("core",),
              "C3": ("occasional",), "C4": ("low value", "low-value")}
STYLE_WORDS = {"B1": ("bulk",), "B2": ("holiday",), "B3": ("small-basket", "small basket")}


@dataclass
class Group:
    kind: str                 # segment | tier | style | all
    codes: list[str]          # segment codes in the group
    label: str

    @property
    def single(self) -> bool:
        return self.kind == "segment" and len(self.codes) == 1


def _has(text: str, word: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", text) is not None


def group_words(text: str) -> list[tuple[int, str, str]]:
    """Tier and style words as (position, 'tier'|'style', code), in order of appearance, so
    "bulk or holiday" stays two groups. Bare codes ("the C1 tier", "B2", "C1-*") count too."""
    used = text.lower().replace("’", "'")
    words = []
    for kind, table in (("tier", TIER_WORDS), ("style", STYLE_WORDS)):
        for code, ws in table.items():
            for w in ws:
                for m in re.finditer(rf"(?<![\w-]){re.escape(w)}(?![\w-])", used):
                    words.append((m.start(), kind, code))
    for m in re.finditer(r"(?<![\w-])c([1-4])(?:-\*)?(?![\w-])", used):
        words.append((m.start(), "tier", f"C{m.group(1)}"))
    for m in re.finditer(r"(?<![\w-])b([1-3])(?:-\*)?(?![\w-])", used):
        words.append((m.start(), "style", f"B{m.group(1)}"))
    return sorted(set(words))


def resolve(text: str, seg: pd.DataFrame) -> list[Group]:
    """Every group named in `text`: segment codes or full names, or tier/style words.

    "Core / Holiday buyers", "core holiday buyers" and "c2-b2" are the segment C2-B2;
    "holiday buyers" alone is the whole buying style B2; "top spenders" the tier C1.
    """
    lower = text.lower().replace("’", "'")
    groups, used = [], lower
    for code in re.findall(r"\bc\d-b\d\b", lower):
        code = code.upper()
        if code in set(seg.SEGMENT):
            groups.append(Group("segment", [code], f"{segment_name(code)} ({code})"))
    for code, name in zip(seg.SEGMENT, seg.SEGMENT_NAME):
        if name.lower() in used and code not in {c for g in groups for c in g.codes}:
            groups.append(Group("segment", [code], f"{name} ({code})"))
            used = used.replace(name.lower(), " ")
    words = group_words(used)
    i = 0
    while i < len(words):
        pos, kind, code = words[i]
        nxt = words[i + 1] if i + 1 < len(words) else None
        # "core holiday buyers": a tier word directly followed by a style word is one segment.
        if kind == "tier" and nxt and nxt[1] == "style" and nxt[0] - pos < 25 and \
                not re.search(r"\b(or|and|vs|versus|than|with|compared)\b", used[pos:nxt[0]]):
            seg_code = f"{code}-{nxt[2]}"
            groups.append(Group("segment", [seg_code], f"{segment_name(seg_code)} ({seg_code})"))
            i += 2
            continue
        if kind == "style" and nxt and nxt[1] == "tier" and nxt[0] - pos < 30 and \
                re.search(r"\b(but|who are|that are|and are|in the|among)\b", used[pos:nxt[0]]):
            seg_code = f"{nxt[2]}-{code}"
            groups.append(Group("segment", [seg_code], f"{segment_name(seg_code)} ({seg_code})"))
            i += 2
            continue
        if kind == "tier":
            codes = [c for c in seg.SEGMENT if c.startswith(code)]
            groups.append(Group("tier", codes, f"{TIER_NAMES[code]} ({code}, all buying styles)"))
        else:
            codes = [c for c in seg.SEGMENT if c.endswith(code)]
            groups.append(Group("style", codes, f"{BEHAVIOR_NAMES[code]} ({code}, all value tiers)"))
        i += 1
    unique, seen = [], set()
    for g in groups:
        key = (g.kind, tuple(g.codes))
        if key not in seen:
            seen.add(key)
            unique.append(g)
    return unique


def resolve_one(text: str, seg: pd.DataFrame) -> Group | None:
    # "C1-all", "C1-*", "*-B2", "all-B2": a whole tier or buying style.
    text = re.sub(r"^\s*(c[1-4])-(all|any|\*|x)\s*$", r"\1", str(text), flags=re.IGNORECASE)
    text = re.sub(r"^\s*(all|any|\*|x)-(b[1-3])\s*$", r"\2", text, flags=re.IGNORECASE)
    if str(text).strip().lower() in ("all", "everyone", "all customers", "total", "overall", "all segments"):
        return Group("all", list(seg.SEGMENT), "all customers")
    if str(text).strip().upper() in TIER_NAMES:
        code = str(text).strip().upper()
        return Group("tier", [c for c in seg.SEGMENT if c.startswith(code)], f"{TIER_NAMES[code]} ({code}, all buying styles)")
    if str(text).strip().upper() in BEHAVIOR_NAMES:
        code = str(text).strip().upper()
        return Group("style", [c for c in seg.SEGMENT if c.endswith(code)], f"{BEHAVIOR_NAMES[code]} ({code}, all value tiers)")
    found = resolve(str(text), seg)
    return found[0] if found else None


def resolve_scheme(text: str) -> str | None:
    t = str(text).strip().lower()
    if t in SCHEME_NAMES:
        return t
    for alias, key in sorted(SCHEME_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if alias in t:
            return key
    return None


# --------------------------------------------------------------------------------------
# Direction from wording
# --------------------------------------------------------------------------------------

UP = r"\b(most|highest|largest|biggest|greatest|max(imum)?|top|more|higher|larger|greater|best|longest|oldest|longer|older)\b"
DOWN = r"\b(least|lowest|fewest|smallest|min(imum)?|bottom|less|lower|fewer|worst|shortest|newest|shorter|newer)\b"


def strip_group_words(text: str) -> str:
    """Remove names whose words look like superlatives: "Top spenders" is a tier and "top 20%"
    is a metric, neither asks for the top of a ranking."""
    return re.sub(r"\btop[- ]?(spenders?|tier|20%?|20 ?per ?cent)\b", " ", text, flags=re.IGNORECASE)


def metric_match_lengths(text: str, level: str = "segment") -> dict[str, int]:
    """For each metric named in `text`, the length of its longest matching synonym."""
    lower = text.lower().replace("’", "'")
    table = SCHEME_METRICS if level == "scheme" else SEGMENT_METRICS
    out = {}
    for m in table.values():
        best = max((len(s) for s in m.synonyms
                    if s not in GROUP_WORDS and re.search(rf"(?<![\w]){re.escape(s)}(?![\w])", lower)), default=0)
        if best:
            out[m.key] = best
    return out


def asked_direction(question: str, m: Metric) -> int:
    """+1 if the wording asks about the metric's high end, -1 for the low end, 0 if unclear.

    "most recent" on days-since-last-order is the LOW end (fewest days); "most at risk" on
    chance-of-ordering is the low end; "most seasonal" on Oct-Dec share is the high end.
    """
    q = strip_group_words(question.lower())
    # A metric whose name holds a direction word ("min segment share") isn't asking for a direction.
    named = q
    for s in sorted(m.synonyms, key=len, reverse=True):
        if " " in s and s not in m.low_words + m.high_words:
            named = re.sub(rf"(?<![\w]){re.escape(s)}(?![\w])", " ", named)
    up, down = re.search(UP, named), re.search(DOWN, named)
    sup = 0
    if up and down:  # "from cheapest to most expensive": the first direction named is the order asked for
        sup = 1 if up.start() < down.start() else -1
    elif up:
        sup = 1
    elif down:
        sup = -1
    word = 0
    lows = [mm.start() for w in m.low_words for mm in re.finditer(rf"(?<![\w-]){re.escape(w)}(?![\w-])", q)]
    highs = [mm.start() for w in m.high_words for mm in re.finditer(rf"(?<![\w-]){re.escape(w)}(?![\w-])", q)]
    if lows and highs:  # "from cheapest to most expensive": the first one named sets the order
        return -1 if min(lows) < min(highs) else 1
    if lows:
        word = -1
    if highs:
        word = 1
    if word == 0:
        word = 1  # the metric named directly: "highest return rate" is the high end
    if sup == 0:
        return word if any(_has(q, w) for w in m.low_words + m.high_words) else 0
    return sup * word


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------

@dataclass
class Result:
    facts: list[str]
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)

    def markdown(self) -> str:
        text = " ".join(self.facts)
        if len(self.table) > 1 and len(self.table.columns) > 1:
            text += "\n\n" + markdown_table(self.table)
        if self.notes:
            text += "\n\n" + "\n".join(f"*{n}*" for n in self.notes)
        return text


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for rec in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in rec) + " |")
    return "\n".join(lines)


class Semantics:
    """All calculations over the result tables. Every public method returns a Result."""

    def __init__(self, segments: pd.DataFrame, backtest: pd.DataFrame, k_choices: pd.DataFrame,
                 cleaning: pd.DataFrame):
        self.seg = segments.copy().set_index("SEGMENT", drop=False)
        self.seg["NON_UK_SHARE"] = 1 - self.seg.UK_SHARE
        self.seg["REV_PER_CUSTOMER"] = self.seg.EXPECTED_REVENUE / self.seg.CUSTOMERS
        self.seg["NON_UK_CUSTOMERS"] = self.seg.CUSTOMERS * (1 - self.seg.UK_SHARE)  # estimates: sums stay unrounded
        self.seg["UK_CUSTOMERS"] = self.seg.CUSTOMERS * self.seg.UK_SHARE
        self.seg["EXPECTED_ORDERERS"] = self.seg.CUSTOMERS * self.seg.MEAN_P_ACTIVE
        self.bt = backtest
        self.k = k_choices
        self.clean = cleaning.sort_values("STEP")

    # ---- helpers --------------------------------------------------------------------

    def group_value(self, g: Group, m: Metric) -> tuple[float | None, str | None]:
        """A metric for a group, honouring how it aggregates. Returns (value, note)."""
        part = self.seg.loc[g.codes]
        if len(part) == 1:
            return float(part[m.column].iloc[0]), None
        if m.agg == "sum":
            return float(part[m.column].sum()), None
        if m.agg == "ratio":
            return float(part.EXPECTED_REVENUE.sum() / part.CUSTOMERS.sum()), None
        if m.agg == "weighted":
            v = float((part[m.column] * part.CUSTOMERS).sum() / part.CUSTOMERS.sum())
            return v, f"{cap(m.label)} for a group is the customer-weighted average of its segments."
        return None, (f"{cap(m.label)} is a median per segment, and medians can't be combined "
                      "across segments, so the segment values are shown instead.")

    def _segment_rows(self, codes: list[str], m: Metric) -> pd.DataFrame:
        part = self.seg.loc[codes]
        return pd.DataFrame({"Segment": [f"{n} ({c})" for c, n in zip(part.SEGMENT, part.SEGMENT_NAME)],
                             cap(m.label): [fmt(m, v) for v in part[m.column]]})

    # ---- calculations ---------------------------------------------------------------

    def value(self, groups: list[Group], metrics: list[Metric]) -> Result:
        facts, notes, rows = [], [], []
        for g in groups:
            parts = []
            for m in metrics:
                v, note = self.group_value(g, m)
                if v is None:
                    notes.append(note)
                    tbl = self._segment_rows(g.codes, m)
                    rows.extend(tbl.to_dict("records"))
                    parts.append(f"{m.label} is only available per segment (below)")
                else:
                    if note:
                        notes.append(note)
                    parts.append(f"{m.label}: {fmt(m, v)}")
                    rows.append({"Group": g.label, "Metric": m.label, "Value": fmt(m, v)})
            facts.append(f"{g.label}: " + "; ".join(parts) + ".")
        return Result(facts, pd.DataFrame(rows), list(dict.fromkeys(notes)))

    def profile(self, g: Group) -> Result:
        keys = ["customers", "expected_revenue", "p_buys_again", "recency_days", "orders", "net_revenue_12m",
                "order_value", "units_per_order", "unit_price", "lines_per_order", "return_rate", "q4_share", "uk_share"]
        if not g.single:
            rows = []
            for code in g.codes:
                r = self.seg.loc[code]
                rows.append({"Segment": f"{r.SEGMENT_NAME} ({code})",
                             "Customers": fmt(SEGMENT_METRICS["customers"], r.CUSTOMERS),
                             "Predicted revenue next quarter": fmt(SEGMENT_METRICS["expected_revenue"], r.EXPECTED_REVENUE),
                             "Chance of ordering next quarter": pct(r.MEAN_P_ACTIVE),
                             "What stands out": r.TRAITS})
            total = self.seg.loc[g.codes].CUSTOMERS.sum()
            return Result([f"{g.label} covers {len(g.codes)} segments with {total:,} customers in total."],
                          pd.DataFrame(rows))
        r = self.seg.loc[g.codes[0]]
        lines = [f"{SEGMENT_METRICS[k].label.capitalize()}: {fmt(SEGMENT_METRICS[k], r[SEGMENT_METRICS[k].column])}"
                 for k in keys]
        share = r.CUSTOMERS / self.seg.CUSTOMERS.sum()
        fact = (f"{g.label}: {r.CUSTOMERS:,} customers ({pct(share)} of all customers). "
                f"What stands out: {r.TRAITS}.")
        table = pd.DataFrame({"Measure": [l.split(": ")[0] for l in lines], "Value": [l.split(": ", 1)[1] for l in lines]})
        return Result([fact], table, ["Medians describe the typical customer in the segment over the last 12 months."])

    def share(self, groups: list[Group], m: Metric) -> Result:
        if m.agg != "sum":
            raise ValueError(f"a share of the total needs an additive measure (customers or expected_revenue), not {m.key}")
        total = float(self.seg[m.column].sum())
        facts, rows = [], []
        for g in groups:
            part = float(self.seg.loc[g.codes, m.column].sum())
            facts.append(f"{g.label} holds {pct(part / total)} of all {m.label} ({fmt(m, part)} of {fmt(m, total)}).")
            rows.append({"Group": g.label, cap(m.label): fmt(m, part), "Share of total": pct(part / total)})
        return Result(facts, pd.DataFrame(rows))

    def share_within(self, part: Group, whole: Group, m: Metric) -> Result:
        p = float(self.seg.loc[part.codes, m.column].sum())
        w = float(self.seg.loc[whole.codes, m.column].sum())
        fact = (f"{part.label} holds {pct(p / w)} of {whole.label}'s {m.label} "
                f"({fmt(m, p)} of {fmt(m, w)}).")
        return Result([fact], pd.DataFrame([{"Part": part.label, "Whole": whole.label, cap(m.label): fmt(m, p),
                                             "Whole total": fmt(m, w), "Share": pct(p / w)}]))

    def rank(self, m: Metric, level: str, direction: int, n: int = 1, within: list[Group] | None = None,
             nth: int = 1) -> Result:
        """Rank segments, tiers or buying styles. direction +1 = highest first. Ties are kept.
        `nth` > 1 asks for the second, third ... in that order."""
        codes = list(self.seg.SEGMENT)
        if within:
            codes = [c for c in codes if any(c in g.codes for g in within)]
        if level in ("tier", "style"):
            keys = sorted({c[:2] if level == "tier" else c[3:] for c in codes})
            groups = [resolve_one(k, self.seg.reset_index(drop=True)) for k in keys]
            groups = [Group(g.kind, [c for c in g.codes if c in codes], g.label) for g in groups]
            if all(len(g.codes) == 1 for g in groups):
                # "within bulk buyers, which tier ...": each tier is one segment, so rank the segments.
                return self.rank(m, "segment", direction, n, within, nth)
            restricted = any(len(g.codes) < len(resolve_one(g.codes[0][:2] if level == "tier" else g.codes[0][3:],
                                                            self.seg.reset_index(drop=True)).codes) for g in groups)
            if restricted:  # say which segments each group stands for
                groups = [Group(g.kind, g.codes, f"{g.label.split(' (')[0]} ({', '.join(g.codes)})") for g in groups]
            if m.agg == "median":
                return self._rank_median_groups(m, level, groups, direction)
            vals = []
            for g in groups:
                v, note = self.group_value(g, m)
                if v is None:
                    raise ValueError(note)
                vals.append((g.label, v, note))
            table = pd.DataFrame(vals, columns=["Group", "value", "note"])
            label_col = "Group"
        else:
            part = self.seg.loc[codes]
            table = pd.DataFrame({"Segment": [f"{nm} ({c})" for c, nm in zip(part.SEGMENT, part.SEGMENT_NAME)],
                                  "value": part[m.column].astype(float).to_numpy(), "note": None})
            label_col = "Segment"
        table = table.sort_values("value", ascending=direction < 0, kind="stable").reset_index(drop=True)
        best = table.value.iloc[0]
        tied = table[table.value == best]
        side = m.high if direction > 0 else m.low
        if nth > 1 and nth <= len(table):
            r = table.iloc[nth - 1]
            word = {2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(nth, f"{nth}th")
            fact = (f"{r[label_col]} has the {word} {'highest' if direction > 0 else 'lowest'} {m.label}: "
                    f"{fmt(m, r.value)}.")
            shown = table.head(max(nth, n, 3))
        elif n <= 1:
            if len(tied) > 1:
                names = ", ".join(tied[label_col])
                fact = f"{len(tied)} are tied for {('highest' if direction > 0 else 'lowest')} {m.label} at {fmt(m, best)}: {names}."
            else:
                fact = (f"{tied[label_col].iloc[0]} has the {('highest' if direction > 0 else 'lowest')} "
                        f"{m.label}: {fmt(m, best)}, meaning {side}.")
            shown = table.head(max(len(tied), 3))
        else:
            shown = table.head(n)
            order = ", ".join(f"{r[label_col]} {fmt(m, r.value)}" for _, r in shown.iterrows())
            fact = f"Ranked by {m.label}, {'highest' if direction > 0 else 'lowest'} first: {order}."
        notes = [x for x in table.note.dropna().unique()]
        out = shown[[label_col]].assign(**{cap(m.label): [fmt(m, v) for v in shown.value]})
        out.insert(0, "Rank", range(1, len(out) + 1))
        return Result([fact], out, notes)

    def _rank_median_groups(self, m: Metric, level: str, groups: list[Group], direction: int) -> Result:
        """Tiers or styles on a per-segment median: medians can't be combined, so each group is
        shown as the range of its segments' medians, and a group leads only if every one of its
        segments beats every segment of the others."""
        spans = []
        for g in groups:
            v = self.seg.loc[g.codes, m.column].astype(float)
            spans.append((g, float(v.min()), float(v.max()), float(v.mean())))
        spans.sort(key=lambda s: s[3], reverse=direction > 0)
        word = "value tier" if level == "tier" else "buying style"
        lead, rest = spans[0], spans[1:]
        clear = (lead[1] > max(s[2] for s in rest)) if direction > 0 else (lead[2] < min(s[1] for s in rest))
        name = lead[0].label.split(" (")[0] + f" ({lead[0].codes[0][:2] if level == 'tier' else lead[0].codes[0][3:]})"
        span = lambda s: f"{fmt(m, s[1])} to {fmt(m, s[2])}"  # noqa: E731
        if clear:
            fact = (f"{name} has the {'highest' if direction > 0 else 'lowest'} {m.label}: every one of its segments "
                    f"({span(lead)}) is {'above' if direction > 0 else 'below'} every segment of the other "
                    f"{word}s.")
        else:
            fact = (f"No {word} is {'highest' if direction > 0 else 'lowest'} in every segment. Ordered by the "
                    f"average of their segments' medians: " + ", ".join(
                        f"{s[0].label.split(' (')[0]} {span(s)}" for s in spans) + ".")
        rows = []
        for g, *_ in spans:
            part = self.seg.loc[g.codes].sort_values(m.column, ascending=direction < 0)
            rows += [{cap(word): g.label.split(" (")[0], "Segment": f"{r.SEGMENT_NAME} ({r.SEGMENT})",
                      cap(m.label): fmt(m, r[m.column])} for _, r in part.iterrows()]
        note = (f"{cap(m.label)} is a median per segment, and medians can't be combined, so each {word} is "
                "shown as the range of its segments.")
        return Result([fact], pd.DataFrame(rows), [note])

    def mean_per_segment(self, groups: list[Group], m: Metric) -> Result:
        """'Average segment size': a group's total divided by its number of segments."""
        facts, rows = [], []
        for g in groups:
            total = float(self.seg.loc[g.codes, m.column].sum())
            avg = total / len(g.codes)
            facts.append(f"{g.label}: {len(g.codes)} segments averaging {fmt(m, avg)} {m.label} each "
                         f"({fmt(m, total)} in total).")
            rows.append({"Group": g.label, "Segments": len(g.codes), f"Average {m.label}": fmt(m, avg),
                         "Total": fmt(m, total)})
        median = float(self.seg.loc[groups[0].codes, m.column].median())
        return Result(facts, pd.DataFrame(rows) if len(rows) > 1 else pd.DataFrame(),
                      [f"The median segment has {fmt(m, median)} {m.label}."])

    def compare(self, a: Group, b: Group, metrics: list[Metric], asked: int = 0, question_metric: Metric | None = None) -> Result:
        facts, rows, notes = [], [], []
        for m in metrics:
            va, na = self.group_value(a, m)
            vb, nb = self.group_value(b, m)
            if va is None or vb is None:
                paired = self._paired(a, b, m)
                if paired:
                    fact, prow, a_wins, b_wins, n = paired
                    if asked and m is (question_metric or metrics[0]):
                        yes = a_wins == n if asked > 0 else b_wins == n
                        no = b_wins == n if asked > 0 else a_wins == n
                        fact = ("Yes. " if yes else "No. " if no else "Not consistently. ") + fact
                    facts.append(fact)
                    rows.extend(prow)
                else:  # show every segment of both groups
                    for g in (a, b):
                        rows.extend({"Measure": m.label, "Segment": r["Segment"], "Value": list(r.values())[1]}
                                    for r in self._segment_rows(g.codes, m).to_dict("records"))
                    facts.append(f"{cap(m.label)} is a median per segment, so the segments of each group are listed.")
                continue
            notes += [x for x in (na, nb) if x]
            verdict = ""
            if asked and m is (question_metric or metrics[0]) and va != vb:
                verdict = "Yes. " if (va > vb) == (asked > 0) else "No. "
            if va == vb:
                tail = "They are equal."
            else:
                top = a if va > vb else b
                name = top.label if top.kind == "combined" else top.label.split(" (")[0] + (
                    f" ({top.codes[0]})" if top.single else "")
                tail = f"{name} is higher, meaning {m.high}."
                tail += " " + difference(m, va, vb)
            facts.append(f"{verdict}{cap(m.label)}: {a.label} {fmt(m, va)} vs {b.label} {fmt(m, vb)}. {tail}")
            rows.append({"Measure": m.label, a.label: fmt(m, va), b.label: fmt(m, vb)})
        if not facts:
            return Result(["These measures can't be compared for these groups."], pd.DataFrame(), list(dict.fromkeys(notes)))
        return Result(facts, pd.DataFrame(rows), list(dict.fromkeys(notes)))

    def crosstab(self, m: Metric, as_share: bool = False) -> Result:
        """Value tier (rows) x buying style (columns). Every cell is one segment; totals are
        shown only for measures that add up (or average, weighted by customers). `as_share`
        shows each cell of an additive measure as a share of the grand total."""
        tiers, styles = list(TIER_NAMES), list(BEHAVIOR_NAMES)
        if as_share and m.agg == "sum":
            total = float(self.seg[m.column].sum())
            rows = []
            for t in tiers:
                row = {"Value tier": f"{TIER_NAMES[t]} ({t})"}
                row.update({f"{BEHAVIOR_NAMES[s]} ({s})": pct(self.seg.loc[f"{t}-{s}", m.column] / total) for s in styles})
                row["All styles"] = pct(self.seg.loc[[f"{t}-{s}" for s in styles], m.column].sum() / total)
                rows.append(row)
            row = {"Value tier": "All tiers"}
            row.update({f"{BEHAVIOR_NAMES[s]} ({s})": pct(self.seg.loc[[f"{t}-{s}" for t in tiers], m.column].sum() / total)
                        for s in styles})
            row["All styles"] = pct(1.0)
            rows.append(row)
            top = self.seg[m.column].idxmax()
            fact = (f"{cap(m.label)} by value tier and buying style, as shares of the total ({fmt(m, total)}). "
                    f"The largest cell is {segment_name(top)} ({top}) at {pct(self.seg.loc[top, m.column] / total)}.")
            return Result([fact], pd.DataFrame(rows))
        grid = {t: {s: float(self.seg.loc[f"{t}-{s}", m.column]) for s in styles} for t in tiers}
        rows = []
        for t in tiers:
            row = {"Value tier": f"{TIER_NAMES[t]} ({t})"}
            row.update({f"{BEHAVIOR_NAMES[s]} ({s})": fmt(m, grid[t][s]) for s in styles})
            if m.agg != "median":
                v, _ = self.group_value(resolve_one(t, self.seg.reset_index(drop=True)), m)
                row["All styles"] = fmt(m, v)
            rows.append(row)
        notes = []
        if m.agg != "median":
            total_row = {"Value tier": "All tiers"}
            for s in styles:
                v, _ = self.group_value(resolve_one(s, self.seg.reset_index(drop=True)), m)
                total_row[f"{BEHAVIOR_NAMES[s]} ({s})"] = fmt(m, v)
            v, note = self.group_value(Group("all", list(self.seg.SEGMENT), "all"), m)
            total_row["All styles"] = fmt(m, v)
            rows.append(total_row)
            if note:
                notes.append(note)
        else:
            notes.append(f"Each cell is the median customer's {m.label} in that segment. Medians don't add "
                         "up, so there are no totals; for a measure with totals, ask for predicted revenue "
                         "next quarter or customers.")
        if m.key == "net_revenue_12m":
            notes.append("Total sales per segment aren't in the published results, only each segment's median.")
        top_t, top_s = max(((t, s) for t in tiers for s in styles), key=lambda ts: grid[ts[0]][ts[1]])
        fact = (f"{cap(m.label)} by value tier and buying style. The largest cell is "
                f"{TIER_NAMES[top_t]} / {BEHAVIOR_NAMES[top_s]} ({top_t}-{top_s}) at {fmt(m, grid[top_t][top_s])}.")
        return Result([fact], pd.DataFrame(rows), notes)

    def _paired(self, a: Group, b: Group, m: Metric):
        """Two buying styles (or two tiers) on a median: compare them tier by tier (or style by style)."""
        if a.kind != b.kind or a.kind not in ("style", "tier"):
            return None
        rows, a_wins, b_wins = [], 0, 0
        for ca in a.codes:
            other = next(c for c in b.codes if (c[:2] == ca[:2] if a.kind == "style" else c[3:] == ca[3:]))
            va, vb = float(self.seg.loc[ca, m.column]), float(self.seg.loc[other, m.column])
            a_wins += va > vb
            b_wins += vb > va
            rows.append({"Measure": m.label, a.label.split(" (")[0]: f"{ca} {fmt(m, va)}",
                         b.label.split(" (")[0]: f"{other} {fmt(m, vb)}"})
        n = len(rows)
        across = "value tiers" if a.kind == "style" else "buying styles"
        an, bn = a.label.split(" (")[0], b.label.split(" (")[0]
        ties = n - a_wins - b_wins
        fact = (f"{cap(m.label)} is a median per segment, so the groups are compared segment by segment: "
                f"{an} is higher in {a_wins} of {n} {across} and {bn} in {b_wins}"
                + (f" ({ties} tied)." if ties else "."))
        return fact, rows, a_wins, b_wins, n

    def compare_many(self, groups: list[Group], metrics: list[Metric]) -> Result:
        """Three or more groups side by side on one or more measures, naming the top and bottom of each."""
        if any(not g.single for g in groups) and any(m.agg == "median" for m in metrics):
            codes = list(dict.fromkeys(c for grp in groups for c in grp.codes))
            return self.segment_table(codes, metrics)
        facts, notes, rows = [], [], {g.label: {"Group": g.label} for g in groups}
        for m in metrics:
            vals = []
            for g in groups:
                v, note = self.group_value(g, m)
                vals.append((g.label, v))
                rows[g.label][cap(m.label)] = fmt(m, v)
                if note:
                    notes.append(note)
            vals.sort(key=lambda t: -t[1])
            facts.append(f"Highest {m.label}: {vals[0][0]} {fmt(m, vals[0][1])}; "
                         f"lowest: {vals[-1][0]} {fmt(m, vals[-1][1])}.")
        return Result(facts, pd.DataFrame(list(rows.values())), list(dict.fromkeys(notes)))

    def segment_table(self, codes: list[str], metrics: list[Metric], traits: bool = False) -> Result:
        """One row per segment, one column per measure; totals for measures that add up."""
        part = self.seg.loc[codes]
        table = pd.DataFrame({"Segment": [f"{n} ({c})" for c, n in zip(part.SEGMENT, part.SEGMENT_NAME)]})
        for m in metrics:
            table[cap(m.label)] = [fmt(m, v) for v in part[m.column]]
        if traits:
            table["What stands out"] = list(part.TRAITS)
        facts = [f"{len(codes)} segment(s): " + ", ".join(cap(m.label) for m in metrics)
                 + (" and traits" if traits else "") + "."]
        for m in metrics:
            if m.agg in ("sum", "weighted", "ratio") and len(codes) > 1:
                v, _ = self.group_value(Group("combined", codes, "together"), m)
                facts.append(f"Together: {m.label} {fmt(m, v)}.")
        notes = (["Medians are per segment and aren't combined."]
                 if any(m.agg == "median" for m in metrics) and len(codes) > 1 else [])
        return Result(facts, table, notes)

    def breakdown(self, level: str, m: Metric, more: list[Metric] | None = None) -> Result:
        """Totals per value tier or buying style, with each one's share of the whole."""
        if more:
            keys = list(TIER_NAMES) if level == "tier" else list(BEHAVIOR_NAMES)
            base = self.seg.reset_index(drop=True)
            rows, notes = [], []
            for k in keys:
                g = resolve_one(k, base)
                row = {"Group": g.label}
                for mm in [m, *more]:
                    v, note = self.group_value(g, mm)
                    row[cap(mm.label)] = fmt(mm, v) if v is not None else "per segment only"
                    if note:
                        notes.append(note)
                rows.append(row)
            word = "value tier" if level == "tier" else "buying style"
            return Result([f"{len(keys)} {word}s: " + ", ".join(mm.label for mm in [m, *more]) + "."],
                          pd.DataFrame(rows), list(dict.fromkeys(notes)))
        if m.agg == "median":
            return self.crosstab(m)
        keys = list(TIER_NAMES) if level == "tier" else list(BEHAVIOR_NAMES)
        base = self.seg.reset_index(drop=True)
        rows, total_v = [], self.group_value(Group("all", list(self.seg.SEGMENT), "all"), m)[0]
        for k in keys:
            g = resolve_one(k, base)
            v, _ = self.group_value(g, m)
            row = {"Group": g.label, cap(m.label): fmt(m, v)}
            if m.agg == "sum":
                row["Share of total"] = pct(v / total_v)
            rows.append(row)
        word = "value tier" if level == "tier" else "buying style"
        return Result([f"{cap(m.label)} by {word} (total {fmt(m, total_v)})."], pd.DataFrame(rows))

    def count_where(self, m: Metric, op: str, threshold: float, within: list[Group] | None = None,
                    report: Metric | None = None) -> Result:
        """Segments whose `m` meets a condition. `report` is a second, additive measure to total
        over those segments ("expected revenue from segments whose P(buys again) > 50%")."""
        ops = {">": lambda s: s > threshold, ">=": lambda s: s >= threshold,
               "<": lambda s: s < threshold, "<=": lambda s: s <= threshold,
               "=": lambda s: (s - threshold).abs() < 1e-9, "==": lambda s: (s - threshold).abs() < 1e-9,
               "!=": lambda s: (s - threshold).abs() >= 1e-9}
        if op not in ops:
            raise ValueError(f"op must be one of {', '.join(ops)}")
        pool = self.seg if not within else self.seg.loc[list(dict.fromkeys(c for g in within for c in g.codes))]
        mask = ops[op](pool[m.column])
        part = pool[mask]
        total = self.seg.CUSTOMERS.sum()
        word = {"=": "equal to", "==": "equal to", "!=": "other than", ">": "above", "<": "below",
                ">=": "at or above", "<=": "at or below"}.get(op, op)
        facts = [f"{len(part)} of {len(pool)} segments have {m.label} {word} {fmt(m, threshold)}. They hold "
                 f"{part.CUSTOMERS.sum():,} customers, {pct(part.CUSTOMERS.sum() / total)} of all {total:,}."]
        table = self._segment_rows(list(part.SEGMENT), m).assign(Customers=[f"{c:,}" for c in part.CUSTOMERS])
        if report is not None and report.key not in (m.key, "customers"):
            if report.agg == "sum":
                got, whole = float(part[report.column].sum()), float(self.seg[report.column].sum())
                facts.append(f"Together they have {fmt(report, got)} {report.label}, {pct(got / whole if whole else 0)} "
                             f"of the {fmt(report, whole)} total.")
            elif len(part):
                v, _ = self.group_value(Group("combined", list(part.SEGMENT), "these segments"), report)
                if v is not None:
                    facts.append(f"Together their {report.label} is {fmt(report, v)}.")
            table[cap(report.label)] = [fmt(report, v) for v in part[report.column]]
        return Result(facts, table)

    def traits(self, word: str) -> Result:
        hit = self.seg[self.seg.TRAITS.str.contains(word, case=False, na=False)]
        if hit.empty:
            return Result([f"No segment's distinguishing traits mention '{word}'."])
        table = pd.DataFrame({"Segment": [f"{n} ({c})" for c, n in zip(hit.SEGMENT, hit.SEGMENT_NAME)],
                              "What stands out": hit.TRAITS, "Customers": [f"{c:,}" for c in hit.CUSTOMERS]})
        return Result([f"{len(hit)} segment(s) are described as '{word}': " + ", ".join(table.Segment) + "."], table)

    def schemes_by_fold(self, metrics: list[Metric], schemes: list[str]) -> Result:
        """Each fold's value, side by side, for the named schemes."""
        rows, facts = [], []
        for key in schemes:
            part = self.bt[self.bt.SCHEME == key].set_index("RUN_ID")
            for m in metrics:
                q2 = part[m.column].get("backtest_q2")
                q3 = part[m.column].get("backtest_q3")
                if q2 is None or q3 is None or pd.isna(q2) or pd.isna(q3):
                    continue
                rows.append({"Scheme": SCHEME_NAMES[key], "Measure": m.label, "q2 backtest": fmt(m, q2),
                             "q3 backtest": fmt(m, q3)})
                better = "q2" if q2 > q3 else "q3" if q3 > q2 else "neither"
                facts.append(f"{SCHEME_NAMES[key]}, {m.label}: {fmt(m, q2)} in q2 vs {fmt(m, q3)} in q3 "
                             f"(mean {fmt(m, (q2 + q3) / 2)}), so it was higher in {better}. {difference(m, q2, q3)}")
        return Result(facts or ["Those schemes don't have that measure in both backtests."], pd.DataFrame(rows),
                      ["q2: fit on 2011-03-10, scored on 2011-06-10. q3: fit on 2011-06-10, scored on 2011-09-10."])

    def fold_dates(self) -> Result:
        d = self.bt.groupby("RUN_ID")[["TRAIN_SNAPSHOT", "EVAL_SNAPSHOT"]].first()
        facts = [f"{run.replace('backtest_', '')}: fit on {r.TRAIN_SNAPSHOT}, scored on {r.EVAL_SNAPSHOT}, with outcomes "
                 "over the next 91 days." for run, r in d.iterrows()]
        return Result(facts, d.reset_index().rename(columns={"RUN_ID": "Backtest", "TRAIN_SNAPSHOT": "Fit (train) snapshot",
                                                              "EVAL_SNAPSHOT": "Scored (eval) snapshot"}))

    def k_detail(self, axis: str | None, question: str) -> Result:
        """Questions about the k-selection table itself: one k, a ranking of k, or the candidates."""
        q = question.lower()
        cols = {"stability": "STABILITY_ARI", "ari": "STABILITY_ARI", "silhouette": "SILHOUETTE",
                "r2": "NEXT_REVENUE_R2", "r²": "NEXT_REVENUE_R2", "r-squared": "NEXT_REVENUE_R2",
                "smallest segment": "MIN_SHARE", "min share": "MIN_SHARE", "minimum share": "MIN_SHARE"}
        wanted = list(dict.fromkeys(c for w, c in cols.items() if w in q))
        col = wanted[0] if wanted else None
        k = self.k if not axis else self.k[self.k.AXIS == axis]
        m = re.search(r"\bk\s*=?\s*(\d)\b", q)
        label = {"STABILITY_ARI": "stability ARI", "SILHOUETTE": "silhouette", "NEXT_REVENUE_R2": "next-revenue R²",
                 "MIN_SHARE": "smallest segment share"}
        def val(c, v):
            return pct(v) if c == "MIN_SHARE" else f"{v:.4f}"
        if len(wanted) > 1 or (col and re.search(r"\btable\b", q)):
            part = k[["AXIS", "K", *wanted, "CHOSEN"]].rename(columns=label)
            return Result([f"k candidates with {', '.join(label[c] for c in wanted)}."], part)
        if m and col:
            part = k[k.K == int(m.group(1))]
            facts = [f"{r.AXIS.capitalize()} axis, k={int(r.K)}: {label[col]} {val(col, r[col])}." for _, r in part.iterrows()]
            return Result(facts or [f"No k={m.group(1)} was evaluated."], part[["AXIS", "K", col]].rename(columns={col: label[col]}))
        if col and re.search(r"\b(highest|lowest|best|worst|most|least|max|min)", q):
            down = re.search(r"\b(lowest|worst|least|min)", q) is not None
            facts = []
            for ax, part in k.groupby("AXIS", sort=False):
                r = part.sort_values(col, ascending=down).iloc[0]
                facts.append(f"{ax.capitalize()} axis: k={int(r.K)} has the {'lowest' if down else 'highest'} "
                             f"{label[col]} ({val(col, r[col])}).")
            return Result(facts, k[["AXIS", "K", col]].rename(columns={col: label[col]}))
        if re.search(r"\b(how many|candidates|range)\b", q):
            facts = [f"{ax.capitalize()} axis: {len(part)} candidates, k={int(part.K.min())} to {int(part.K.max())}."
                     for ax, part in k.groupby("AXIS", sort=False)]
            return Result(facts, k[["AXIS", "K", "ELIGIBLE", "CHOSEN"]])
        return self.k_choice(axis)

    def _bt(self, fold: str | None):
        if fold in ("q2", "q3"):
            return self.bt[self.bt.RUN_ID == f"backtest_{fold}"], f"the {fold} backtest"
        return self.bt, "mean of two backtests"

    def schemes(self, metrics: list[Metric], schemes: list[str] | None = None, direction: int = 0,
                fold: str | None = None, average: bool = False) -> Result:
        """Schemes on backtest measures. `average` adds the mean across the schemes shown."""
        result = self._schemes(metrics, schemes, direction, fold)
        if average:
            keys = schemes or list(SCHEME_NAMES)
            bt, basis = self._bt(fold)
            m0 = metrics[0]
            per = bt[bt.SCHEME.isin(keys)].groupby("SCHEME")[m0.column].mean().dropna()
            if len(per):
                result.facts.insert(0, f"Average {m0.label} across the {len(per)} schemes ({basis}): "
                                       f"{fmt(m0, per.mean())}.")
        return result

    def _schemes(self, metrics: list[Metric], schemes: list[str] | None = None, direction: int = 0,
                 fold: str | None = None) -> Result:
        keys = schemes or list(SCHEME_NAMES)
        bt, basis = self._bt(fold)
        if any(m.key == "n_segments" for m in metrics):
            part = bt[bt.SCHEME.isin(keys)].dropna(subset=["N_SEGMENTS"])
            counts = part.groupby("SCHEME").N_SEGMENTS.apply(lambda s: " and ".join(str(int(v)) for v in s))
            facts = [f"{SCHEME_NAMES[k]}: {v} segments ({basis})." for k, v in counts.items()]
            return Result(facts, pd.DataFrame({"Scheme": [SCHEME_NAMES[k] for k in counts.index],
                                               "Segments": list(counts)}),
                          ["Segment counts differ by backtest because k is chosen afresh each time."])
        g = bt[bt.SCHEME.isin(keys)].groupby("SCHEME")
        rows, facts = [], []
        m0 = metrics[0]
        means = g[m0.column].mean().dropna()
        for m in metrics:
            for key, s in g[m.column]:
                s = s.dropna()
                if s.empty:
                    continue
                rows.append({"Scheme": SCHEME_NAMES[key], "Measure": m.label,
                             "Mean of 2 backtests": fmt(m, s.mean()),
                             "Range": f"{fmt(m, s.min())} to {fmt(m, s.max())}"})
        table = pd.DataFrame(rows)
        if direction and len(means) > 1:
            order = means.sort_values(ascending=direction < 0)
            best = order.iloc[0]
            tied = order[order == best]
            names = ", ".join(SCHEME_NAMES[k] for k in tied.index)
            facts.append(f"{names} {'has' if len(tied) == 1 else 'have'} the {'highest' if direction > 0 else 'lowest'} "
                         f"{m0.label} ({fmt(m0, best)}, {basis}).")
            ranked = ", ".join(f"{SCHEME_NAMES[k]} {fmt(m0, v)}" for k, v in order.items())
            facts.append(f"Order: {ranked}.")
        elif len(means) == 1:
            k = means.index[0]
            facts.append(f"{SCHEME_NAMES[k]}: {m0.label} {fmt(m0, means.iloc[0])} ({basis}).")
        else:
            facts.append("; ".join(f"{SCHEME_NAMES[k]} {fmt(m0, v)}" for k, v in means.sort_values(ascending=False).items())
                         + f" ({m0.label}, {basis}).")
        notes = []
        if "outcome_model_continuous" in means.index:
            notes.append("The unsegmented outcome model isn't a segmentation; it's the ceiling the segmented schemes are compared against.")
        if len(metrics) == 1 and not table.empty:
            table = table.drop(columns="Measure").rename(columns={"Mean of 2 backtests": cap(m0.label)})
        if len(metrics) > 1:
            table = table.pivot_table(index="Scheme", columns="Measure", values="Mean of 2 backtests", aggfunc="first").reset_index()
        return Result(facts, table, notes)

    def compare_schemes(self, a: str, b: str, metrics: list[Metric], asked: int = 0, fold: str | None = None) -> Result:
        facts, rows = [], []
        bt, basis = self._bt(fold)
        for i, m in enumerate(metrics):
            va = bt[bt.SCHEME == a][m.column].mean()
            vb = bt[bt.SCHEME == b][m.column].mean()
            if pd.isna(va) or pd.isna(vb):
                continue
            verdict = ("Yes. " if (va > vb) == (asked > 0) else "No. ") if asked and i == 0 and va != vb else ""
            facts.append(f"{verdict}{SCHEME_NAMES[a]}: {fmt(m, va)} vs {SCHEME_NAMES[b]}: {fmt(m, vb)} on {m.label}, "
                         f"so {SCHEME_NAMES[a]} {m.high if va > vb else m.low}. {difference(m, va, vb)}")
            rows.append({"Measure": m.label, SCHEME_NAMES[a]: fmt(m, va), SCHEME_NAMES[b]: fmt(m, vb)})
        return Result(facts or ["Those schemes don't both have that measure."], pd.DataFrame(rows),
                      [f"Each value is from the {basis}." if fold else "Each value is the mean of two out-of-time backtests."])

    def k_choice(self, axis: str | None = None) -> Result:
        k = self.k if not axis else self.k[self.k.AXIS == axis]
        facts, rows = [], []
        for ax, part in k.groupby("AXIS", sort=False):
            chosen = part[part.CHOSEN.astype(bool)].iloc[0]
            crit = "out-of-fold R² for next-quarter revenue" if chosen.CRITERION == "next_revenue_r2" else "silhouette"
            col = "NEXT_REVENUE_R2" if chosen.CRITERION == "next_revenue_r2" else "SILHOUETTE"
            best = part.loc[part[part.ELIGIBLE.astype(bool)][col].idxmax()]
            fact = (f"{cap(ax)} axis: k={int(chosen.K)} was chosen by {crit} "
                    f"({chosen[col]:.4f}), with stability ARI {chosen.STABILITY_ARI:.4f} and smallest segment "
                    f"{pct(chosen.MIN_SHARE)}.")
            if int(best.K) != int(chosen.K):
                fact += (f" k={int(best.K)} scored slightly higher ({best[col]:.4f}), but the rule prefers fewer "
                         "segments when a larger k is better by no more than 0.01.")
            facts.append(fact)
            for _, r in part.iterrows():
                rows.append({"Axis": ax, "k": int(r.K), "Stability ARI": f"{r.STABILITY_ARI:.3f}",
                             "Smallest segment": pct(r.MIN_SHARE), "Silhouette": f"{r.SILHOUETTE:.3f}",
                             "Next-revenue R²": f"{r.NEXT_REVENUE_R2:.3f}", "Eligible": "yes" if r.ELIGIBLE else "no",
                             "Chosen": "yes" if r.CHOSEN else ""})
        if not axis:
            facts.append(f"The production hybrid crosses {self.seg.SEGMENT.str[:2].nunique()} value tiers with "
                         f"{self.seg.SEGMENT.str[3:].nunique()} buying styles: {len(self.seg)} segments.")
        return Result(facts, pd.DataFrame(rows),
                      ["Eligible means bootstrap stability ARI ≥ 0.85 and every segment holding ≥ 5% of customers."])

    def cleaning(self, step: str | None = None, from_raw: bool = False) -> Result:
        c = self.clean
        raw, final = int(c.ROWS_REMAINING.iloc[0]), int(c.ROWS_REMAINING.iloc[-1])
        rows = [{"Step": r.STAGE, "Rows left": f"{int(r.ROWS_REMAINING):,}",
                 "Rows removed": "" if pd.isna(r.CHANGE) else f"{int(-r.CHANGE):,}"} for r in c.itertuples()]
        if step:
            hit = c[c.STAGE.str.contains(step, case=False, regex=False)]
            if hit.empty:  # "customer-id step": match on words, ignoring hyphens and "step"
                words = [w for w in re.findall(r"[a-z]+", step.lower()) if w not in ("step", "the", "rows", "removing")]
                if words:
                    hit = c[c.STAGE.str.lower().str.replace("-", " ").apply(lambda s: all(w in s for w in words))]
            if not hit.empty:
                r = hit.iloc[0]
                if from_raw:
                    gone = raw - int(r.ROWS_REMAINING)
                    return Result([f"From the raw data to '{r.STAGE}', {gone:,} rows were removed "
                                   f"({pct(gone / raw)} of {raw:,}); {int(r.ROWS_REMAINING):,} rows were left."],
                                  pd.DataFrame(rows))
                removed = ""
                if not pd.isna(r.CHANGE):
                    before = int(r.ROWS_REMAINING - r.CHANGE)
                    removed = (f" It removed {int(-r.CHANGE):,} rows, {pct(-r.CHANGE / before)} of the "
                               f"{before:,} rows before it ({pct(-r.CHANGE / raw)} of the raw data).")
                return Result([f"Step '{r.STAGE}': {int(r.ROWS_REMAINING):,} rows left.{removed}"], pd.DataFrame(rows))
        biggest = c.loc[c.CHANGE.idxmin()]
        facts = [f"Cleaning took the data from {raw:,} raw rows to {final:,} product lines with a customer ID, "
                 f"removing {raw - final:,} rows ({pct((raw - final) / raw)}), so {pct(final / raw)} of the raw rows "
                 f"survived. The biggest single step was "
                 f"'{biggest.STAGE}' ({int(-biggest.CHANGE):,} rows)."]
        return Result(facts, pd.DataFrame(rows))
