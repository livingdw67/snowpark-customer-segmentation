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
           ("customers", "size", "biggest", "largest", "smallest", "how many")),
    Metric("expected_revenue", "EXPECTED_REVENUE", "predicted revenue next quarter", "money", "sum",
           "more predicted revenue", "less predicted revenue", ("valuable",), (),
           ("expected revenue", "predicted revenue", "forecast revenue", "valuable", "worth", "value next quarter")),
    Metric("p_buys_again", "MEAN_P_ACTIVE", "chance of ordering next quarter", "percent", "weighted",
           "more likely to order again", "more at risk of churning",
           ("likely to buy", "likely to order", "loyal", "retention", "retain", "active"),
           ("churn", "at risk", "risk", "unlikely"),
           ("buy again", "buys again", "order again", "p(buys again)", "p_active", "probability", "likely",
            "churn", "retention", "at risk", "loyal")),
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
           "larger orders by value", "smaller orders by value", (), (),
           ("order value", "aov", "basket value", "spend per order")),
    Metric("units_per_order", "MEDIAN_AVG_UNITS_PER_ORDER", "units per order", "number", "median",
           "bigger baskets", "smaller baskets", (), (), ("units", "basket size", "units per order", "bulk")),
    Metric("unit_price", "MEDIAN_AVG_UNIT_PRICE", "unit price", "money", "median",
           "pay more per unit", "pay less per unit", ("premium", "expensive"), ("cheap",),
           ("unit price", "price", "premium", "expensive", "cheap")),
    Metric("lines_per_order", "MEDIAN_AVG_LINES_PER_ORDER", "product lines per order", "number", "median",
           "wider baskets", "narrower baskets", (), (), ("lines", "product lines", "variety")),
    Metric("return_rate", "MEDIAN_RETURN_RATE", "return rate", "percent", "median",
           "return more", "return less", (), (), ("return", "returns", "refund")),
    Metric("q4_share", "MEDIAN_Q4_REVENUE_SHARE", "share of revenue bought in October-December",
           "percent", "median", "more seasonal", "more year-round",
           ("seasonal", "holiday-heavy", "christmas"), ("year-round", "steady"),
           ("seasonal", "q4", "holiday season", "christmas", "october", "year-round")),
    Metric("tenure_days", "MEDIAN_TENURE_DAYS", "days since first order (tenure)", "days", "median",
           "longer-standing customers", "newer customers", ("longest-standing",), (), ("tenure", "first order", "how long", "longest-standing", "newest", "age")),
    Metric("uk_share", "UK_SHARE", "share of customers in the UK", "percent", "weighted",
           "more UK-based", "more international", (), (), ("uk", "united kingdom", "domestic")),
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


def metrics_named_in(text: str, level: str = "segment") -> list[str]:
    """Metrics whose synonyms appear in `text`, longest match first."""
    lower = text.lower()
    table = SCHEME_METRICS if level == "scheme" else SEGMENT_METRICS
    hits = []
    for m in table.values():
        best = max((len(s) for s in m.synonyms if s in lower), default=0)
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


def pct(v: float) -> str:
    return f"{v:.1%}"


# --------------------------------------------------------------------------------------
# Groups: resolve what a question or plan refers to
# --------------------------------------------------------------------------------------

TIER_WORDS = {"C1": ("top spenders", "top spender", "top tier", "top-tier"), "C2": ("core",),
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
    # Tier and style words, kept in order of appearance so "bulk or holiday" stays two groups.
    words = []
    for code, ws in TIER_WORDS.items():
        for w in ws:
            for m in re.finditer(rf"(?<![\w-]){re.escape(w)}(?![\w-])", used):
                words.append((m.start(), "tier", code))
    for code, ws in STYLE_WORDS.items():
        for w in ws:
            for m in re.finditer(rf"(?<![\w-]){re.escape(w)}(?![\w-])", used):
                words.append((m.start(), "style", code))
    words.sort()
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
    if str(text).strip().lower() in ("all", "everyone", "all customers", "total", "overall"):
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


def asked_direction(question: str, m: Metric) -> int:
    """+1 if the wording asks about the metric's high end, -1 for the low end, 0 if unclear.

    "most recent" on days-since-last-order is the LOW end (fewest days); "most at risk" on
    chance-of-ordering is the low end; "most seasonal" on Oct-Dec share is the high end.
    """
    q = question.lower()
    sup = 0
    if re.search(UP, q):
        sup = 1
    if re.search(DOWN, q):
        sup = -1 if sup == 0 else 0  # both present: unclear
    word = 0
    if any(_has(q, w) for w in m.low_words):
        word = -1
    if any(_has(q, w) for w in m.high_words):
        word = 1 if word == 0 else 0
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
                "order_value", "units_per_order", "unit_price", "return_rate", "q4_share", "uk_share"]
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

    def rank(self, m: Metric, level: str, direction: int, n: int = 1, within: list[Group] | None = None) -> Result:
        """Rank segments, tiers or buying styles. direction +1 = highest first. Ties are kept."""
        codes = list(self.seg.SEGMENT)
        if within:
            codes = [c for c in codes if any(c in g.codes for g in within)]
        if level in ("tier", "style"):
            keys = sorted({c[:2] if level == "tier" else c[3:] for c in codes})
            groups = [resolve_one(k, self.seg.reset_index(drop=True)) for k in keys]
            groups = [Group(g.kind, [c for c in g.codes if c in codes], g.label) for g in groups]
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
        if n <= 1:
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

    def compare(self, a: Group, b: Group, metrics: list[Metric], asked: int = 0, question_metric: Metric | None = None) -> Result:
        facts, rows, notes = [], [], []
        for m in metrics:
            va, na = self.group_value(a, m)
            vb, nb = self.group_value(b, m)
            if va is None or vb is None:
                notes.append(na or nb)
                continue
            notes += [x for x in (na, nb) if x]
            verdict = ""
            if asked and m is (question_metric or metrics[0]) and va != vb:
                verdict = "Yes. " if (va > vb) == (asked > 0) else "No. "
            if va == vb:
                tail = "They are equal."
            else:
                top = a if va > vb else b
                tail = f"{top.label.split(' (')[0]} is higher, meaning {m.high}."
            facts.append(f"{verdict}{cap(m.label)}: {a.label} {fmt(m, va)} vs {b.label} {fmt(m, vb)}. {tail}")
            rows.append({"Measure": m.label, a.label: fmt(m, va), b.label: fmt(m, vb)})
        if not facts:
            return Result(["These measures can't be compared for these groups."], pd.DataFrame(), list(dict.fromkeys(notes)))
        return Result(facts, pd.DataFrame(rows), list(dict.fromkeys(notes)))

    def crosstab(self, m: Metric) -> Result:
        """Value tier (rows) x buying style (columns). Every cell is one segment; totals are
        shown only for measures that add up (or average, weighted by customers)."""
        tiers, styles = list(TIER_NAMES), list(BEHAVIOR_NAMES)
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

    def count_where(self, m: Metric, op: str, threshold: float) -> Result:
        ops = {">": lambda s: s > threshold, ">=": lambda s: s >= threshold,
               "<": lambda s: s < threshold, "<=": lambda s: s <= threshold}
        if op not in ops:
            raise ValueError(f"op must be one of {', '.join(ops)}")
        mask = ops[op](self.seg[m.column])
        part = self.seg[mask]
        total = self.seg.CUSTOMERS.sum()
        shown = threshold if m.unit != "percent" else threshold
        facts = [f"{len(part)} of 12 segments have {m.label} {op} {fmt(m, shown)}. They hold "
                 f"{part.CUSTOMERS.sum():,} customers, {pct(part.CUSTOMERS.sum() / total)} of all {total:,}."]
        table = self._segment_rows(list(part.SEGMENT), m).assign(Customers=[f"{c:,}" for c in part.CUSTOMERS])
        return Result(facts, table)

    def traits(self, word: str) -> Result:
        hit = self.seg[self.seg.TRAITS.str.contains(word, case=False, na=False)]
        if hit.empty:
            return Result([f"No segment's distinguishing traits mention '{word}'."])
        table = pd.DataFrame({"Segment": [f"{n} ({c})" for c, n in zip(hit.SEGMENT, hit.SEGMENT_NAME)],
                              "What stands out": hit.TRAITS, "Customers": [f"{c:,}" for c in hit.CUSTOMERS]})
        return Result([f"{len(hit)} segment(s) are described as '{word}': " + ", ".join(table.Segment) + "."], table)

    def schemes(self, metrics: list[Metric], schemes: list[str] | None = None, direction: int = 0) -> Result:
        keys = schemes or list(SCHEME_NAMES)
        g = self.bt[self.bt.SCHEME.isin(keys)].groupby("SCHEME")
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
                         f"{m0.label} ({fmt(m0, best)}, mean of two backtests).")
            ranked = ", ".join(f"{SCHEME_NAMES[k]} {fmt(m0, v)}" for k, v in order.items())
            facts.append(f"Order: {ranked}.")
        elif len(means) == 1:
            k = means.index[0]
            facts.append(f"{SCHEME_NAMES[k]}: {m0.label} {fmt(m0, means.iloc[0])} (mean of two backtests).")
        else:
            facts.append("; ".join(f"{SCHEME_NAMES[k]} {fmt(m0, v)}" for k, v in means.sort_values(ascending=False).items())
                         + f" ({m0.label}, mean of two backtests).")
        notes = []
        if "outcome_model_continuous" in means.index:
            notes.append("The unsegmented outcome model isn't a segmentation; it's the ceiling the segmented schemes are compared against.")
        if len(metrics) == 1 and not table.empty:
            table = table.drop(columns="Measure").rename(columns={"Mean of 2 backtests": cap(m0.label)})
        if len(metrics) > 1:
            table = table.pivot_table(index="Scheme", columns="Measure", values="Mean of 2 backtests", aggfunc="first").reset_index()
        return Result(facts, table, notes)

    def compare_schemes(self, a: str, b: str, metrics: list[Metric], asked: int = 0) -> Result:
        facts, rows = [], []
        for i, m in enumerate(metrics):
            va = self.bt[self.bt.SCHEME == a][m.column].mean()
            vb = self.bt[self.bt.SCHEME == b][m.column].mean()
            if pd.isna(va) or pd.isna(vb):
                continue
            verdict = ("Yes. " if (va > vb) == (asked > 0) else "No. ") if asked and i == 0 and va != vb else ""
            facts.append(f"{verdict}{SCHEME_NAMES[a]}: {fmt(m, va)} vs {SCHEME_NAMES[b]}: {fmt(m, vb)} on {m.label}, "
                         f"so {SCHEME_NAMES[a]} {m.high if va > vb else m.low}.")
            rows.append({"Measure": m.label, SCHEME_NAMES[a]: fmt(m, va), SCHEME_NAMES[b]: fmt(m, vb)})
        return Result(facts or ["Those schemes don't both have that measure."], pd.DataFrame(rows),
                      ["Each value is the mean of two out-of-time backtests."])

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
        return Result(facts, pd.DataFrame(rows),
                      ["Eligible means bootstrap stability ARI ≥ 0.85 and every segment holding ≥ 5% of customers."])

    def cleaning(self, step: str | None = None) -> Result:
        c = self.clean
        raw, final = int(c.ROWS_REMAINING.iloc[0]), int(c.ROWS_REMAINING.iloc[-1])
        rows = [{"Step": r.STAGE, "Rows left": f"{int(r.ROWS_REMAINING):,}",
                 "Rows removed": "" if pd.isna(r.CHANGE) else f"{int(-r.CHANGE):,}"} for r in c.itertuples()]
        if step:
            hit = c[c.STAGE.str.contains(step, case=False)]
            if not hit.empty:
                r = hit.iloc[0]
                removed = "" if pd.isna(r.CHANGE) else f" It removed {int(-r.CHANGE):,} rows."
                return Result([f"Step '{r.STAGE}': {int(r.ROWS_REMAINING):,} rows left.{removed}"], pd.DataFrame(rows))
        biggest = c.loc[c.CHANGE.idxmin()]
        facts = [f"Cleaning took the data from {raw:,} raw rows to {final:,} product lines with a customer ID, "
                 f"removing {raw - final:,} rows ({pct((raw - final) / raw)}). The biggest single step was "
                 f"'{biggest.STAGE}' ({int(-biggest.CHANGE):,} rows)."]
        return Result(facts, pd.DataFrame(rows))
