"""Power-run bank (v3): new data questions generated from templates, for the 80%-power comparison.

The held-out set (questions_v2.py) gave 175 paired data questions; detecting a 3.4-point gap with
80% power needs over a thousand. Hand-writing that many isn't practical, so this module generates
them: every value-tier, buying-style and segment reference, measure and wording below is combined
under a fixed seed, then sampled to the held-out set's mix of data-question types and wording styles.
Truths reuse the helpers in questions.py and are computed from the data, never typed in.

    build(n) -> list of items (same fields as questions.py), n data questions, all sets="both"

IDs start with X. Follow-ups use a generated question of another type as their parent.
"""
from __future__ import annotations

import itertools
import random
import re

from questions import ALL, LABEL, STYLES, TIERS, H, ans, each, fmt, name, pair, readings, seg_value

SEED = 20261016
# Held-out data mix (175 questions); build() scales it to n.
MIX = dict(lookup=32, total_share=22, ranking=20, comparison=16, breakdown=8, filter=6, model_eval=8,
           followup=28, ambiguous=19, unanswerable=16)
STYLE_MIX = dict(terse=0.30, casual=0.40, formal=0.20, typo=0.10)
PREFIX = dict(lookup="XLK", total_share="XTS", ranking="XRK", comparison="XCP", breakdown="XBD", filter="XFL",
              model_eval="XME", followup="XFU", ambiguous="XAM", unanswerable="XUN")

TIER_SHORT = {"C1": ["top spenders", "top spender"], "C2": ["core"], "C3": ["occasional"], "C4": ["low value", "low-value"]}
STYLE_SHORT = {"B1": ["bulk"], "B2": ["holiday"], "B3": ["small basket", "small-basket"]}
METRICS = ["cust", "share", "er", "rpc", "p", "uk", "rec", "ord", "nr", "upo", "price", "lpo", "ret", "q4", "aov", "ten"]
MLABEL = dict(LABEL, rpc="expected revenue per customer")


# ----------------------------------------------------------------------------- wording helpers
def seg_ref(code, style, rng):
    t, b = code.split("-")
    ts, bs = rng.choice(TIER_SHORT[t]), rng.choice(STYLE_SHORT[b])
    if style == "formal":
        return rng.choice([name(code), f"the {name(code)} segment", f"segment {code}"])
    if style == "terse":
        return rng.choice([code, code.lower(), f"{ts} {bs}"])
    return rng.choice([f"{ts} {bs} buyers", code.lower(), name(code).lower(), f"{ts} {bs} buyers"])


def tier_ref(t, style, rng):
    s = TIER_SHORT[t][0]
    if style == "formal":
        return rng.choice([f"the {TIERS[t]} tier", f"the {TIERS[t]} ({t}) tier"])
    if style == "terse":
        return rng.choice([s, f"{s} tier", t.lower()])
    return rng.choice([f"the {s} tier", s if t == "C1" else f"{s} customers"])


def style_ref(b, style, rng):
    s = STYLE_SHORT[b][0]
    if style == "formal":
        return rng.choice([STYLES[b], f"the {STYLES[b]} buying style", f"{STYLES[b]} ({b})"])
    if style == "terse":
        return rng.choice([f"{s} buyers", b.lower(), s])
    return rng.choice([f"{s} buyers", STYLES[b].lower()])


def group_ref(sel, style, rng):
    if sel in TIERS:
        return tier_ref(sel, style, rng)
    if sel in STYLES:
        return style_ref(sel, style, rng)
    return seg_ref(sel, style, rng)


def scope_word(sel, style, rng):
    """Adjective form for 'which ___ segment': '' for all, 'core', 'holiday buyers'."""
    if sel == "all":
        return ""
    if sel in TIERS:
        return (TIERS[sel] if style == "formal" else rng.choice(TIER_SHORT[sel])) + " "
    return (STYLES[sel] if style == "formal" else rng.choice([STYLE_SHORT[sel][0], STYLES[sel].lower()])) + " "


KEEP = {"c1", "c2", "c3", "c4", "b1", "b2", "b3", "uk", "aov", "q4"}


def typo(text, rng):
    """One misspelling in a word of five or more letters: a dropped, doubled or swapped letter."""
    words = [(m.start(), m.group()) for m in re.finditer(r"[A-Za-z]{5,}", text) if m.group().lower() not in KEEP]
    if not words:
        return text
    at, w = rng.choice(words)
    i = rng.randrange(1, len(w) - 1)
    kind = rng.choice(["drop", "drop", "swap", "double"])
    if kind == "drop":
        new = w[:i] + w[i + 1:]
    elif kind == "swap" and w[i] != w[i + 1]:
        new = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    else:
        new = w[:i] + w[i] + w[i:]
    return text[:at] + new + text[at + len(w):]


def styled(templates, style, rng, **refs):
    """Pick a template for the style (typo uses casual or terse wording, then misspells one word)."""
    base = rng.choice(["casual", "terse"]) if style == "typo" else style
    text = rng.choice(templates[base]).format(**refs)
    return typo(text, rng) if style == "typo" else text


def v(h, code, m):
    return h.v(code, "er") / h.v(code, "cust") if m == "rpc" else h.v(code, m)


def money(x):
    return f"£{x:,.2f}" if abs(x) < 100 else f"£{x:,.0f}"


# ----------------------------------------------------------------------------- lookups
LOOKUP = {
    "cust": dict(terse=["{s} customers", "{s} size", "customers {s}", "{s} headcount"],
                 casual=["how many customers are in {s}?", "how big is {s}?", "what's the customer count for {s}?"],
                 formal=["How many customers are in {S}?", "What is the customer count of {S}?"]),
    "share": dict(terse=["{s} share of customers", "% of customers in {s}"],
                  casual=["what share of all customers are {s}?", "what percent of our customers are in {s}?"],
                  formal=["What share of all customers belongs to {S}?", "What proportion of all customers is in {S}?"]),
    "er": dict(terse=["{s} expected revenue", "expected revenue {s}", "{s} predicted revenue"],
               casual=["what's the expected revenue for {s} next quarter?", "how much revenue do we expect from {s} next quarter?"],
               formal=["What is the expected revenue next quarter for {S}?", "What is the predicted next-quarter revenue of {S}?"]),
    "rpc": dict(terse=["{s} expected revenue per customer", "revenue per customer {s} next quarter"],
                casual=["what's each {s} customer expected to bring in next quarter?", "expected revenue per head for {s}?"],
                formal=["What is the expected revenue per customer next quarter for {S}?"]),
    "p": dict(terse=["{s} p buys again", "{s} chance of ordering", "p(buys again) {s}"],
              casual=["how likely are {s} to order again next quarter?", "what's the chance {s} buy again?"],
              formal=["What is the probability that {S} order again next quarter?", "What is the P(buys again) for {S}?"]),
    "uk": dict(terse=["{s} uk share", "uk share {s}"],
               casual=["what share of {s} are in the uk?", "what percent of {s} are uk-based?"],
               formal=["What is the UK share of {S}?", "What percentage of {S} are based in the UK?"]),
    "rec": dict(terse=["{s} recency", "days since last order {s}"],
                casual=["how long since {s} last ordered?", "how recently have {s} ordered?"],
                formal=["What is the median recency of {S}?", "How many days have passed since the last order, at the median, for {S}?"]),
    "ord": dict(terse=["{s} orders per year", "orders last 12 months {s}"],
                casual=["how often do {s} order?", "how many orders do {s} place a year?"],
                formal=["What is the median number of orders in the last 12 months for {S}?"]),
    "nr": dict(terse=["{s} net revenue", "{s} annual spend", "net revenue 12m {s}"],
               casual=["how much do {s} spend a year?", "what's the 12-month net revenue for {s}?"],
               formal=["What is the median net revenue over the last 12 months for {S}?"]),
    "upo": dict(terse=["{s} units per order", "basket size {s}"],
                casual=["how many units do {s} buy per order?", "what's the basket size for {s}?"],
                formal=["What is the median number of units per order for {S}?"]),
    "price": dict(terse=["{s} unit price", "unit price {s}"],
                  casual=["what do {s} pay per unit?", "what's the typical unit price for {s}?"],
                  formal=["What is the median unit price paid by {S}?"]),
    "lpo": dict(terse=["{s} lines per order", "product lines per order {s}"],
                casual=["how many product lines are in a typical order from {s}?", "how many lines per order for {s}?"],
                formal=["What is the median number of product lines per order for {S}?"]),
    "ret": dict(terse=["{s} return rate", "return rate {s}"],
                casual=["what's the return rate for {s}?", "how much of what {s} buy comes back as returns?"],
                formal=["What is the median return rate of {S}?"]),
    "q4": dict(terse=["{s} q4 share", "q4 revenue share {s}"],
               casual=["how much of their spend do {s} do in october to december?", "what's the q4 share for {s}?"],
               formal=["What share of revenue do {S} buy in October to December?", "What is the median Q4 revenue share of {S}?"]),
    "aov": dict(terse=["{s} aov", "{s} order value", "aov {s}"],
                casual=["what's the average order value for {s}?", "how big are orders from {s}, in pounds?"],
                formal=["What is the median order value of {S}?"]),
    "ten": dict(terse=["{s} tenure", "tenure {s}"],
                casual=["how long have {s} been customers?", "how long ago did {s} first order?"],
                formal=["What is the median tenure of {S}?", "How many days since the first order, at the median, for {S}?"]),
}


def lookup_truth(code, m):
    if m == "rpc":
        return lambda h: ans([v(h, code, m)], truth=f"{code} {name(code)}: expected revenue per customer {money(v(h, code, m))}")
    return seg_value(code, m)


def lookups():
    for code, m in itertools.product(ALL, METRICS):
        def make(style, rng, code=code, m=m):
            s = seg_ref(code, style if style != "typo" else "casual", rng)
            text = styled(LOOKUP[m], style, rng, s=s, S=seg_ref(code, "formal", rng))
            return text, lookup_truth(code, m), dict(kind="lookup", code=code, m=m)
        yield (code, m), make


# ----------------------------------------------------------------------------- totals and shares
GROUP_MEASURES = {
    "cust": (dict(terse=["{g} customers", "{g} headcount", "total customers {g}"],
                  casual=["how many customers are in {g}?", "how many {g} do we have?"],
                  formal=["How many customers are in {G}?"]),
             lambda h, g: h.tot("cust", g), lambda x: f"{x:,.0f}"),
    "er": (dict(terse=["{g} expected revenue", "total expected revenue {g}"],
                casual=["how much revenue do we expect from {g} next quarter?", "what's the total expected revenue for {g}?"],
                formal=["What is the total expected revenue next quarter for {G}?"]),
           lambda h, g: h.tot("er", g), lambda x: f"£{x:,.0f}"),
    "cust_share": (dict(terse=["{g} share of customers", "% of customers {g}"],
                        casual=["what share of all customers are {g}?", "what percent of our customers are {g}?"],
                        formal=["What percentage of all customers are in {G}?"]),
                   lambda h, g: h.tot("cust", g) / h.tot("cust"), lambda x: f"{x:.1%}"),
    "er_share": (dict(terse=["{g} share of expected revenue", "{g} % of predicted revenue"],
                      casual=["what share of next quarter's expected revenue comes from {g}?"],
                      formal=["What share of total expected revenue do {G} account for?"]),
                 lambda h, g: h.tot("er", g) / h.tot("er"), lambda x: f"{x:.1%}"),
    "orderers": (dict(terse=["expected orderers {g}", "{g} expected orderers"],
                      casual=["how many of {g} will order next quarter?", "how many {g} are expected to order next quarter?"],
                      formal=["How many customers in {G} are expected to order next quarter?"]),
                 lambda h, g: h.orderers(g), lambda x: f"~{x:,.0f}"),
    "rpc": (dict(terse=["{g} expected revenue per customer", "revenue per customer {g}"],
                 casual=["what's the expected revenue per customer for {g}?"],
                 formal=["What is the expected revenue per customer next quarter for {G}?"]),
            lambda h, g: h.tot("er", g) / h.tot("cust", g), lambda x: f"£{x:,.2f}"),
    "p_w": (dict(terse=["overall chance of buying again {g}", "{g} p buys again overall"],
                 casual=["what's the overall chance that {g} buy again?"],
                 formal=["What is the customer-weighted probability of buying again for {G}?"]),
            lambda h, g: h.w("p", g), lambda x: f"{x:.1%} customer-weighted"),
    "uk_w": (dict(terse=["{g} uk share overall", "overall uk share {g}"],
                  casual=["what share of {g} are in the uk overall?"],
                  formal=["What is the overall UK share of {G}?"]),
             lambda h, g: h.w("uk", g), lambda x: f"{x:.1%} customer-weighted"),
    "nonuk": (dict(terse=["non-uk customers {g}", "{g} customers outside uk"],
                   casual=["roughly how many of {g} are outside the uk?"],
                   formal=["Approximately how many customers in {G} are outside the UK?"]),
              lambda h, g: h.nonuk(g), lambda x: f"~{x:,.0f}"),
    "avg_seg_er": (dict(terse=["average expected revenue per {g} segment"],
                        casual=["what's the average expected revenue per segment in {g}?"],
                        formal=["What is the average expected revenue per segment in {G}?"]),
                   lambda h, g: h.tot("er", g) / len(h.g(g)), lambda x: f"£{x:,.0f}"),
}
PAIR_MEASURES = {
    "cust": (dict(terse=["{g1} + {g2} customers", "customers {g1} and {g2} combined"],
                  casual=["how many customers are in {g1} and {g2} combined?"],
                  formal=["How many customers are in {G1} and {G2} combined?"]),
             lambda h, gs: h.tot("cust", gs), lambda x: f"{x:,.0f}"),
    "er": (dict(terse=["expected revenue {g1} + {g2}", "{g1} and {g2} expected revenue combined"],
                casual=["what's the combined expected revenue of {g1} and {g2}?"],
                formal=["What is the combined expected revenue of {G1} and {G2}?"]),
           lambda h, gs: h.tot("er", gs), lambda x: f"£{x:,.0f}"),
}
EXCEPT_MEASURES = {
    "cust": (dict(terse=["customers excluding {g}", "customers outside {g}"],
                  casual=["how many customers are not in {g}?"],
                  formal=["How many customers are outside {G}?"]),
             lambda h, g: h.tot("cust") - h.tot("cust", g), lambda x: f"{x:,.0f}"),
    "er": (dict(terse=["expected revenue excluding {g}"],
                casual=["what's the expected revenue from everyone except {g}?"],
                formal=["What is the expected revenue of all customers outside {G}?"]),
           lambda h, g: h.tot("er") - h.tot("er", g), lambda x: f"£{x:,.0f}"),
}
WITHIN = dict(terse=["share of {outer} that are {inner}", "{inner} as % of {outer}"],
              casual=["what share of {outer} are {inner}?", "what percent of {outer} are {inner}?"],
              formal=["What proportion of {Outer} are {Inner}?"])


def totals():
    groups = list(TIERS) + list(STYLES)
    for g, mk in itertools.product(groups, GROUP_MEASURES):
        def make(style, rng, g=g, mk=mk):
            tpl, fn, show = GROUP_MEASURES[mk]
            base = "casual" if style == "typo" else style
            text = styled(tpl, style, rng, g=group_ref(g, base, rng), G=group_ref(g, "formal", rng))
            return text, (lambda h: ans([fn(h, g)], truth=show(fn(h, g)))), dict(kind="total", sel=g, mk=mk)
        yield ("grp", g, mk), make
    pairs = list(itertools.combinations(TIERS, 2)) + list(itertools.combinations(STYLES, 2))
    for (g1, g2), mk in itertools.product(pairs, PAIR_MEASURES):
        def make(style, rng, g1=g1, g2=g2, mk=mk):
            tpl, fn, show = PAIR_MEASURES[mk]
            base = "casual" if style == "typo" else style
            text = styled(tpl, style, rng, g1=group_ref(g1, base, rng), g2=group_ref(g2, base, rng),
                          G1=group_ref(g1, "formal", rng), G2=group_ref(g2, "formal", rng))
            return text, (lambda h: ans([fn(h, [g1, g2])], truth=show(fn(h, [g1, g2])))), dict(kind="total", sel=[g1, g2], mk=mk)
        yield ("pair", g1, g2, mk), make
    for g, mk in itertools.product(groups, EXCEPT_MEASURES):
        def make(style, rng, g=g, mk=mk):
            tpl, fn, show = EXCEPT_MEASURES[mk]
            base = "casual" if style == "typo" else style
            text = styled(tpl, style, rng, g=group_ref(g, base, rng), G=group_ref(g, "formal", rng))
            return text, (lambda h: ans([fn(h, g)], truth=show(fn(h, g)))), dict(kind="except", sel=g, mk=mk)
        yield ("except", g, mk), make
    for code, outer_is_tier in itertools.product(ALL, (True, False)):
        def make(style, rng, code=code, outer_is_tier=outer_is_tier):
            t, b = code.split("-")
            outer, inner = (t, b) if outer_is_tier else (b, t)
            base = "casual" if style == "typo" else style
            text = styled(WITHIN, style, rng, outer=group_ref(outer, base, rng), inner=group_ref(inner, base, rng),
                          Outer=group_ref(outer, "formal", rng), Inner=group_ref(inner, "formal", rng))

            def truth(h):
                x = h.v(code, "cust") / h.tot("cust", outer)
                return ans([x], truth=f"{x:.1%} ({h.v(code, 'cust'):,.0f} of {h.tot('cust', outer):,.0f})")
            return text, truth, dict(kind="within", code=code)
        yield ("within", code, outer_is_tier), make


# ----------------------------------------------------------------------------- rankings
# (verb form, high words, low words): "has" -> "has the X", "is" -> "is X", "verb" -> "X"
RANK_WORDS = {
    "cust": ("has", ["most customers"], ["fewest customers"]),
    "er": ("has", ["highest expected revenue", "most expected revenue"], ["lowest expected revenue"]),
    "rpc": ("has", ["highest expected revenue per customer"], ["lowest expected revenue per customer"]),
    "p": ("is", ["most likely to buy again", "most likely to order next quarter"], ["least likely to buy again"]),
    "uk": ("has", ["highest uk share"], ["lowest uk share"]),
    "rec": ("is", ["most lapsed"], ["most recent"]),
    "ord": ("verb", ["orders most often"], ["orders least often"]),
    "nr": ("verb", ["spends the most per year"], ["spends the least per year"]),
    "upo": ("has", ["biggest baskets in units", "most units per order"], ["fewest units per order"]),
    "price": ("verb", ["pays the most per unit"], ["pays the least per unit"]),
    "lpo": ("has", ["most lines per order"], ["fewest lines per order"]),
    "q4": ("is", ["most seasonal"], ["least seasonal"]),
    "aov": ("has", ["highest order value", "biggest orders by value"], ["lowest order value", "smallest orders by value"]),
    "ten": ("has", ["longest tenure"], ["shortest tenure"]),
}
PLAIN = dict(cust="customers", er="expected revenue", rpc="expected revenue per customer", p="p buys again",
             uk="uk share", rec="recency", ord="orders per year", nr="net revenue", upo="units per order",
             price="unit price", lpo="lines per order", q4="q4 share", aov="order value", ten="tenure")


def seg_rank(h, m, sel, asc, n):
    vals = sorted(((c, v(h, c, m)) for c in h.g(sel)), key=lambda x: x[1], reverse=not asc)
    cut = vals[n - 1][1]
    return [x for i, x in enumerate(vals) if i < n or abs(x[1] - cut) < 1e-12]


def rank_truth(m, sel, asc, n):
    def f(h):
        top = seg_rank(h, m, sel, asc, n)
        return ans([x for _, x in top], [c for c, _ in top],
                   truth=", ".join(f"{c} {name(c)} {money(x) if m == 'rpc' else fmt(m, x)}" for c, x in top))
    return f


def rank_ok(h, m, sel, asc, n):
    """No three-way ties, and a clear gap below the last one named (CSV and live round differently)."""
    vals = sorted((v(h, c, m) for c in h.g(sel)), reverse=not asc)
    top = seg_rank(h, m, sel, asc, n)
    if len(top) > n + 1 or len(top) >= len(vals):
        return False
    nxt = vals[len(top)]
    return abs(nxt - top[-1][1]) > 0.02 * max(abs(nxt), abs(top[-1][1]), 1e-9)


def rankings(h):
    scopes = ["all"] + list(TIERS) + list(STYLES)
    for m, sel, asc in itertools.product(RANK_WORDS, scopes, (False, True)):
        if not rank_ok(h, m, sel, asc, 1):
            continue

        def make(style, rng, m=m, sel=sel, asc=asc):
            form, hi, lo = RANK_WORDS[m]
            w = rng.choice(lo if asc else hi)
            sc = scope_word(sel, "formal" if style == "formal" else "casual", rng)
            pred = {"has": f"has the {w}", "is": f"is {w}", "verb": w}[form]
            tpl = dict(terse=[f"{w} {sc}segment", f"{sc}segment with the {w}" if form == "has" else f"{w} {sc}segment",
                              f"{'lowest' if asc else 'highest'} {PLAIN[m]} {sc}segment"],
                       casual=[f"which {sc}segment {pred}?", f"which {sc}segment {pred}?"],
                       formal=[f"Which {sc}segment {pred}?"])
            return styled(tpl, style, rng), rank_truth(m, sel, asc, 1), dict(kind="rank", m=m, sel=sel, asc=asc)
        yield ("rank", m, sel, asc), make
    for m, asc in itertools.product(RANK_WORDS, (False, True)):
        if not rank_ok(h, m, "all", asc, 3):
            continue

        def make(style, rng, m=m, asc=asc):
            word = "bottom" if asc else "top"
            tpl = dict(terse=[f"{word} 3 segments by {PLAIN[m]}", f"{word} three segments {PLAIN[m]}"],
                       casual=[f"what are the {word} 3 segments by {PLAIN[m]}?", f"which three segments have the {'lowest' if asc else 'highest'} {PLAIN[m]}?"],
                       formal=[f"List the three segments with the {'lowest' if asc else 'highest'} {MLABEL[m]}."])
            return styled(tpl, style, rng), rank_truth(m, "all", asc, 3), dict(kind="rank3", m=m, asc=asc)
        yield ("rank3", m, asc), make
    for level, mk, asc in itertools.product(("tier", "style"), ("cust", "er", "p_w", "rpc"), (False, True)):
        groups = list(TIERS) if level == "tier" else list(STYLES)

        def make(style, rng, level=level, mk=mk, asc=asc, groups=groups):
            fn = GROUP_MEASURES[mk][1]
            what = {"cust": ("most customers", "fewest customers"), "er": ("most expected revenue", "least expected revenue"),
                    "p_w": ("highest chance of buying again", "lowest chance of buying again"),
                    "rpc": ("highest expected revenue per customer", "lowest expected revenue per customer")}[mk][asc]
            unit = "value tier" if level == "tier" else "buying style"
            tpl = dict(terse=[f"{unit} with the {what}", f"{what} {unit}"],
                       casual=[f"which {unit} has the {what}?"], formal=[f"Which {unit} has the {what}?"])

            def truth(h):
                vals = sorted(((g, fn(h, g)) for g in groups), key=lambda x: x[1], reverse=not asc)
                g, x = vals[0]
                return ans([x], words=[TIERS.get(g, STYLES.get(g, g)).split()[0].lower()],
                           truth=f"{g} {TIERS.get(g) or STYLES.get(g)} {x:,.4g}")
            return styled(tpl, style, rng), truth, dict(kind="grank", level=level, mk=mk, asc=asc)
        yield ("grank", level, mk, asc), make


# ----------------------------------------------------------------------------- comparisons
YESNO = {
    "p": "are {a} more likely to buy again than {b}?", "rec": "have {a} ordered more recently than {b}?",
    "ten": "are {a} newer customers than {b}?", "cust": "is {a} bigger than {b}?",
    "aov": "do {a} place bigger orders by value than {b}?", "price": "do {a} pay more per unit than {b}?",
    "ord": "do {a} order more often than {b}?", "q4": "are {a} more seasonal than {b}?",
    "uk": "are {a} more uk-based than {b}?", "er": "do {a} have more expected revenue than {b}?",
    "nr": "do {a} spend more per year than {b}?", "upo": "do {a} buy more units per order than {b}?",
}
LOWER_IS_YES = {"rec", "ten"}  # "more recently" and "newer" mean lower values


def comparisons(h):
    pairs = [(a, b) for a, b in itertools.permutations(ALL, 2) if a[:2] == b[:2] or a[3:] == b[3:]]
    for (a, b), m in itertools.product(pairs, ["cust", "er", "rpc", "p", "uk", "rec", "ord", "nr", "upo", "price", "lpo", "q4", "aov", "ten"]):
        va, vb = v(h, a, m), v(h, b, m)
        if abs(va - vb) <= 0.02 * max(abs(va), abs(vb), 1e-9):
            continue
        if a > b and m not in YESNO:  # one order per pair for the plain forms
            continue

        def make(style, rng, a=a, b=b, m=m):
            ra, rb = seg_ref(a, "casual", rng), seg_ref(b, "casual", rng)
            fa, fb = seg_ref(a, "formal", rng), seg_ref(b, "formal", rng)
            if m in YESNO and style in ("casual", "typo", "formal") and rng.random() < 0.6:
                q_ = YESNO[m].format(a=name(a) if style == "formal" else ra, b=name(b) if style == "formal" else rb)
                q_ = q_[0].upper() + q_[1:] if style == "formal" else q_

                def truth(h):
                    xa, xb = v(h, a, m), v(h, b, m)
                    word = "yes" if (xa < xb if m in LOWER_IS_YES else xa > xb) else "no"
                    return ans([xa, xb], words=[word], truth=f"{a} {fmt(m, xa)} vs {b} {fmt(m, xb)} -> {word}")
                return (typo(q_, rng) if style == "typo" else q_), truth, dict(kind="cmp", a=a, b=b, m=m)
            tpl = dict(terse=[f"{a.lower()} vs {b.lower()} {PLAIN[m]}", f"{ra} vs {rb} {PLAIN[m]}"],
                       casual=[f"how do {ra} and {rb} compare on {PLAIN[m]}?"],
                       formal=[f"Compare {fa} and {fb} on {MLABEL[m]}."])

            def truth(h):
                xa, xb = v(h, a, m), v(h, b, m)
                show = money if m == "rpc" else (lambda x: fmt(m, x))
                return ans([xa, xb], truth=f"{a} {show(xa)} vs {b} {show(xb)}")
            return styled(tpl, style, rng), truth, dict(kind="cmp", a=a, b=b, m=m)
        yield ("cmp", a, b, m), make
    gpairs = list(itertools.combinations(TIERS, 2)) + list(itertools.combinations(STYLES, 2))
    for (g1, g2), mk in itertools.product(gpairs, ("cust", "er", "p_w", "rpc")):
        def make(style, rng, g1=g1, g2=g2, mk=mk):
            fn = GROUP_MEASURES[mk][1]
            what = {"cust": "customers", "er": "expected revenue", "p_w": "chance of buying again",
                    "rpc": "expected revenue per customer"}[mk]
            tpl = dict(terse=["{g1} vs {g2} " + what], casual=["how do {g1} and {g2} compare on " + what + "?"],
                       formal=["Compare {G1} and {G2} on " + what + "."])
            base = "casual" if style == "typo" else style
            text = styled(tpl, style, rng, g1=group_ref(g1, base, rng), g2=group_ref(g2, base, rng),
                          G1=group_ref(g1, "formal", rng), G2=group_ref(g2, "formal", rng))
            return text, (lambda h: ans([fn(h, g1), fn(h, g2)], truth=f"{g1} {fn(h, g1):,.4g} vs {g2} {fn(h, g2):,.4g}")), \
                dict(kind="gcmp", g1=g1, g2=g2, mk=mk)
        yield ("gcmp", g1, g2, mk), make


# ----------------------------------------------------------------------------- breakdowns
def breakdowns():
    for m in [x for x in METRICS if x not in ("rpc",)]:
        def make(style, rng, m=m):
            tpl = dict(terse=[f"{PLAIN.get(m, MLABEL[m])} by segment", f"{PLAIN.get(m, MLABEL[m])} for every segment"],
                       casual=[f"show me the {PLAIN.get(m, MLABEL[m])} for each segment", f"list {PLAIN.get(m, MLABEL[m])} by segment"],
                       formal=[f"Provide the {MLABEL[m]} for each of the 12 segments."])
            return styled(tpl, style, rng), each(m), dict(kind="each", m=m, sel="all")
        yield ("each", m), make
    for sel, m in itertools.product(list(TIERS) + list(STYLES), ["cust", "er", "p", "rec", "ord", "nr", "aov", "price", "upo", "ten", "uk"]):
        def make(style, rng, sel=sel, m=m):
            by = "buying style" if sel in TIERS else "value tier"
            base = "casual" if style == "typo" else style
            g, G = group_ref(sel, base, rng), group_ref(sel, "formal", rng)
            tpl = dict(terse=[f"{{g}} {PLAIN.get(m, MLABEL[m])} by {by}", f"{PLAIN.get(m, MLABEL[m])} for each {{g}} segment"],
                       casual=[f"break down {{g}} by {by} on {PLAIN.get(m, MLABEL[m])}", f"what's the {PLAIN.get(m, MLABEL[m])} for each {{g}} segment?"],
                       formal=[f"Break down {{G}} by {by}, showing {MLABEL[m]}."])
            return styled(tpl, style, rng, g=g, G=G), each(m, sel, 1.0), dict(kind="each", m=m, sel=sel)
        yield ("each", sel, m), make
    for level, mk in itertools.product(("tier", "style"), ("cust", "er", "orderers", "p_w", "uk_w", "rpc", "er_share", "cust_share")):
        groups = list(TIERS) if level == "tier" else list(STYLES)

        def make(style, rng, level=level, mk=mk, groups=groups):
            fn = GROUP_MEASURES[mk][1]
            what = {"cust": "customers", "er": "expected revenue", "orderers": "expected orderers",
                    "p_w": "chance of buying again", "uk_w": "uk share", "rpc": "expected revenue per customer",
                    "er_share": "share of expected revenue", "cust_share": "share of customers"}[mk]
            unit = "value tier" if level == "tier" else "buying style"
            tpl = dict(terse=[f"{what} by {unit}", f"{what} per {unit}"],
                       casual=[f"show me {what} for each {unit}", f"how does {what} break down by {unit}?"],
                       formal=[f"Provide a breakdown of {what} by {unit}."])
            return styled(tpl, style, rng), (lambda h: ans([fn(h, g) for g in groups],
                                                          truth=", ".join(f"{g} {fn(h, g):,.4g}" for g in groups))), \
                dict(kind="gbd", level=level, mk=mk)
        yield ("gbd", level, mk), make


# ----------------------------------------------------------------------------- filters
UNIT = dict(cust="customers", er="expected revenue", p="chance of buying again", uk="uk share", rec="median recency",
            ord="orders a year", nr="median net revenue", upo="units per order", price="unit price",
            lpo="lines per order", q4="q4 revenue share", aov="order value", ten="median tenure")
NICE = dict(cust=50, er=10_000, p=0.05, uk=0.01, rec=10, ord=1, nr=250, upo=25, price=0.1, lpo=2, q4=0.05, aov=50, ten=50)


def show_threshold(m, t):
    if m in ("er", "nr", "aov"):
        return f"£{t:,.0f}"
    if m == "price":
        return f"£{t:.2f}"
    if m in ("p", "uk", "q4"):
        return f"{t:.0%}"
    if m in ("rec", "ten"):
        return f"{t:g} days"
    return f"{t:g}"


def thresholds(h, m):
    vals = sorted({v(h, c, m) for c in ALL})
    out = []
    for lo, hi in zip(vals, vals[1:]):
        step = NICE[m]
        t = round(((lo + hi) / 2) / step) * step
        if lo + 0.03 * abs(t) < t < hi - 0.03 * abs(t) and t > 0:
            out.append(t)
    return sorted(set(out))


def filters(h):
    for m in UNIT:
        for t, above in itertools.product(thresholds(h, m), (True, False)):
            cs = [c for c in ALL if (v(h, c, m) > t if above else v(h, c, m) < t)]
            if not 1 <= len(cs) <= 9:
                continue
            for form in ("which", "count", "cust", "er"):
                if form == "cust" and m == "cust" or form == "er" and m == "er":
                    continue

                def make(style, rng, m=m, t=t, above=above, form=form):
                    word = rng.choice(["above", "over", "more than"]) if above else rng.choice(["below", "under", "less than"])
                    cond = f"{UNIT[m]} {word} {show_threshold(m, t)}"
                    tpl = {"which": dict(terse=[f"segments with {cond}"], casual=[f"which segments have {cond}?"],
                                         formal=[f"Which segments have {cond}?"]),
                           "count": dict(terse=[f"number of segments with {cond}"], casual=[f"how many segments have {cond}?"],
                                         formal=[f"How many segments have {cond}?"]),
                           "cust": dict(terse=[f"customers in segments with {cond}"],
                                        casual=[f"how many customers are in segments with {cond}?"],
                                        formal=[f"How many customers are in segments with {cond}?"]),
                           "er": dict(terse=[f"expected revenue of segments with {cond}"],
                                      casual=[f"how much expected revenue sits in segments with {cond}?"],
                                      formal=[f"What is the total expected revenue of segments with {cond}?"])}[form]

                    def truth(h):
                        cs = [c for c in ALL if (v(h, c, m) > t if above else v(h, c, m) < t)]
                        if form == "which":
                            return ans([], cs, truth=", ".join(f"{c} {fmt(m, v(h, c, m))}" for c in cs))
                        if form == "count":
                            return ans([len(cs)], cs, truth=f"{len(cs)}: {', '.join(cs)}")
                        rep = "cust" if form == "cust" else "er"
                        tot = sum(h.v(c, rep) for c in cs)
                        return ans([tot], cs, truth=f"{len(cs)} segments ({', '.join(cs)}); {fmt(rep, tot)}")
                    return styled(tpl, style, rng), truth, dict(kind="filter", m=m, t=t, above=above, form=form)
                yield ("filter", m, t, above, form), make


# ----------------------------------------------------------------------------- model evaluation
SCHEMES = {
    "hybrid": (["hybrid", "the hybrid"], "the hybrid scheme"),
    "rfm_score_baseline": (["rfm", "the rfm baseline", "classic rfm"], "the RFM baseline"),
    "two_axis_grid": (["the two-axis grid", "the grid"], "the two-axis grid"),
    "two_axis_value": (["the value-axis-only scheme"], "the value-axis-only scheme"),
    "outcome_guided": (["the outcome tiers (3-4 segments)"], "the outcome-guided tiers (3-4 segments)"),
    "outcome_guided_grid_size": (["the outcome tiers at grid size (9-12 segments)"], "the outcome-guided tiers at grid size (9-12 segments)"),
    "outcome_model_continuous": (["the unsegmented outcome model"], "the unsegmented outcome model"),
}
EVAL_COLS = {
    "NEXT_ACTIVE_AUC": ("buys-again auc", "buys-again AUC", lambda x: f"{x:.4f}"),
    "TOP20_REVENUE_CAPTURE": ("top-20% revenue capture", "top-20% revenue capture", lambda x: f"{x:.1%}"),
    "REVENUE_SPEARMAN": ("revenue spearman", "revenue Spearman correlation", lambda x: f"{x:.4f}"),
    "RULE_FIDELITY": ("rule fidelity", "rule fidelity", lambda x: f"{x:.1%}"),
    "VALUE_ETA2": ("value eta squared", "value separation (eta squared)", lambda x: f"{x:.4f}"),
    "BEHAVIOR_ETA2": ("buying-style eta squared", "buying-style separation (eta squared)", lambda x: f"{x:.4f}"),
    "PERSISTENCE": ("persistence", "persistence", lambda x: f"{x:.1%}"),
    "REVENUE_FORECAST_BIAS": ("revenue forecast bias", "revenue forecast bias", lambda x: f"{x:.1%}"),
}


def model_evals(h):
    import math
    for s, col, fold in itertools.product(SCHEMES, EVAL_COLS, (None, "backtest_q2", "backtest_q3")):
        x = h.bt_mean(s, col) if fold is None else h.bt_fold(s, col, fold)
        if x is None or (isinstance(x, float) and math.isnan(x)):
            continue

        def make(style, rng, s=s, col=col, fold=fold):
            short, formal = rng.choice(SCHEMES[s][0]), SCHEMES[s][1]
            lab, flab, show = EVAL_COLS[col]
            when = {None: "", "backtest_q2": " in the q2 backtest", "backtest_q3": " in the q3 backtest"}[fold]
            tpl = dict(terse=[f"{short} {lab}{' q2' if fold == 'backtest_q2' else ' q3' if fold == 'backtest_q3' else ''}"],
                       casual=[f"what's the {lab} for {short}{when}?", f"what {lab} did {short} get{when}?"],
                       formal=[f"What was the {flab} of {formal}{when or ', averaged over the two backtests'}?"])

            def truth(h):
                y = h.bt_mean(s, col) if fold is None else h.bt_fold(s, col, fold)
                return ans([y], truth=show(y) + (" (mean of 2 folds)" if fold is None else ""))
            return styled(tpl, style, rng), truth, dict(kind="eval", s=s, col=col, fold=fold)
        yield ("eval", s, col, fold), make
    for axis, k in itertools.product(("value", "behavior"), range(3, 8)):
        for col, lab in (("STABILITY_ARI", "stability ari"), ("SILHOUETTE", "silhouette"), ("NEXT_REVENUE_R2", "next-revenue r2"),
                         ("MIN_SHARE", "smallest segment share")):
            def make(style, rng, axis=axis, k=k, col=col, lab=lab):
                ax = "behaviour" if axis == "behavior" else "value"
                tpl = dict(terse=[f"{ax} axis k={k} {lab}"], casual=[f"what {lab} did k={k} get on the {ax} axis?"],
                           formal=[f"What was the {lab} for k = {k} on the {ax} axis?"])

                def truth(h):
                    y = float(h.krow(axis, k)[col])
                    return ans([y], truth=f"{axis} k={k}: {col} {y:.4f}")
                return styled(tpl, style, rng), truth, dict(kind="k", axis=axis, k=k, col=col)
            yield ("k", axis, k, col), make
    steps = [("sheet overlap", "rows removed by dropping the sheet overlap", "how many rows did dropping the sheet overlap remove?", "CHANGE"),
             ("exact duplicates", "rows removed as exact duplicates", "how many exact duplicate rows were removed?", "CHANGE"),
             ("write-offs", "rows removed as write-offs", "how many write-off rows were taken out?", "CHANGE"),
             ("customer id", "rows dropped for having no customer id", "how many rows had no customer id?", "CHANGE"),
             ("raw rows", "raw row count", "how many raw rows were there before cleaning?", "ROWS_REMAINING"),
             ("sheet overlap", "rows left after removing the sheet overlap", "how many rows were left after the sheet overlap was removed?", "ROWS_REMAINING"),
             ("exact duplicates", "rows left after removing exact duplicates", "how many rows remained once exact duplicates were gone?", "ROWS_REMAINING")]
    for word, lab, casual, col in steps:
        def make(style, rng, word=word, lab=lab, casual=casual, col=col):
            tpl = dict(terse=[lab], casual=[casual], formal=[f"What was the number of {lab} in the data cleaning?"])

            def truth(h):
                y = abs(h.clean(word, col))
                return ans([y], truth=f"{y:,.0f}")
            return styled(tpl, style, rng), truth, dict(kind="clean", word=word, col=col)
        yield ("clean", word, col, lab), make
    for col in ("NEXT_ACTIVE_AUC", "TOP20_REVENUE_CAPTURE", "BEHAVIOR_ETA2", "VALUE_ETA2"):
        def make(style, rng, col=col):
            lab, flab, show = EVAL_COLS[col]
            tpl = dict(terse=[f"best segmentation scheme by {lab}, not counting the unsegmented model"],
                       casual=[f"leaving out the unsegmented model, which scheme has the best {lab}?"],
                       formal=[f"Excluding the unsegmented outcome model, which segmentation scheme has the highest mean {flab}?"])

            def truth(h):
                best = max((s for s in SCHEMES if s != "outcome_model_continuous"), key=lambda s: h.bt_mean(s, col))
                return ans([h.bt_mean(best, col)], words=[SCHEMES[best][0][0].replace("the ", "").split()[0]],
                           truth=f"{best} {show(h.bt_mean(best, col))}")
            return styled(tpl, style, rng), truth, dict(kind="best", col=col)
        yield ("best", col), make


# ----------------------------------------------------------------------------- follow-ups (built on a parent's meta)
FU_STYLES = dict(terse=0.55, casual=0.40, typo=0.05)


def followup_for(meta, rng):
    """(text, truth) for a follow-up to a parent with this meta, or None."""
    k = meta["kind"]
    if k == "lookup":
        code, m = meta["code"], meta["m"]
        others = [c for c in ALL if c != code and (c[:2] == code[:2] or c[3:] == code[3:])]
        choice = rng.random()
        if choice < 0.45:
            o = rng.choice(others)
            return rng.choice([f"and {seg_ref(o, 'casual', rng)}?", f"what about {seg_ref(o, 'terse', rng)}?"]), lookup_truth(o, m)
        m2 = rng.choice([x for x in ("cust", "er", "p", "rec", "ord", "aov", "price", "ten", "nr", "uk") if x != m])
        return rng.choice([f"and their {PLAIN[m2]}?", f"what about {PLAIN[m2]}?"]), lookup_truth(code, m2)
    if k == "rank":
        m, sel, asc = meta["m"], meta["sel"], meta["asc"]
        r = rng.random()
        if r < 0.4 and rank_ok_cache.get((m, sel, not asc)):
            return rng.choice(["and the lowest?" if not asc else "and the highest?",
                               "what about the opposite end?"]), rank_truth(m, sel, not asc, 1)
        m2 = rng.choice(["cust", "er", "p"])
        if m2 == m:
            m2 = "aov"

        def truth(h, m=m, sel=sel, asc=asc, m2=m2):
            top = seg_rank(h, m, sel, asc, 1)
            return ans([h.v(c, m2) for c, _ in top], [c for c, _ in top],
                       truth="; ".join(f"{c} {fmt(m2, h.v(c, m2))}" for c, _ in top))
        q_ = {"cust": ["how many customers does it have?", "how big is that segment?"],
              "er": ["what's its expected revenue?", "how much revenue do we expect from it?"],
              "p": ["how likely is it to buy again?", "and its chance of ordering next quarter?"],
              "aov": ["what's its order value?"]}[m2]
        return rng.choice(q_), truth
    if k == "total" and isinstance(meta["sel"], str) and meta["mk"] in ("cust", "er"):
        g, mk = meta["sel"], meta["mk"]
        r = rng.random()
        if r < 0.4:
            return rng.choice(["% of total", "what share of the total is that?", "as a percentage of everyone?"]), \
                (lambda h: ans([h.tot(mk, g) / h.tot(mk)], truth=f"{h.tot(mk, g) / h.tot(mk):.1%}"))
        if mk == "er" and r < 0.7:
            return rng.choice(["per customer?", "and per head?"]), \
                (lambda h: ans([h.tot("er", g) / h.tot("cust", g)], truth=f"£{h.tot('er', g) / h.tot('cust', g):,.2f}"))
        others = [x for x in (TIERS if g in TIERS else STYLES) if x != g]
        o = rng.choice(others)
        return rng.choice([f"and {group_ref(o, 'casual', rng)}?", f"what about {group_ref(o, 'terse', rng)}?"]), \
            (lambda h: ans([h.tot(mk, o)], truth=f"{h.tot(mk, o):,.0f}"))
    if k == "cmp":
        a, b, m = meta["a"], meta["b"], meta["m"]
        if rng.random() < 0.5 and m != "rpc":
            return rng.choice(["what's the difference?", "what's the gap between them?"]), \
                (lambda h: ans([abs(v(h, a, m) - v(h, b, m))], truth=f"{abs(v(h, a, m) - v(h, b, m)):,.4g}"))
        m2 = rng.choice([x for x in ("cust", "er", "p", "aov", "ord", "rec") if x != m])
        return rng.choice([f"and on {PLAIN[m2]}?", f"what about their {PLAIN[m2]}?"]), pair(a, b, m2)
    if k == "gbd":
        mk, level = meta["mk"], meta["level"]
        if mk in ("cust", "er", "orderers", "p_w", "rpc"):
            groups = list(STYLES) if level == "tier" else list(TIERS)
            fn = GROUP_MEASURES[mk][1]
            return rng.choice([f"and by {'buying style' if level == 'tier' else 'value tier'}?", "same by the other axis"]), \
                (lambda h: ans([fn(h, g) for g in groups], truth=", ".join(f"{g} {fn(h, g):,.4g}" for g in groups)))
    if k == "filter" and meta["form"] in ("which", "count"):
        m, t, above = meta["m"], meta["t"], meta["above"]
        rep = rng.choice(["cust", "er"]) if m not in ("cust", "er") else ("er" if m == "cust" else "cust")

        def truth(h):
            cs = [c for c in ALL if (v(h, c, m) > t if above else v(h, c, m) < t)]
            tot = sum(h.v(c, rep) for c in cs)
            return ans([tot], truth=f"{fmt(rep, tot)} ({', '.join(cs)})")
        return ({"cust": ["how many customers is that?", "how many customers do they have together?"],
                 "er": ["what's their combined expected revenue?", "how much expected revenue do they hold?"]}[rep][rng.randrange(2)], truth)
    if k == "eval" and meta["fold"] is not None:
        s, col, fold = meta["s"], meta["col"], meta["fold"]
        other = "backtest_q3" if fold == "backtest_q2" else "backtest_q2"
        return rng.choice([f"and {other[-2:]}?", f"what about the {other[-2:]} backtest?"]), \
            (lambda h: ans([h.bt_fold(s, col, other)], truth=EVAL_COLS[col][2](h.bt_fold(s, col, other))))
    if k == "eval":
        s, col = meta["s"], meta["col"]
        o = rng.choice([x for x in SCHEMES if x != s])
        return rng.choice([f"and {SCHEMES[o][0][0]}?", f"what about {SCHEMES[o][0][0]}?"]), \
            (lambda h: ans([h.bt_mean(o, col)], truth=EVAL_COLS[col][2](h.bt_mean(o, col)) + " (mean of 2 folds)"))
    return None


rank_ok_cache: dict = {}


# ----------------------------------------------------------------------------- ambiguous
VAGUE = {  # adjective -> readings builder
    ("good", "valuable", "high-value", "vip", "premium", "key"): lambda h: [
        ("Top spenders tier", [h.tot("cust", "C1")], []),
        ("segments with P(buys again) above 50%", [sum(h.v(c, "cust") for c in ALL if h.v(c, "p") > 0.5)], []),
        ("ask what counts as good", [], [])],
    ("loyal", "regular", "repeat"): lambda h: [
        ("segments with P(buys again) above 50%", [sum(h.v(c, "cust") for c in ALL if h.v(c, "p") > 0.5)], []),
        ("Top spenders tier", [h.tot("cust", "C1")], []), ("ask what counts as loyal", [], [])],
    ("active", "engaged"): lambda h: [
        ("expected orderers next quarter", [h.orderers()], []),
        ("segments ordering within 30 days", [sum(h.v(c, "cust") for c in ALL if h.v(c, "rec") < 30)], []),
        ("ask what counts as active", [], [])],
    ("at-risk", "lapsed", "dormant", "churning"): lambda h: [
        ("segments with P(buys again) under 25%", [sum(h.v(c, "cust") for c in ALL if h.v(c, "p") < 0.25)], []),
        ("segments more than 90 days since last order", [sum(h.v(c, "cust") for c in ALL if h.v(c, "rec") > 90)], []),
        ("ask what counts as at risk", [], [])],
    ("cheap", "bargain", "price-sensitive"): lambda h: [
        ("Bulk buyers (lowest unit prices)", [h.tot("cust", "B1")], []), ("Low value tier", [h.tot("cust", "C4")], []),
        ("ask what cheap means", [], [])],
    ("big", "large"): lambda h: [
        ("Bulk buyers", [h.tot("cust", "B1")], []), ("Top spenders tier", [h.tot("cust", "C1")], []), ("ask", [], [])],
    ("small",): lambda h: [
        ("Small-basket buyers", [h.tot("cust", "B3")], []), ("Low value tier", [h.tot("cust", "C4")], []), ("ask", [], [])],
    ("new", "recent"): lambda h: [
        ("segments with median tenure under 400 days", [sum(h.v(c, "cust") for c in ALL if h.v(c, "ten") < 400)], []),
        ("ask what counts as new", [], [])],
    ("frequent",): lambda h: [
        ("segments ordering 10+ times a year", [sum(h.v(c, "cust") for c in ALL if h.v(c, "ord") >= 10)], []),
        ("ask what counts as frequent", [], [])],
    ("seasonal", "christmas"): lambda h: [("Holiday buyers", [h.tot("cust", "B2")], []), ("ask", [], [])],
}
VALUE_READINGS = lambda h, c: [("expected revenue next quarter", [h.v(c, "er")], []),  # noqa: E731
                               ("median net revenue 12m", [h.v(c, "nr")], []),
                               ("expected revenue per customer", [h.v(c, "er") / h.v(c, "cust")], [])]


def ambiguous(h):
    for c, form in itertools.product(ALL, ("revenue", "worth", "valuable", "spend")):
        def make(style, rng, c=c, form=form):
            s = seg_ref(c, "casual" if style in ("typo", "casual") else style, rng)
            text = {"revenue": rng.choice([f"{s} revenue", f"revenue for {s}"]),
                    "worth": rng.choice([f"what is {s} worth?", f"how much are {s} worth?"]),
                    "valuable": rng.choice([f"how valuable are {s}?", f"how valuable is {s}?"]),
                    "spend": rng.choice([f"spend {s}", f"how much do {s} spend?"])}[form]
            text = (text[0].upper() + text[1:]) if style == "formal" else text
            text = typo(text, rng) if style == "typo" else text
            if form == "spend":
                return text, (lambda h: readings(("median net revenue 12m", [h.v(c, "nr")], []),
                                                 ("median order value", [h.v(c, "aov")], []),
                                                 ("expected revenue", [h.v(c, "er")], []))), dict(kind="amb")
            return text, (lambda h: readings(*VALUE_READINGS(h, c))), dict(kind="amb")
        yield ("amb_seg", c, form), make
    for sel, word in itertools.product(["all"] + list(TIERS) + list(STYLES), ("largest", "best", "worst")):
        if word == "largest":
            a, b = seg_rank(h, "cust", sel, False, 1)[0][0], seg_rank(h, "er", sel, False, 1)[0][0]
            if a == b:
                continue
        def make(style, rng, sel=sel, word=word):
            sc = scope_word(sel, "formal" if style == "formal" else "casual", rng)
            text = {"largest": rng.choice([f"which {sc}segment is the largest?", f"biggest {sc}segment"]),
                    "best": rng.choice([f"which {sc}segment performs best?", f"best {sc}segment"]),
                    "worst": rng.choice([f"which {sc}segment is the worst?", f"weakest {sc}segment"])}[word]
            text = (text[0].upper() + text[1:]) if style == "formal" else text
            text = typo(text, rng) if style == "typo" else text

            def truth(h):
                if word == "largest":
                    r = [("by customers", "cust", False), ("by expected revenue", "er", False)]
                elif word == "best":
                    r = [("expected revenue", "er", False), ("P(buys again)", "p", False), ("expected revenue per customer", "rpc", False)]
                else:
                    r = [("lowest expected revenue", "er", True), ("lowest P(buys again)", "p", True), ("most lapsed", "rec", False)]
                return readings(*[(lab, [x for _, x in seg_rank(h, m, sel, asc, 1)], [c for c, _ in seg_rank(h, m, sel, asc, 1)])
                                  for lab, m, asc in r])
            return text, truth, dict(kind="amb")
        yield ("amb_rank", sel, word), make
    for adjs, fn in VAGUE.items():
        for adj in adjs:
            def make(style, rng, adj=adj, fn=fn):
                text = rng.choice({"terse": [f"{adj} customers count", f"number of {adj} customers"],
                                   "casual": [f"how many {adj} customers do we have?", f"how many {adj} buyers are there?"],
                                   "formal": [f"How many {adj} customers are there?"]}["casual" if style == "typo" else style])
                text = typo(text, rng) if style == "typo" else text
                return text, (lambda h: readings(*fn(h))), dict(kind="amb")
            yield ("amb_vague", adj), make
    for g in list(TIERS) + list(STYLES):
        for form in ("engaged", "retention", "price", "size"):
            if form == "size" and g in STYLES:
                continue

            def make(style, rng, g=g, form=form):
                r = group_ref(g, "casual" if style in ("typo", "casual") else style, rng)
                text = {"engaged": f"how engaged are {r}?", "retention": rng.choice([f"retention for {r}", f"how well do we retain {r}?"]),
                        "price": f"what's the price for {r}?", "size": f"what is the size of {r}?"}[form]
                text = (text[0].upper() + text[1:]) if style == "formal" else text
                text = typo(text, rng) if style == "typo" else text

                def truth(h):
                    cs = h.g(g)
                    if form == "engaged":
                        return readings(("P(buys again), customer-weighted", [h.w("p", g)], []),
                                        ("median recency per segment", [h.v(c, "rec") for c in cs], cs),
                                        ("median orders per segment", [h.v(c, "ord") for c in cs], cs),
                                        ("expected orderers", [h.orderers(g)], []))
                    if form == "retention":
                        return readings(("P(buys again), customer-weighted", [h.w("p", g)], []),
                                        ("P(buys again) per segment", [h.v(c, "p") for c in cs], cs))
                    if form == "price":
                        return readings(("median unit price per segment", [h.v(c, "price") for c in cs], cs),
                                        ("median order value per segment", [h.v(c, "aov") for c in cs], cs))
                    return readings(("customers", [h.tot("cust", g)], []), ("share of customers", [h.tot("cust", g) / h.tot("cust")], []),
                                    ("expected revenue", [h.tot("er", g)], []))
                return text, truth, dict(kind="amb")
            yield ("amb_group", g, form), make
    for s in SCHEMES:
        for form in ("good", "hit"):
            def make(style, rng, s=s, form=form):
                short = rng.choice(SCHEMES[s][0]) if style != "formal" else SCHEMES[s][1]
                text = {"good": f"how good is {short}?", "hit": f"what's the hit rate for {short}?"}[form]
                text = (text[0].upper() + text[1:]) if style == "formal" else text
                text = typo(text, rng) if style == "typo" else text
                return text, (lambda h: readings(("buys-again AUC", [h.bt_mean(s, "NEXT_ACTIVE_AUC")], []),
                                                 ("top-20 capture", [h.bt_mean(s, "TOP20_REVENUE_CAPTURE")], []),
                                                 ("ask", [], []))), dict(kind="amb")
            yield ("amb_scheme", s, form), make


# ----------------------------------------------------------------------------- unanswerable
NOT_HELD = [
    ("average age", "No age field."), ("gender split", "No gender field."), ("income level", "No income data."),
    ("favourite product category", "No product data."), ("best-selling products", "No product data."),
    ("gross margin", "No cost or margin data."), ("profit", "No cost or margin data."),
    ("average discount", "No discount data."), ("marketing channel", "No channel data."),
    ("acquisition source", "No acquisition data."), ("email open rate", "No campaign data."),
    ("campaign response rate", "No campaign data."), ("nps", "No survey data."),
    ("satisfaction score", "No survey data."), ("phone numbers", "No contact data or customer rows."),
    ("email addresses", "No contact data or customer rows."), ("customer names", "No customer names or rows."),
    ("postcodes", "No address data."), ("average delivery time", "No delivery data."),
    ("preferred payment method", "No payment data."), ("website visits", "No web data."),
    ("lifetime value", "No lifetime value; only next-quarter expected revenue."),
    ("credit limit", "No credit data."), ("industry", "No firmographic data."), ("company size", "No firmographic data."),
]
ASK_FORMS = dict(terse=["{g} {a}", "{a} {g}"], casual=["what's the {a} for {g}?", "can you tell me the {a} of {g}?"],
                 formal=["What is the {a} of {G}?"])
EVENTS = [
    ("how many {g} churned last month?", "No monthly churn history."),
    ("how many {g} moved up a tier last quarter?", "Migration isn't in these tables."),
    ("what did {g} spend in march 2011?", "No revenue by past month."),
    ("weekly orders for {g} in 2010", "No orders by week."),
    ("what will {g} spend over the next two years?", "Forecasts cover next quarter only; giving next quarter with that caveat is fine."),
    ("which sales rep looks after {g}?", "No account-owner data."),
    ("how many {g} came from the website?", "No acquisition channel data."),
    ("how many {g} opened last month's newsletter?", "No campaign data."),
]
COUNTRIES = ["belgium", "ireland", "iceland", "israel", "lebanon", "singapore", "bahrain", "malta", "lithuania",
             "the netherlands", "germany", "spain", "australia", "france", "the usa", "canada"]


def unanswerables():
    groups = ALL + list(TIERS) + list(STYLES)
    for g, (a, why) in itertools.product(groups, NOT_HELD):
        def make(style, rng, g=g, a=a, why=why):
            base = "casual" if style == "typo" else style
            text = styled(ASK_FORMS, style, rng, a=a, g=group_ref(g, base, rng), G=group_ref(g, "formal", rng))
            return text, dict(kind="decline", truth=why), dict(kind="un")
        yield ("un", g, a), make
    for g, (tpl, why) in itertools.product(groups, EVENTS):
        def make(style, rng, g=g, tpl=tpl, why=why):
            text = tpl.format(g=group_ref(g, "casual" if style in ("typo", "casual") else "terse", rng))
            text = typo(text, rng) if style == "typo" else text
            return (text[0].upper() + text[1:] if style == "formal" else text), dict(kind="decline", truth=why), dict(kind="un")
        yield ("event", g, tpl), make
    for g, country in itertools.product(groups, COUNTRIES):
        def make(style, rng, g=g, country=country):
            text = rng.choice([f"how many {group_ref(g, 'casual', rng)} are in {country}?",
                               f"{group_ref(g, 'terse', rng)} customers in {country}"])
            text = typo(text, rng) if style == "typo" else text
            return text, dict(kind="decline", truth="Only UK share; no country breakdown."), dict(kind="un")
        yield ("country", g, country), make


# ----------------------------------------------------------------------------- assembly
def counts_for(n, mix):
    total = sum(mix.values())
    raw = {k: n * x / total for k, x in mix.items()}
    out = {k: int(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


def style_sequence(n, rng, mix=STYLE_MIX):
    seq = [s for s, c in counts_for(n, {k: round(x * 1000) for k, x in mix.items()}).items() for _ in range(c)]
    rng.shuffle(seq)
    return seq


def build(n, D, seen=(), exclude=frozenset(), seed=SEED, id_prefix="X", used=frozenset()):
    """n new data questions: the held-out type mix scaled to n, wording styles in the practice mix.

    D is one copy of the data (the CSVs), used only to skip near-ties and empty filters at generation
    time. `seen` holds earlier question texts, which are never repeated; `exclude` holds template keys
    (repr) to leave out, such as ones whose truth differs between the CSV and live copies. An extension bank
    passes its own seed and ID prefix, and the keys an earlier bank `used`, which are tried only after
    every unused one."""
    rng = random.Random(seed)
    h = H(D)
    for m, sel, asc in itertools.product(RANK_WORDS, ["all"] + list(TIERS) + list(STYLES), (False, True)):
        rank_ok_cache[(m, sel, asc)] = rank_ok(h, m, sel, asc, 1)
    pools = dict(lookup=list(lookups()), total_share=list(totals()), ranking=list(rankings(h)),
                 comparison=list(comparisons(h)), breakdown=list(breakdowns()), filter=list(filters(h)),
                 model_eval=list(model_evals(h)), ambiguous=list(ambiguous(h)), unanswerable=list(unanswerables()))
    want = counts_for(n, MIX)
    texts = {t.lower() for t in seen}
    items, metas = [], {}
    for typ in MIX:
        if typ == "followup":
            continue
        pool = pools[typ][:]
        rng.shuffle(pool)
        pool.sort(key=lambda km: repr(km[0]) in used)  # stable: unused templates first
        styles = style_sequence(want[typ], rng)
        i = 0
        uses = {}
        while len([x for x in items if x["type"] == typ]) < want[typ]:
            assert i < 3 * len(pool), f"{typ}: pool of {len(pool)} too small for {want[typ]}"
            key, make = pool[i % len(pool)]
            i += 1
            if repr(key) in exclude:
                continue
            if uses.get(key, 0) >= (1 if i <= len(pool) else 2):
                continue
            style = styles[len([x for x in items if x["type"] == typ])]
            text, truth, meta = make(style, rng)
            if text.lower() in texts:
                continue
            texts.add(text.lower())
            uses[key] = uses.get(key, 0) + 1
            iid = f"{id_prefix}{PREFIX[typ][1:]}{len([x for x in items if x['type'] == typ]) + 1:03d}"
            items.append(dict(id=iid, type=typ, style=style, q=text, truth=truth, sets="both", parent=None,
                              expect={"ambiguous": "ambiguous", "unanswerable": "decline"}.get(typ, "answer"), key=repr(key)))
            metas[iid] = meta
    # follow-ups: each on a different parent whose kind has a follow-up
    parents = [it for it in items if it["type"] not in ("ambiguous", "unanswerable")]
    rng.shuffle(parents)
    fu_styles = style_sequence(want["followup"], rng, FU_STYLES)
    fus = []
    for p in parents:
        if len(fus) == want["followup"]:
            break
        made = followup_for(metas[p["id"]], rng)
        if not made:
            continue
        text, truth = made
        style = fu_styles[len(fus)]
        text = typo(text, rng) if style == "typo" else text
        fus.append(dict(id=f"{id_prefix}FU{len(fus) + 1:03d}", type="followup", style=style, q=text, truth=truth, sets="both",
                        parent=p["id"], expect="answer", key=f"fu:{p['key']}"))
    assert len(fus) == want["followup"], (len(fus), want["followup"])
    return items + fus, {k: len(v) for k, v in pools.items()}


def evaluate(items, D):
    h = H(D)
    return {it["id"]: (it["truth"](h) if callable(it["truth"]) else it["truth"]) for it in items}
