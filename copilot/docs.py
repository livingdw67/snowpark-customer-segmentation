"""Answers to "how" and "why" questions come from the project's own documentation.

README.md is split into its sections, plus a glossary generated from the metric catalogue, and
the sections that best match the question (keyword overlap with a few domain synonyms) are
handed to the methodology agent as its only source.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from .semantics import SCHEME_METRICS, SCHEME_NAMES, SEGMENT_METRICS, metrics_named_in

README = Path(__file__).resolve().parent.parent / "README.md"

STOP = set("""a an the and or of to in on for is are was were be been how why what which who does do did
it its this that these those with by from as at we our you your about into can could would should
there their them they than then so if not no yes all any each more most less least i me my get got
made make makes used use using work works""".split())

# Question words that should also match the vocabulary the README uses.
SYNONYMS = {
    "forecast": ["outcome", "expected", "revenue", "poisson", "gradient", "boosting", "predict"],
    "forecasts": ["outcome", "expected", "revenue", "poisson", "gradient", "boosting", "predict"],
    "predict": ["outcome", "gradient", "boosting", "calibrated", "probability"],
    "prediction": ["outcome", "gradient", "boosting", "calibrated"],
    "k": ["axis", "stability", "silhouette", "chosen", "tolerance"],
    "clusters": ["segments", "bisectingkmeans", "hdbscan", "natural"],
    "cluster": ["segments", "bisectingkmeans", "hdbscan"],
    "segments": ["segment", "axis", "tiers"],
    "ari": ["stability", "bootstrap", "rand"],
    "stable": ["stability", "ari", "persistence"],
    "churn": ["active", "buys", "again", "probability"],
    "accurate": ["auc", "backtest", "capture", "evaluation"],
    "accuracy": ["auc", "backtest", "capture", "evaluation"],
    "tested": ["backtest", "evaluation", "folds", "tests"],
    "validated": ["backtest", "evaluation", "folds"],
    "rescore": ["task", "monthly", "rescore_customers"],
    "cleaned": ["cleaning", "duplicates", "overlap", "staging"],
    "data": ["online", "retail", "uci", "transactions"],
    "calculated": ["poisson", "regressor", "model", "trained"],
    "computed": ["poisson", "regressor", "model", "trained"],
    "probability": ["calibrated", "temperature", "active"],
    "period": ["2009", "2011", "snapshots", "covers"], "cover": ["2009", "2011", "snapshots", "period"],
    "covers": ["2009", "2011", "snapshots", "period"], "dates": ["2009", "2011", "snapshots"],
    "rescored": ["re-scoring", "monthly", "task"], "re-scored": ["re-scoring", "monthly", "task"],
    "often": ["monthly", "re-scoring"], "split": ["backtest", "fold", "fits", "scores"],
    "criterion": ["r2", "r²", "value", "axis", "chooses"], "r2": ["r²", "criterion", "value", "axis"],
    "medians": ["median", "combined", "pooled"], "averaged": ["median", "combined", "pooled"],
    "snowflake": ["never", "leaves", "procedures", "stored"],
    "estimated": ["calibrated", "classifier", "gradient", "boosting", "model"],
    "active": ["buys", "again", "classifier", "calibrated", "probability"],
    "algorithms": ["scikit", "learn", "bisectingkmeans", "histgradientboostingclassifier", "used"],
    "models": ["scikit", "learn", "histgradientboostingregressor", "classifier", "used"],
}
DEFINITION = re.compile(r"\b(mean|means|meaning|definition|define|stand for|what is|what's)\b", re.IGNORECASE)


@dataclass
class Section:
    title: str
    text: str


def _glossary() -> list[Section]:
    out = []
    for m in SEGMENT_METRICS.values():
        agg = {"sum": "It adds up across segments.",
               "weighted": "Across several segments it is a customer-weighted average.",
               "median": "It is a median per segment and can't be combined across segments.",
               "ratio": "For a group it is the group's total divided by its customers."}[m.agg]
        how = {"p_buys_again": " It is estimated by a HistGradientBoostingClassifier, calibrated with "
                                "temperature scaling on a held-out customer-grouped split (option C).",
               "expected_revenue": " It is predicted by a HistGradientBoostingRegressor with Poisson loss, "
                                    "trained on every past quarterly snapshot pooled (option C)."}.get(m.key, "")
        out.append(Section(f"Glossary: {m.label}",
                           f"{m.label} (column {m.column}; also called {', '.join(m.synonyms)}). "
                           f"A high value means {m.high}; a low value means {m.low}. {agg}{how}"))
    for m in SCHEME_METRICS.values():
        out.append(Section(f"Glossary: {m.label}",
                           f"{m.label} (column {m.column}), a backtest measure. A high value means {m.high}."))
    out.append(Section("Glossary: value tiers C1 C2 C3 C4",
                       "Value tiers come from the outcome-guided model (option C) and are ordered by predicted "
                       "next-quarter value: C1 = Top spenders (highest), C2 = Core, C3 = Occasional, "
                       "C4 = Low value (lowest). A segment code like C2-B2 is <value tier>-<buying style>."))
    out.append(Section("Glossary: buying styles B1 B2 B3",
                       "Buying styles come from the behavior axis of the two-axis model (option B): "
                       "B1 = Bulk buyers (the largest baskets, wholesale-like), B2 = Holiday buyers (most of "
                       "their revenue falls in October-December), B3 = Small-basket buyers (smaller baskets "
                       "at higher unit prices). A segment code like C2-B2 is <value tier>-<buying style>."))
    faq = {
        "data period and snapshots": "The data is UCI Online Retail II, transactions from 1 December 2009 to "
            "9 December 2011. Customer features are built at five quarterly snapshots (2010-12-10, 2011-03-10, "
            "2011-06-10, 2011-09-10 and 2011-12-10), each looking back 365 days, with outcomes over the next 91 days.",
        "backtest split train eval": "Each backtest fits on one snapshot and scores the next one: the q2 fold fits "
            "on 2011-03-10 and scores 2011-06-10; the q3 fold fits on 2011-06-10 and scores 2011-09-10. Scores are "
            "compared with what customers actually did in the 91 days after the scoring snapshot.",
        "re-scoring how often": "Customers are re-scored monthly: a Snowflake Task calls RESCORE_CUSTOMERS at "
            "06:00 UTC on the 1st of each month, using the frozen production models.",
        "next-revenue R2 criterion": "The value axis chooses its number of segments k by out-of-fold R² of "
            "next-quarter log revenue explained by segment membership; the behavior axis uses silhouette. Both "
            "require bootstrap stability ARI ≥ 0.85 and every segment holding ≥ 5% of customers, and prefer fewer "
            "segments within a tolerance of 0.01.",
        "why medians can't be combined": "A tier's median isn't the average of its segments' medians: the median "
            "of the pooled customers depends on every customer's value, and the published results only hold each "
            "segment's median. So tier medians aren't reported; per-segment medians are.",
        "why Snowflake": "Customer data never leaves Snowflake: steps 01-04 are SQL, and all fitting, scoring and "
            "evaluation run in Python stored procedures inside Snowflake. The local machine only issues CALL "
            "statements, and the charts read aggregated tables only.",
    }
    for title, text in faq.items():
        out.append(Section(f"Glossary: {title}", text))
    out.append(Section("Glossary: schemes", "The compared schemes are: " +
                       "; ".join(f"{k} = {v}" for k, v in SCHEME_NAMES.items()) + "."))
    return out


def sections(readme: Path = README) -> list[Section]:
    text = readme.read_text(encoding="utf-8")
    text = re.sub(r"```mermaid.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images
    parts, title, buf = [], "Overview", []
    for line in text.splitlines():
        m = re.match(r"^#{1,4}\s+(.*)", line)
        if m:
            if "".join(buf).strip():
                parts.append(Section(title, "\n".join(buf).strip()))
            title, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        parts.append(Section(title, "\n".join(buf).strip()))
    return parts + _glossary()


def _terms(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9_²]+", text.lower()) if w not in STOP and len(w) > 1 or w == "k"]
    expanded = list(words)
    for w in words:
        expanded += SYNONYMS.get(w, [])
    return expanded


def retrieve(question: str, k: int = 2, corpus: list[Section] | None = None) -> list[Section]:
    corpus = corpus or sections()
    terms = _terms(question)
    docs = [set(re.findall(r"[a-z0-9_²]+", (s.title + " " + s.text).lower())) for s in corpus]
    n = len(corpus)
    scores = []
    for s, words in zip(corpus, docs):
        title = s.title.lower()
        score = 0.0
        for t in set(terms):
            df = sum(t in d for d in docs)
            if t in words:
                score += math.log(1 + n / (1 + df))
            if t in title:
                score += 1.5 * math.log(1 + n / (1 + df))
        if s.title.startswith("Glossary") and DEFINITION.search(question):
            score *= 1.5  # "what does X mean" is usually a definition
        scores.append(score)
    ranked = sorted(range(n), key=lambda i: -scores[i])
    found = [corpus[i] for i in ranked[:k] if scores[i] > 0]
    codes = re.findall(r"\b([cb])[1-4]\b", question.lower())
    if codes:  # "what does B2 mean": the code glossary first
        title = "Glossary: buying styles" if codes[0] == "b" else "Glossary: value tiers"
        entry = next((s for s in corpus if s.title.startswith(title)), None)
        if entry:
            return [entry] + [s for s in found if s is not entry][: k - 1]
    if DEFINITION.search(question):
        # A definition of a known metric comes straight from its glossary entry.
        named = metrics_named_in(question)
        if named:
            entry = next((s for s in corpus if s.title == f"Glossary: {SEGMENT_METRICS[named[0]].label}"), None)
            if entry:
                found = [entry] + [s for s in found if s is not entry][: k - 1]
    return found
