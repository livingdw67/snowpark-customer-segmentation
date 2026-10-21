"""How often each kind of question is asked in practice, and the counts that gives in 250 tests.

The Segment Assistant is a general chat over the segment results: people ask it for figures,
for explanations of the method, and for writing (persona cards, emails). Cortex Analyst is a
text-to-SQL service, so the people using an Analyst app ask it data questions, with a few
non-data questions typed into the same box.

Basis for the weights (see README.md for the full argument):
- Ambiguous / unanswerable share of data questions = 20%, split 55 / 45. Wang et al. 2023,
  "Know What I Don't Know" (ACL Findings), from ~3,000 failed questions in a commercial
  text-to-SQL product: "nearly 20% of the questions are problematic, including 55% ambiguous
  and 45% unanswerable".
- Follow-ups = 20% of answerable data turns. Assumption: conversational BI users refine
  ("users often ask a series of related questions to refine their search"), but no published
  log study gives a rate.
- Docs 14%, writing 10%, off-topic 6% of Assistant traffic. Assumption, from what the README
  says the assistant is for (follow-ups, explanations, drafting) and who would use it
  (marketing and account teams).
- Wording: terse 30%, casual 40%, formal 20%, typos 10%. Typos: commonly cited 10-15% of
  search queries are misspelled (Cucerzan & Brill 2004). Ask Data's research found
  questions are short and underspecified. The rest is an assumption.
Every weight is reported next to its result, and owen.py re-scores under perturbed weights.
"""

# First-turn answerable data questions: relative frequency within that group.
DATA_SUBTYPES = {
    "lookup": 0.285,       # one figure for one segment
    "total_share": 0.20,   # totals, shares, per-customer figures over groups
    "ranking": 0.18,       # top / bottom / nth, within a group
    "comparison": 0.145,   # two things side by side, yes/no, differences
    "breakdown": 0.07,     # by tier, by style, crosstab, every segment
    "filter": 0.05,        # segments meeting a threshold
    "model_eval": 0.07,    # backtests, k choice, data cleaning
}

# Share of all 250 tests by type, per system.
WEIGHTS = {
    "assistant": {
        "data": 0.56,          # answerable data questions incl. follow-ups (140)
        "ambiguous": 0.076,    # 19 = 55% of the 20% problematic share of 175 data questions
        "unanswerable": 0.064, # 16 = 45% of it
        "docs": 0.14,          # 35 methodology, definitions, advice
        "writing": 0.10,       # 25 persona cards, emails, summaries
        "off_topic": 0.06,     # 15 greetings, general knowledge, coding, 2 injection attempts
    },
    "analyst": {
        "data": 0.76,          # 190 incl. follow-ups
        "ambiguous": 0.104,    # 26
        "unanswerable": 0.088, # 22
        "non_data": 0.048,     # 12 docs / writing / off-topic questions typed into a data chat
    },
}

COUNTS = {
    "assistant": {"lookup": 32, "total_share": 22, "ranking": 20, "comparison": 16, "breakdown": 8,
                  "filter": 6, "model_eval": 8, "followup": 28, "ambiguous": 19, "unanswerable": 16,
                  "docs": 35, "writing": 25, "off_topic": 15},
    "analyst": {"lookup": 43, "total_share": 30, "ranking": 27, "comparison": 22, "breakdown": 11,
                "filter": 8, "model_eval": 11, "followup": 38, "ambiguous": 26, "unanswerable": 22,
                "docs": 6, "writing": 3, "off_topic": 3},
}

STYLE_TARGET = {"terse": 0.30, "casual": 0.40, "formal": 0.20, "typo": 0.10}

# Types grouped for reporting and for the Owen decomposition: both systems have items in each.
REPORT_GROUP = {
    "lookup": "lookup", "total_share": "total_share", "ranking": "ranking", "comparison": "comparison",
    "breakdown": "breakdown", "filter": "filter", "model_eval": "model_eval", "followup": "followup",
    "ambiguous": "ambiguous", "unanswerable": "unanswerable",
    "docs": "non_data", "writing": "non_data", "off_topic": "non_data",
}

for system, counts in COUNTS.items():
    assert sum(counts.values()) == 250, (system, sum(counts.values()))
for system, w in WEIGHTS.items():
    assert abs(sum(w.values()) - 1) < 1e-9, system
