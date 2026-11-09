# Segment Assistant

A chat app over the results, for the people who'd use the segments. A LangGraph router sends each request to one of four agents:

| Agent | Handles | How it answers |
|---|---|---|
| **Calculation** | *"Which segment has the most recent customers?"* · *"Crosstab of expected revenue by tier and buying style"* | The model picks one calculation and its inputs as JSON. Code computes the numbers and writes the answer. |
| **Documentation** | *"How are the forecasts made?"* · *"What does P(buys again) mean?"* | Answers from the matching sections of the README and docs/, and cites them. |
| **Writing** | *"Write a persona card for Core / Holiday buyers"* | Writes from the segment's profile, with the checks below. |
| **Out of scope** | Coding, general knowledge, role-play | A fixed, polite refusal. |

## Why the model doesn't write SQL

The first version let the model write SQL. An adversarial test of 137 questions found only 47% answered correctly, and most errors carried real numbers attached to wrong conclusions:
- "most recent" sorted the wrong way;
- shares came out as 100% because the total was filtered with the part;
- one segment's figure was reported as a whole tier's;
- a number the user typed was trusted as fact.

Checking that each figure came from the query caught almost none of these, because the error was in the meaning, not the digits.

The rebuild moves every definition into code ([assistant/semantics.py](../assistant/semantics.py)):
- **A metric catalogue** says what each measure means, which direction is "better", and how it combines: customers and predicted revenue add up, probabilities are customer-weighted, medians are never combined.
- **About a dozen tested calculations:**
  - value (including revenue per customer and combined groups), profile and share;
  - rank (within a tier or style when asked) and compare (with differences, ratios and tier-by-tier medians);
  - crosstab, threshold count and traits;
  - backtest schemes (per fold or averaged), k choice and cleaning.
- **The model only chooses** a calculation and its inputs. Code resolves names ("core holiday buyers" → C2-B2), corrects the sort direction from the question's wording, swaps in the metric the question names, reports ties, and declines anything that needs customer-level rows.

**Writing-agent guards:**
- Figures must come from the segment profile; figures the model proposes itself must be labelled "(proposed)".
- A number the user supplies that contradicts the data is dropped, with a note.
- A year-round segment described as seasonal gets flagged.
- Unknown segment names get a question back instead of a guess.
- Country targeting is declined, and injected instructions ("ignore all previous rules…") are stripped.

**Why an LLM at all?** The calculation agent is a natural-language front end to parameterised queries. A dashboard with dropdowns would answer those questions with no model, no inference cost and no risk of a wrong pick. The model earns its place where a dashboard can't help: follow-ups ("what about holiday buyers?"), explanations grounded in this README, and drafting persona cards and campaigns from the segment profiles. Its job in the calculation agent is narrow on purpose: choosing what to compute, never computing it.

Every answer shows how it was produced: the routing decision, the plan and any adjustment code made to it, the rows computed, and a figure check ([assistant/verify.py](../assistant/verify.py)). Follow-ups like "what about holiday buyers?" edit the previous plan.

**Accuracy on fresh questions.** Each round, a tester agent writes new adversarial questions, checked against every earlier round's so none repeat, in formal, casual, terse and typo-laden wording. Every answer is graded against truths computed with pandas from the result CSVs:

| Round | Questions graded | Correct | On round 2's 5 categories |
|---|---|---|---|
| 1: model-written SQL | 137 | 47% | |
| 2: first semantic layer | 200 (5 categories) | 63% | 63% |
| 3: second rebuild | 200 (9 categories) | 69% | 73% |
| 4 | 150 (9 categories) | 73% | 76% |
| 5 | 300 (9 categories) | 71% | 74% |

**Accuracy has plateaued at about 70% since round 3.** Each round's fixes repair most of that round's failures when the same questions are replayed (57 of 62 after round 3, 39 of 41 after round 4, 84 of 88 after round 5), but new questions keep finding new gaps. More wording rules are giving diminishing returns. The next thing to try is a larger model (`qwen2.5:14b` locally, or Cortex), which hasn't been tested yet.

Round 5 by category:

| Category | Correct |
|---|---|
| Comparisons | 93% |
| Crosstabs | 87% |
| Backtest, k and cleaning | 76% |
| Declines and injection | 76% |
| Methodology and advice | 72% |
| Rankings | 69% |
| Lookups and totals | 65% |
| Follow-ups | 63% |
| Writing | 50% |

No answer in round 5 was unsafe. The most common failure is an unhelpful answer (wrong format, partial, or a unit ignored) at 12% of questions, then a wrong claim or direction (8%) and a wrong number (6%).

**Fixes after round 5:**
- **The question's wording beats the plan:** several exclusions ("except C4-B3 and C1-B1"), "every segment" meaning all of them, a filter's measure kept apart from the measure being totalled ("expected revenue from segments whose P(buys again) is above 50%"), "share of X that are Y" divided by X, "how many" vs "what proportion" in every calculation, ordinals ("second smallest") and per-segment averages.
- **Rankings:** groups narrowed to one segment each are ranked and labelled as segments; tiers and styles on a median are shown as the range of their segments instead of an error.
- **Writing agent:** wording the data contradicts (lapsed, seasonal, bulk, infrequent, a "December spike") is rewritten, and if it survives two rewrites it is removed rather than noted; £ and % offers are marked "(proposed)"; SMS, tweets and notes are stripped of headings, greetings, persona cards, codes and internal segment names.
- **Units and declines:** pence, thousands, millions and a no-exchange-rate note; SQL, "which customer" and "which country" requests are declined.

Still open from round 5: a terse "why?" follow-up that misstates the k rule, a "2-sentence idea" that reads recency as frequency, and two campaigns that the clean-up leaves thin (no offer or an empty body).

**Regression check.** [assistant/evaluate.py](../assistant/evaluate.py) re-asks 67 questions from development and earlier rounds, scoring words ("No", "lower") as well as figures against pandas. On the last full run with `qwen2.5:7b`:

| Set | Correct |
|---|---|
| Tuning (21) | 21 (100%) |
| Held out (11) | 11 (100%) |
| Earlier red-team failures (20) | 20 (100%) |
| Round 5 failures (15) | 15 (100%) |

Every figure in every answer traced back to the data. Most of these questions were tuned against, so 100% means past fixes still hold, not that new questions will be answered correctly; the rounds above measure that. Results are in `results/assistant_eval.csv`; rerun with `python -m assistant.evaluate`.

**Against Cortex Analyst, on questions weighted to real use.** The rounds above pick questions to break the Assistant. [practice_eval/](../practice_eval/) asks them in the proportions people actually ask them (lookups, rankings, follow-ups, ambiguous and unanswerable questions, documentation and writing requests), with every truth computed from the data, and puts the same data questions to Snowflake's Cortex Analyst. All of it ran on the Snowflake build's numbers, which are kept in [practice_eval/frozen_results/](../practice_eval/frozen_results/):

- **Practice mix, 250 questions each:** Assistant 82.8%, Analyst 89.6%. On the 238 data questions both answered, 85.3% vs 90.3% (McNemar p = 0.09).
- **Held-out retest of the fixes:** 250 new questions, frozen before any fix was written and graded blind. The old code scored 80.0% and the fixed code **87.2%**; they disagreed on 18 questions and the fixed code was right on all 18 (p = 8e-6). On the 175 data questions, the fixed Assistant got 91.4% to Analyst's 86.9% (p = 0.23).
- **Power run, 1,400 paired data questions** (sized for 90% power at 3.4 points): Analyst 93.8%, Assistant 87.9%. Analyst is ahead by 5.9 points (95% CI 3.8–8.0, p = 4e-8). The biggest gap is questions the data can't answer, such as other countries or CRM fields, where the Assistant gives the nearest figure instead of declining.

| | Offline (default) | Live |
|---|---|---|
| Data | DuckDB over `results/*.csv` (aggregated, committed) | The same tables from the `ML` schema in Snowflake |
| LLM | Ollama, `qwen2.5:7b` | Snowflake Cortex, `llama3.1-70b` |
| Needs | [Ollama](https://ollama.com) | `.env` with Snowflake credentials |

Offline mode exists so anyone can try the app without a Snowflake account. It reads the segment-level result tables exported to `results/` (no customer rows), and DuckDB runs the same fixed queries over them. Live mode is untested. The calculations work on segment-level tables, so customer-level questions are declined in both modes.

```bash
pip install -r requirements-assistant.txt   # no scikit-learn needed
ollama pull qwen2.5:7b
streamlit run app.py
```

`ASSISTANT_MODE` (`auto`/`local`/`live`/`databricks`), `ASSISTANT_LLM` (`auto`/`ollama`/`cortex`/`databricks`), `OLLAMA_MODEL`, `CORTEX_MODEL`, `DATABRICKS_CHAT_ENDPOINT` and `LLM_TIMEOUT` override the defaults. The app only runs fixed, read-only queries, and offline DuckDB has file and network access switched off. In live mode, also run the app under a read-only role.
