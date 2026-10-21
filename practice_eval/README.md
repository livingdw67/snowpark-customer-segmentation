# Practice-mix eval: Segment Assistant vs Cortex Analyst

Built 2026-10-16 for this repo. The scripts find the repo root from their own location; set `SEGMENTATION_REPO` to point them at another checkout.

It asks two systems 250 questions each about the customer-segmentation results:

- **The Segment Assistant**, in its default offline setup: DuckDB over `results/*.csv` and Ollama `qwen2.5:7b`, with the code-side semantic layer in `assistant/semantics.py`.
- **Cortex Analyst**, using the semantic model `segmentation_r7.yaml` on `@CUSTOMER_SEGMENTATION.ANALYST.SEMANTIC_MODELS`, over the `CUSTOMER_SEGMENTATION.ANALYST` views. A copy of the model is in `semantic_model/`.

Earlier rounds (1–7) were adversarial. Their questions were picked to find failures, in fixed category quotas. Here, each test type appears about as often as it would in real use, so the headline score estimates what a user would actually experience.

## Status

See **Results** below. Everything was run once, on 2026-10-16.

A second round, also on 2026-10-16, fixed the Assistant's weaknesses and retested it on a fresh, held-out question set: see **After the fixes**. The fixes are on the repo branch `practice-eval-fixes` (commits `9298527` and `98bcfbe`), merged into `main` on 2026-10-20 (PR #2, `f208579`).

A third round (2026-10-16 to 10-20) sized the comparison to 90% power: 1,400 paired data questions, fixed Assistant vs Analyst r8. **Analyst came out ahead, 93.8% to 87.9%.** See **Power run**.

## Composition

**Two sets of 250, built around a shared core.**
- **Shared core:** 175 data questions are asked of both systems: 140 answerable (28 of them follow-ups), 19 ambiguous and 16 unanswerable. This allows a paired, question-by-question comparison.
- **Assistant-only:** the Assistant's other 75 tests are documentation, writing and off-topic questions, the other things people use a chat app for.
- **Analyst-only:** Analyst's other 75 tests are 63 more data questions, plus 12 non-data questions taken from the Assistant's set. Those 12 are the kind people still type into a data chat.

| Type | Assistant | Analyst | Basis |
|---|---|---|---|
| Lookup (one figure for one segment) | 32 | 43 | assumption: the most common BI question |
| Totals and shares | 22 | 30 | assumption |
| Ranking | 20 | 27 | assumption |
| Comparison | 16 | 22 | assumption |
| Breakdown / crosstab | 8 | 11 | assumption |
| Threshold filter | 6 | 8 | assumption |
| Backtests, k, cleaning | 8 | 11 | assumption: rarer for business users |
| Follow-up turn | 28 | 38 | assumption: 20% of answerable data turns |
| Ambiguous | 19 | 26 | Wang et al. 2023: ~20% of real text-to-SQL questions are problematic, 55% ambiguous / 45% unanswerable |
| Unanswerable from the data | 16 | 22 | same |
| Documentation, method, advice | 35 | 6 | assumption: 14% of assistant traffic |
| Writing (personas, emails, summaries) | 25 | 3 | assumption: 10% |
| Off-topic (incl. 2 injection attempts) | 15 | 3 | assumption: 6% |

**Wording mix (both sets):** about 30% terse ("c2-b3 p buys again"), 40% casual, 20% formal and 10% with typos ("expected revnue").
- **Typos:** about 10–15% of search queries are commonly cited as misspelled (Cucerzan & Brill 2004).
- **Short questions:** Tableau's Ask Data research found questions are short and underspecified.
- **The rest of the wording split** is an assumption.

Only one figure is measured from real logs: the ambiguous/unanswerable share, from Wang et al., *Know What I Don't Know*, ACL Findings 2023. It comes from ~3,000 failed questions in a commercial text-to-SQL product. No published log study gives a full type mix for conversational BI. LinkedIn's SQL Bot paper (arXiv 2507.14372) names its top intents but gives no shares. So `owen.py` also re-scores both systems under randomly perturbed weights (Dirichlet, about ±30% on a mid-sized weight) and reports the range.

## Ground truth

- **Code, not hand-typed:** every truth is computed by `questions.py` from the data each system reads, never typed in.
  - The Assistant's truths come from the committed CSVs.
  - Analyst's truths come from the live views, which keep more decimals.
  - Only one truth differs by more than 1% between the two: the C1-B3 return rate, 1.0% vs 0.95%.
- **Ties are expected in full.** E.g., three segments tie at one order a year.
- **Ambiguous questions list every reasonable reading.** An answer passes if it asks which was meant, or if it states the reading it chose and gets that reading's numbers right.
- **Unanswerable questions pass on a clear "the data doesn't hold that".** Offering the nearest available figure, labelled as such, is fine.
- **Documentation, writing and off-topic questions** have written criteria: the README facts, the segment profile, and any length or format constraint.

## Grading

1. **Automatic first pass:** `grade.py auto` checks figures, segment codes, yes/no words and decline wording.
   - Figures are matched rounding-aware, within 0.5% or the precision they're written at.
   - This check is independent of the Assistant's own figure checker.
2. **Full read:** every answer is then read and given a verdict in `results/review_<system>.txt`, using the labels from earlier rounds: correct, wrong number, wrong claim, unhelpful, declined-but-answerable, answered-instead-of-declining, unsafe, error.
3. **Merge:** `grade.py merge` writes `results/grades_<system>.csv` and reports how often the first pass agreed with the read.

Analyst answers are graded as the Cortex Analyst Streamlit quickstart shows them: its interpretation of the question plus the result table. A table that plainly answers a yes/no question counts, even without the word "yes".

For non-data questions the two systems are judged against what each is built to do:
- the Assistant should answer documentation and writing requests;
- Analyst should say it only answers data questions, without inventing anything.

## Owen values

See the docstring in `owen.py`. The headline gap is split with one union per question type and two players per union: *mix* (how often the type is asked) and *accuracy* (how well it's answered). That is the rate/volume structure semantic-rca uses.
- **Why it's needed:** the two test sets have different mixes by design, so the raw gap partly reflects which questions each system was asked.
- **Closed form:** the score is a sum over types, so the Owen values have a closed form. The exact coalition enumeration is checked against it.
- **Confidence intervals:** resample questions within each type, keeping shared questions paired.

## Files

- `composition.py`: weights, counts and their basis.
- `questions.py`: the 313 unique questions and their truth functions. `build_questions.py` evaluates them on both data copies and writes `questions.json`.
- `run_assistant.py`, `run_analyst.py`:
  - each resumes if interrupted;
  - follow-ups run in their parent question's thread;
  - `--limit N` runs a smoke test.
- `grade.py`, `owen.py`: grading and analysis.
- `stats.py`: the statistical tests, written to `results/dashboard/*.json`, the data behind the results page (https://claude.ai/artifact/KjE1HZbKDEmhkVswD539Y6).
  - Wilson intervals for each score.
  - McNemar (exact, mid-p and chi-square) with Newcombe's paired interval and a sample-size estimate.
  - McNemar assumption checks: clustered by conversation (Durkalski 2003), first questions only, and on machine grades.
  - Cohen's kappa between the first pass and the read.
  - Fisher's exact test for follow-ups vs first questions.
  - A permutation chi-square test for wording (the chi-square's expected-count assumption fails).
  - A Monte Carlo Shapley check against the Owen values.
- `results/`: answers, first-pass checks, verdicts, grades and `summary.json`.
- Round two (after the fixes):
  - `questions_v2.py`, `build_questions_v2.py`: the held-out bank, 250 new questions in the same mix (IDs start with H). `questions_v2.json` was frozen before any fix was written; `questions_v2.frozen.txt` holds its sha256 and timestamp.
  - `blind_v2.py`: blind grading of old vs fixed code. Identical answers share a letter; the rest are shuffled under A and B (seed 2026). The key stays in `results/v2_blind_key.json` until `unblind`.
  - `stats_v2.py`: the paired tests on the held-out set, written to `results/dashboard/v2_*.json`.
  - `tuned.py`: the fixed code on the original questions, with unchanged answers keeping their old verdicts.
- Round three (power run):
  - `POWER_RUN_PLAN.md`: the analysis plan, written before any answer, its 90% amendment, and the departures after the run.
  - `questions_v3.py`, `build_questions_v3.py`: the template generator. `questions_v3.json` (875, IDs X) and `questions_v3b.json` (350, IDs Y) were each frozen with a `.frozen.txt` hash.
  - The frozen hashes are of the files with LF line endings, as git stores them. A Windows checkout with `core.autocrlf` adds CRs, so strip them before checking (`tr -d '\r' < questions_v2.json | sha256sum`).
  - `power_grade.py`: the read list, the audit and the 2% rule. `compact` and `mark` drive the full read.
  - `stats_power.py`: the pre-registered tests, written to `results/dashboard/power_*.json`.
  - `power_page_data.py`: writes `power_app/rounds.json`.
  - `power_app/`: the Streamlit preview page for the portfolio. Run `streamlit run streamlit_app.py`.

## Cost

Analyst costs 0.067 credits per message. One run of 250 is about 17 credits, plus a little warehouse time to run the generated SQL. The Assistant runs locally at no cost.

Run from this folder with the repo's venv (`.venv-assistant`; older checkouts may still call it `.venv-copilot`):

```
..\.venv-assistant\Scripts\python build_questions.py
..\.venv-assistant\Scripts\python run_assistant.py
..\.venv-assistant\Scripts\python run_analyst.py
python grade.py auto        # then read and write results/review_<system>.txt
python grade.py merge
..\.venv-assistant\Scripts\python owen.py
```

## Results

**A semantic layer we built ourselves, running a free open-weight model (qwen2.5 7B) on our own machine, came within 5.0 points of Snowflake's paid Cortex Analyst.**
- On the 238 data questions both answered, it got 85.3% right to Analyst's 90.3%.
- The gap falls short of significance at the 5% level (McNemar p = 0.09), though the evidence leans Analyst's way.
- It costs nothing per question; Analyst costs about 0.067 credits a question.
- The 95% interval runs from Analyst 0.4 points worse to 10.5 points better.

| | Segment Assistant (qwen2.5:7b, offline) | Cortex Analyst (segmentation_r7.yaml) |
|---|---|---|
| Practice-mix score | **207 / 250 = 82.8%** (95% CI 78.2–87.1%) | **224 / 250 = 89.6%** (85.9–93.1%) |
| 238 data questions both answered | 203 = 85.3% | 215 = 90.3% |
| of which the 175 in both practice sets | 151 = 86.3% | 157 = 89.7% |
| of which the 63 only in Analyst's set | 52 = 82.5% | 58 = 92.1% |

- **Where the paired questions come from.** The 238 are the 175 data questions in both practice sets, plus the 63 data questions only Analyst's set had. Those 63 were asked of the Assistant in a second run with the same setup (`run_assistant.py --items extra`); its practice-mix score still counts only its own 250.
- **On data questions the two are close, with Analyst slightly ahead.** Only one system was right on 42 of the 238: the Assistant on 15, Analyst on 27 (McNemar p = 0.088). On the first 175 alone the split was 13 vs 19 (p = 0.38). The extra 63 went 2 vs 8.
- **The headline gap is 6.8 points** (95% CI 1.2–12.3). Analyst stays ahead under every perturbed weighting tried; the 5–95% range of the gap is 4.1–9.6 points.
- **Compared with the adversarial rounds:** the Assistant scored about 70% on those, picked to break it. That's a different mix, not a regression; the 82.8% here estimates what a user would see.
- **First-pass checker vs the read:** they agreed on 170/175 auto-checkable Assistant answers and 219/238 Analyst answers.

### Statistical tests

| Test | Result |
|---|---|
| McNemar, 238 paired data questions | 15 vs 27 disagreements; exact p = 0.088, mid-p 0.066, chi-square 2.88 (p = 0.090). Difference +5.0 pts, 95% CI −0.4 to +10.5 (Newcombe). Significance would have needed a 28–14 split; detecting this gap with 80% power needs ~533 paired questions. |
| Only the 175 in both practice sets | 13 vs 19, p = 0.377 |
| Clustered by conversation | p = 0.070 |
| First questions only | 13 vs 21, p = 0.23 |
| Machine first-pass grades | 19 vs 26, p = 0.37 |
| Cohen's kappa, first pass vs read | Assistant 0.89 (0.77–0.98); Analyst 0.62 (0.45–0.77) |
| Fisher, follow-ups vs first questions | Assistant 71.4% vs 92.9%, p = 0.004; Analyst 89.5% vs 92.8%, p = 0.51 |
| Wording, permutation chi-square | p = 0.56 (Assistant), 0.38 (Analyst); plain chi-square invalid (expected counts 4.5 and 2.7) |

- **Grading caveat:** under machine grading Analyst's edge on the 238 shrinks from 12 questions to 7 (p = 0.37), so grading leniency for its bare tables may explain part of it.
- **Shapley vs Owen:** they coincide here. Monte Carlo Shapley over all 22 players matches Owen to within 0.00003, because types don't interact in the score.

### Owen values: why the scores differ

| Type | Assistant weight / accuracy | Analyst weight / accuracy | Mix | Accuracy | Type total |
|---|---|---|---|---|---|
| Non-data (docs, writing, off-topic) | 30.0% / 74.7% | 4.8% / 75.0% | **+2.9** | +0.1 | **+2.9** |
| Follow-up | 11.2% / 71.4% | 15.2% / 89.5% | −0.2 | **+2.4** | **+2.2** |
| Lookup | 12.8% / 93.8% | 17.2% / 100% | +0.5 | +0.9 | +1.4 |
| Ranking | 8.0% / 100% | 10.8% / 81.5% | +0.1 | **−1.7** | **−1.6** |
| Totals and shares | 8.8% / 95.5% | 12.0% / 86.7% | +0.2 | −0.9 | −0.8 |
| Comparison, unanswerable, model eval, filter, breakdown | | | | | +0.7, +0.7, +0.6, +0.5, +0.2 |
| Ambiguous | 7.6% / 73.7% | 10.4% / 76.9% | −0.3 | +0.3 | 0.0 |
| **Total** | | | **+3.4** (CI 0.0–7.4) | **+3.4** (CI −3.5–9.8) | **+6.8** |

All figures in the Mix, Accuracy and Type total columns are percentage points of headline score, Analyst minus Assistant.

- **Half the gap is mix.** The Assistant gets asked documentation and writing questions (30% of its traffic), which it answers at 75%, below the 86% reference. Analyst barely sees them. Mix explains this half, not accuracy.
- **The accuracy half is follow-ups (+2.4) and lookups (+0.9).** Analyst loses some of it back on rankings (−1.7) and totals (−0.9), from two Analyst defects below.

### Assistant failures (43)

1. **Follow-ups (8 of 28 wrong):**
   - It reuses the previous answer's segments when it shouldn't. "and the fewest?" after the largest segment returned that segment's share; FU15 and FU18 went the same way.
   - It drops the level: "and by customers?" after ranking tiers ranked segments instead.
   - Asked "how many customers" in those six segments, it gave everyone (4,261).
2. **Code overrides a correct plan.** At `assistant/graph.py:1168-1180`, a metric the question "names" replaces the plan's.
   - "customers outside uk" became the UK share; the plan was right with `non_uk_customers`. Only the pairs in `COUNT_OF` are exempt.
   - "bigger orders" became order count.
   - "how long since they last ordered" became tenure.
3. **Vague "how many X" questions:**
   - "top customers" and "small customers" answer with all customers (4,261);
   - "big buyers" and "frequent buyers" get a per-segment table and no count.
4. **Routing and documentation:**
   - Five method, advice or writing requests went to the calculation agent.
   - Four documentation answers contradict the README: the hybrid's capture as 66% vs 61% (it is 71% vs 66%), "the forecast handles Christmas", the value-tier method, and the post-cleaning row count.
5. **Writing:**
   - Whole-segmentation requests (slide intro, exec summary, sales-team message) get "Which segment should I write for?".
   - Guard rewrites leave false text: "Low value: recent", and a note that C4-B1 "isn't supported as Bulk buyers" though B1 *is* bulk.
   - It repeats a user's false claim ("our most frequent customers").
6. **Unanswerable:** asked for a sales rep, NPS or lifetime value, it answered with an unrelated figure instead of saying the data doesn't hold it.

### Analyst failures (26)

1. **Seven empty answers from one bug.**
   - **The SQL:** the semantic model says "return every one tied … instead of LIMIT 1", so Analyst writes `WHERE x = (SELECT MAX(x) …)`. On full-precision FLOAT columns (expected revenue, P(buys again), computed non-UK counts) this matches **no rows** in Snowflake, every time.
   - **Confirmed directly:** a `::NUMBER` cast or `QUALIFY RANK() … = 1` returns the right row.
   - **A regression:** round 6's model had no tie rule and wrote this pattern 0 times; round 7 added the rule and wrote it 6 times, 4 of them empty. Here it hit RK02, RK04, RK10, AM19 and FU24, plus the follow-ups FU08 and FU12 that inherited the empty answer.
   - **Fix:** tell Analyst to use `QUALIFY RANK()` for ties, or round the FLOAT columns in the `ANALYST` views.
2. **Nine answers with no total.** Asked for a tier or style total, Analyst returned the component rows without the total, sometimes after promising one in its interpretation (TS13, TS14, TS16, TS20, CP15, AM08, AM21, AM25), or listed 12 rows without picking the answer (AM22).
3. **Non-data requests:** writing and method questions (persona card, campaign ideas, "how were the segments created") got a profile table presented as the answer.
4. **One-offs:**
   - "in percent?" gave 100% (the total was filtered with the part).
   - "duplicate rows removed" summed every cleaning step.
   - "widest baskets" was read as units per order.
   - "most recent segment" was read as "latest snapshot".
   - "frequent" was mapped to bulk.
   - "platinum tier" was mapped silently to C1.
   - Next-quarter revenue per customer was labelled "lifetime value".

### Also found

`assistant/docs.py:137-141`: the glossary tells users to "pull the email list for a segment from" `ML.CUSTOMER_SEGMENTS_CURRENT`, but that view has no email column. The Assistant repeats it (DC31).

## After the fixes

The fixes target the Assistant failures above. They are on the repo branch `practice-eval-fixes` (`9298527`, then `98bcfbe`, which narrows the measure override further), with 27 new unit tests. All 167 unit tests pass, and the 67-question regression check scores 67/67.

1. **Measure words no longer overrule a good plan.** A measure the question names replaces the plan's only when the plan's own measure goes unnamed and the named one matches in full ("per order", "lapse"). "Biger orders" stays order value; UK vs outside the UK follows the wording.
2. **Unknown groups get a question back.** "Top customers", "loyal customers" and "platinum tier" now get "I don't have a group called ... Did you mean ...?" instead of all 4,261 customers.
3. **Typos.** When a group word looks misspelt, the plan's narrower group wins ("occassional small basket" is C3-B3, not all small-basket buyers).
4. **Yes/no comparisons** answer about the group before "than".
5. **Follow-ups.** Direction-only follow-ups ("and the highest?"), "those three", bare counts after a filter, and the tier level are carried over.
6. **"Per segment"** lists every segment; averaging needs an average word.
7. **Missing data is declined.** More countries, channels, sales reps and account owners, NPS and surveys, lifetime value, delivery, movement between segments, and customer names.
8. **Routing.** "Thanks" gets an acknowledgement. Summaries and outlines go to writing; definitions, model quality and "difference between" questions go to the docs agent; contact advice for a group goes to writing. Whole-segmentation writing uses all segments instead of asking which one.

### The held-out retest

Scoring the fixes on the questions they were written against would flatter them. So before any fix was written, a second set of 250 questions in the same mix was written and frozen (`questions_v2.json`, sha256 `be8ffb49…`, 2026-10-16T13:15:39Z). The old code (`d672e7d`) and the fixed code (`98bcfbe`) answered all 250, and Analyst answered the 175 data questions (about 12 credits).

Grading was blind (`blind_v2.py`). Each question's answers were shuffled under the letters A and B, with identical answers merged, so 296 distinct answers were graded without knowing which version wrote which.

| | Old code | Fixed code | Cortex Analyst |
|---|---|---|---|
| All 250 held-out questions | 200 = 80.0% (95% CI 74.6–84.5%) | **218 = 87.2%** (82.5–90.8%) | not asked |
| The 175 held-out data questions | 147 = 84.0% | **160 = 91.4%** | 152 = 86.9% |

- **The fixes helped and broke nothing on new questions.** The two versions disagreed on 18 questions, and the fixed code was right on all 18.
  - Types: lookup +2, ranking +1, comparison +2, follow-up +2, ambiguous +3, unanswerable +3, and docs, writing and off-topic +5.
  - The mix is the same for both versions, so the gain splits additively: each type's Shapley (and Owen) value is its share of questions times its change in accuracy.
- **Fixed Assistant vs Analyst: no significant difference, but a tighter bound.** The Assistant won the disagreements 21 to 13 (p = 0.23). The 95% interval for Analyst minus Assistant is −11.3 to +2.0 points: on questions like these, Analyst is at most about 2 points better.
- **Most of that lead is one Analyst quirk.** Seven of Analyst's 23 misses are its `x = (SELECT MAX(x))` SQL matching no rows on full-precision FLOAT columns. Counted as right, the split is 15 to 14: level. Rounding those columns in the `ANALYST` views would likely fix it.
- **Before the fixes, on the same questions,** the old Assistant got 84.0% to Analyst's 86.9% (18 vs 23, p = 0.53). That's the same picture as the first round.

### Statistical tests (held-out)

| Test | Result |
|---|---|
| McNemar, old vs fixed, 250 questions | 0 vs 18; exact p = 0.000008 (one in 131,072); Holm-adjusted p < 0.001. Gain +7.2 points, 95% CI +3.9 to +10.8 (Newcombe) |
| ↳ clustered by conversation (Durkalski) | p = 0.00002 |
| ↳ first questions only | 0 vs 16, p = 0.00003 |
| ↳ data questions only | 0 vs 13, p = 0.0002 |
| ↳ docs, writing and off-topic only | 0 vs 5, p = 0.063 |
| ↳ machine first-pass grades (data questions) | 0 vs 13, p = 0.0002 |
| McNemar, fixed Assistant vs Analyst, 175 data questions | 21 vs 13; p = 0.23 (Holm 0.46). Analyst − Assistant −4.6 points, 95% CI −11.3 to +2.0 |
| ↳ clustered by conversation | p = 0.21 |
| ↳ first questions only | 16 vs 10, p = 0.33 |
| ↳ machine first-pass grades | 23 vs 14, p = 0.19 |
| ↳ Analyst's 7 FLOAT-equality failures counted right | 15 vs 14, p = 1.0 |
| McNemar, old Assistant vs Analyst, 175 data questions | 18 vs 23, p = 0.53 |

Holm's adjustment covers the three main comparisons, since three tests are three chances of a fluke. The check rows test the main result's assumptions: independence (follow-ups share context), and grader bias (blinding was partial, because some fixes give recognisable answers, such as the question back). The machine first pass applies the same rules to both versions.

**Caveats.**
- The same person wrote the fixes and the held-out questions, after reading the first set's failures, and graded the answers. The new questions may lean toward what the fixes target; a set written by someone else would remove that doubt.
- Each version answered once. The model runs at temperature 0, but docs and writing answers still varied in wording between runs.

### Fixed-code failures on the held-out set (32)

- **Measures and totals:** "UK share" read as a share of the tier (HLK07, HLK25). Combined totals answered as a comparison (HTS15, HFU15). "Average per Core segment" answered with the total (HTS17). "Outside top spenders" (HTS18). "At least 5 times a year" applied to customer counts (HFL06). One plan failed to parse (HTS13).
- **Follow-ups:** "customer count?", "same by expected revenue" and "what's the gap?" (the last refused as off-topic).
- **Ambiguous and missing data:** "capture rate?" and "which segment performs best?" were misread (HAM09, HAM19). Churn in October and two-year spend were answered instead of declined (HUN08, HUN09).
- **Docs (9):** answers that are vague, or say the README doesn't cover something it does (the 44% holiday under-forecast, the CC BY 4.0 licence). "Who should our sales team call this week?" was refused as off-topic.
- **Writing (8):** whole-segmentation pieces now get written, but say "nine distinct groups" (there are 12). The bulk-buyer note on C4-B1 is still wrong. Two requests were misrouted, and two missed what was asked.

### On the questions the fixes came from (tuned on)

The fixed code was also rerun on the original 313 questions (`tuned.py`). Unchanged answers kept their old verdicts; the 63 changed ones were read, not blind.

- Original 250: 207 = 82.8% → **228 = 91.2%** (22 fixed, 1 broke). The break is DC35: the new missing-data rule matches "account managers" and declines a question that was answerable.
- 238 shared data questions: fixed Assistant **95.0%** vs Analyst 90.3% (20 vs 9, p = 0.061). Before the fixes it was 85.3% vs 90.3%.
- One more break among the 63 extra questions: FU37 ("per customer?") went to the docs agent.
- These numbers measure fit to the questions the fixes came from. The held-out numbers above are the fair test.

## Power run

The held-out retest had too few questions to settle anything: 27 split decisions (questions only one system got right) can't separate two systems a few points apart. So the comparison was rerun at the size McNemar's test needs: **1,400 paired data questions**. That is enough for 90% power to detect a 3.4-point difference, by Connor's formula at the held-out set's split share.

The 1,400 are the 175 hand-written held-out questions plus 1,225 generated from templates in the same mix of question types and wording styles. The plan, sample-size rule and tests were written and hashed before any answer (`POWER_RUN_PLAN.md`).

- **Assistant:** the fixed code (`98bcfbe`).
- **Analyst:** semantic model r8 (r7 plus a `QUALIFY RANK()` rule for its FLOAT ranking bug).
- **Cost:** about 82 Analyst credits for the 1,225 new questions.

### Result

| | Segment Assistant | Cortex Analyst |
|---|---|---|
| Right, of 1,400 | 1,231 = **87.9%** (95% CI 86.1–89.5%) | 1,313 = **93.8%** (92.4–94.9%) |
| Only this system right | 71 | **153** |

- **Analyst is ahead by 5.9 points.**
  - 95% interval: +3.8 to +8.0.
  - Exact McNemar p = 0.00000004.
- **The two aren't equivalent.** The 90% interval (+4.1 to +7.6) lies wholly outside the ±3.4-point margin.
- **On the 175 hand-written questions alone they were level, 14 to 13.** That isn't a contradiction. With only 27 split decisions, the exact 95% interval for Analyst's share of them runs from about 32% to 71%. The full run's 68% sits inside it.
- **Achieved power.**
  - Split decisions came to 16.0% of questions, slightly more than the 15.4% planned.
  - At that share the run's power at 3.4 points was 89.0% by Connor's formula and 87.9% by the exact test.

### Where the gap comes from

| Question type | n | Assistant | Analyst | Split (Asst / Analyst) | p | Holm p |
|---|---|---|---|---|---|---|
| Can't be answered (should decline) | 128 | 68.8% | 92.2% | 7 / 37 | 0.000005 | 0.00005 |
| Model scores | 64 | 73.4% | 95.3% | 3 / 17 | 0.003 | 0.021 |
| Breakdowns | 64 | 81.2% | 100.0% | 0 / 12 | 0.0005 | 0.004 |
| Rankings | 160 | 92.5% | 97.5% | 2 / 10 | 0.039 | 0.23 |
| Follow-ups | 224 | 87.5% | 92.9% | 13 / 25 | 0.073 | 0.37 |
| Look up one figure | 256 | 94.9% | 98.8% | 3 / 13 | 0.021 | 0.15 |
| Comparisons | 128 | 97.7% | 99.2% | 1 / 3 | 0.63 | 1 |
| Ambiguous (should ask) | 152 | 88.8% | 87.5% | 17 / 15 | 0.86 | 1 |
| Totals and shares | 176 | 87.5% | 85.8% | 19 / 16 | 0.74 | 1 |
| Filters | 48 | 89.6% | 87.5% | 6 / 5 | 1.0 | 1 |

- **Questions the data can't answer are the biggest gap.**
  - Typical questions ask for a country other than the UK, or a CRM field: email open rate, acquisition source, industry.
  - Analyst says it doesn't hold that. The Assistant answered all 40 of its misses here with the nearest figure it had instead, such as all of a segment's customers for "how many are in Lithuania".
  - The missing-data decline in `9298527` doesn't catch these phrasings.
- **Model scores.** The Assistant confused similarly named schemes. "Outcome tiers at grid size (9–12 segments)" got the 3–4 segment tiers, and "the value-axis-only scheme" was answered from the value axis's k-selection table.
- **Breakdowns.** Asked for each segment's value, the Assistant usually gave one figure for the whole group, or whole-tier totals. Four times it gave "per segment only" placeholders.
- **Leaving out the unanswerable questions, Analyst still leads**, 116 split decisions to 64 (p = 0.0001).
- **Analyst's misses are mostly incomplete answers.** It gives segment rows where a tier or combined total was asked for.
- **Typos** account for 15 of the Assistant's 169 misses: a misspelt tier or style word drops that filter.

### Checks

| Check | Only Assistant right | Only Analyst right | p |
|---|---|---|---|
| All 1,400 (primary) | 71 | 153 | 0.00000004 |
| Follow-ups grouped with their first question (clustered, Durkalski) | 71 | 153 | 0.0000001 |
| First questions only | 58 | 128 | 0.0000003 |
| Hand-written held-out 175 only | 13 | 14 | 1.0 |
| Generated 1,225 only | 58 | 139 | 0.000000007 |
| Assistant's unaudited passes counted right (the plan as written) | 74 | 139 | 0.00001 |
| Graded by the machine first pass instead of the read | 131 | 150 | 0.28 |

- **The result survives every check but the last.** The machine first pass misses errors in both directions, so the result rests on reading the answers.
  - It agreed with the read on 95.1% of the Assistant's answers (Cohen's kappa 0.78) and 93.0% of Analyst's (kappa 0.57).
  - It also has a bug: it passed any eta-squared figure the Assistant gave, right or wrong.

**Caveats.**
- **The read wasn't blind.** It covered 2,800 answers, and the reader knew which system wrote each one. The same rules applied to both systems, and a script checked them for consistency afterwards. `POWER_RUN_PLAN.md` lists every departure from the plan.
- **Each system answered each question once.**
- **The generated questions come from templates.** Real users may phrase things in ways neither set covers.
