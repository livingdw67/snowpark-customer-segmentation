# Power run: analysis plan

Written 2026-10-16, before any system answered a question in `questions_v3.json`. The sample-size rule, tests and grading rules below are fixed before the data exists, so the result can't be steered by choices made after seeing it.

## Question

On data questions, is the fixed Segment Assistant more or less accurate than Cortex Analyst?

## Systems

| | Version |
|---|---|
| Segment Assistant | repo branch `practice-eval-fixes` at `98bcfbe`, offline mode (DuckDB over `results/*.csv`), Ollama `qwen2.5:7b`, temperature 0 |
| Cortex Analyst | REST API with `segmentation_r8.yaml`: r7 plus one rule, rank with `QUALIFY RANK()` rather than `= (SELECT MAX(...))`, which matches nothing on FLOAT columns |

Neither system changes from the first answer to the last. Any question asked twice is reported.

## Sample size

Paired questions = the 175 held-out data questions (`questions_v2.json`, hand-written) plus N new generated ones (`questions_v3.json`, `questions_v3.py`, seed 20261016). The new questions follow the held-out set's mix of data-question types and wording styles.

**Rule.** The total is the number of paired questions McNemar's test needs to detect a 3.4-point difference in accuracy with 80% power at a two-sided 5% level, by Connor's (1987) formula. It uses the share of held-out questions on which the fixed Assistant and Analyst r8 disagree, and is rounded up to the next 25. 3.4 points is the gap the first round showed between the two systems, and the planning figure quoted then (about 1,200 questions).

## Analysis

- **Primary:** an exact two-sided McNemar test on every paired question. The effect is Analyst minus Assistant accuracy, with Newcombe's (method 10) 95% interval.
- **Secondary, equivalence:** two one-sided tests with a margin of ±3.4 points. The systems count as equivalent at the 5% level if the 90% Newcombe interval lies inside (−3.4, +3.4).
- **Checks:**
  - a clustered McNemar test grouping each follow-up with its first question (Durkalski 2003);
  - first questions only;
  - grades from the machine first pass instead of the read;
  - the hand-written 175 and the generated questions separately;
  - by question type;
  - Cohen's kappa between the first pass and the read on the audited answers.
- **Achieved power:** reported for the observed discordance rate, at 3.4 points.

## Grading

Verdicts as before: correct, wrong number, wrong claim, unhelpful, declined-but-answerable, answered-instead-of-declining, misrouted, unsafe, error.

- **Machine first pass** on every answer (`grade.py auto`).
- **Read in full:**
  - every answer the first pass fails or can't check;
  - every ambiguous and unanswerable answer.
- **Audit:** a random 10% of first-pass passes per system (seed 2026). If more than 2% of a system's audited passes turn out wrong, every pass for that system is read.
- **Ambiguous questions:** correct if the answer gives one of the listed readings and says which, or asks which one was meant.
- **Unanswerable questions:** correct if the answer says the data doesn't hold it. For the two-year spend question, a next-quarter figure with that caveat also passes.

## Sample size, computed from the rule

Analyst r8 on the 175 held-out data questions: 161 right (r7: 152). The fixed Assistant got 160. The two disagree on 27 questions: the Assistant alone right on 13, Analyst alone on 14. That's a discordance rate of 15.4% (r7: 19.4%).

Connor's formula at 3.4 points, 80% power and a two-sided 5% level gives 1,045.2 paired questions, rounded up to **1,050**. That means **875 new questions**, costing about 59 Analyst credits.

The total is lower than the earlier "about 1,200" because fixing Analyst's FLOAT bug removed disagreements, and fewer disagreements mean a smaller test reaches the same power.

## Amendment: 90% power (2026-10-17T00:47Z)

The total is raised to the size the same rule gives at **90%** power: 1,398.2, rounded up to **1,400** paired questions. That's 350 more than above, generated as `questions_v3b.json` (seed 20261017, IDs starting Y), about 23 more Analyst credits. Every other part of the plan is unchanged.

The change was decided while both systems were still answering `questions_v3.json`, and before any of those answers was graded or compared, so it can't have been steered by the result. At 1,400 questions the 95% interval on the difference is expected to be about ±2.1 points wide, against ±2.4 at 1,050.

## After the run: where it departed from the plan (2026-10-20)

Written after grading. Each item says what changed and how it could affect the result. The result is in the README under **Power run**.

1. **Every first-pass pass was read, for both systems.**
   - **Analyst:** the plan required this. Its pooled audit (v3 and v3b) found 2 of 87 passes wrong (XFU084, YTS030), which is 2.3%, over the 2% limit.
   - **Assistant:** its audit found 1 of 87 wrong, so the plan didn't require it. Its passes were read anyway, so that errors missed in the sample wouldn't count against Analyst alone.
   - **What the full read found:** 17 more wrong Assistant passes (18 of 862 in all, 2.1%) and 3 more wrong Analyst passes (5 of 864).
   - **As the plan was written** (the Assistant's unaudited passes counted right), the result is 74 vs 139 split decisions, p = 0.00001. Analyst − Assistant is +4.6 points, 95% interval +2.6 to +6.7, so the conclusion is the same. This is reported as a check.
2. **Kappa** was computed on every read answer the first pass could check, not only the audited ones, because all of them were read.
3. **The read was not blind.** The reader knew which system wrote each answer. The rules below were applied to both systems, and checked afterwards by script for cases treated differently.
4. **One question was asked twice.** XLK047's first Analyst request failed with a dropped connection and returned no answer (`results/v3_analyst_transport_errors.jsonl`). It was asked once more, and that answer was graded.
5. **Grading rules settled during the read** (they apply to both systems):
   - A "which is highest / lowest" question answered with the list and no pick is **unhelpful**.
   - A one-row "ranking" that states a false superlative is a **wrong claim**. For example, the answer "C1-B3 has the lowest net revenue" when C1-B3 has the highest.
   - A superset that labels the asked values is **correct**. YBD013 was first marked unhelpful and then revised to correct under this rule.
   - "Top / bottom 3 by recency" accepts either reading, most recent or most lapsed (XRK043, XRK100).
6. **First-pass bug, left as registered.** The Assistant labels eta squared "(eta squared, 0-1)". The first pass reads that "0" and "1" as whole numbers with ±0.5 tolerance, so any value under 1.5 matched. The read caught the three wrong answers this let through (XME031, XFU064, YME006). The machine-graded check row uses the first pass as it was.
7. **By-type tests carry Holm's adjustment.** The plan listed "by question type" without naming an adjustment. Ten types are ten tests.
8. **p-values are stored to three significant figures.** They had been stored to four decimal places, which rounded the main result to 0.
