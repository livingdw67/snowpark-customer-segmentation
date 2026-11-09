# Methodology

## Findings worth knowing before using this

1. **There are no natural clusters.** HDBSCAN labels 61–78% of customers as noise on the full feature set, and k-means silhouettes are 0.2–0.3. The segments are a *useful, stable* partition of a continuum, chosen for predictive value and stability, not "discovered tribes". The segment map in the README shows the overlap.
2. **Nothing much beats plain RFM at predicting who buys again.** The hybrid's AUC is 0.02 below the RFM baseline, and even the unsegmented model gains only 0.03. Buying again is driven mostly by recency. The hybrid earns its place on *revenue concentration* (+15 points over RFM) and on *buying style*, not on this metric.
3. **The hybrid's lead over the simpler grid is thin.** Option B's value × behavior grid uses no gradient boosting and separates buying styles just as well (η² 0.34). It captures 66% of next-quarter revenue in its top 20%, against the hybrid's 71%. Across the two backtests the hybrid ranges from 67.5% to 74.3% and the grid from 65.7% to 65.9%, so the 5-point gain is suggestive, not established. If the outcome model's extra machinery matters, the grid is the simpler choice; more backtest folds would settle it.
4. **At 3–4 segments, C collapses to spend bands** (e.g. `net_revenue > 3606`). The model is saying that trailing 12-month spend dominates next-quarter value. The behavior axis is what adds new information.
5. **The £ forecasts are only trustworthy for seasons the model has seen.** C is trained on all past quarters pooled, with the snapshot month as a feature. The fold that ended in the holiday quarter had never seen one, and it under-forecast revenue by 43%. The rankings held up, so use `expected_revenue_next_quarter` to *rank and size* customers, not as a budget number. Adding a September 2010 snapshot, with a shorter lookback because the data starts in December 2009, would give the model one holiday quarter to learn from. That isn't done yet.
6. **Some grid cells are small.** "Top spenders / Holiday buyers" and "Top spenders / Small-basket buyers" hold 44 and 49 customers. The 5% minimum segment size is enforced per axis, not per cell. For a wholesaler, 44 top-spender holiday accounts holding about 13% of predicted value is a key-account list. For broad campaigns, merge small cells into their value tier.
7. **One selection rule was changed after seeing results.** k for each axis is chosen by the best score among stable candidates, preferring fewer segments within a tie tolerance. The tolerance was widened from 0.005 to 0.01 after production picked 7 value segments over 4 for a 0.006 R² gain, with stability just over the bar. Both backtest folds were unaffected. The full k-selection table is in [results/k_selection.csv](../results/k_selection.csv).

## How the segments are built

### Data and cleaning ([03_build_staging.py](../03_build_staging.py))

| Step | Rows | Change |
|---|---|---|
| Raw (both Excel sheets) | 1,067,371 | |
| Remove sheet overlap | 1,044,848 | −22,523 |
| Remove exact duplicates | 1,033,036 | −11,812 |
| Remove stock write-offs (zero/negative-price non-returns) | 1,026,999 | −6,037 |
| Product lines with a customer ID | 794,165 | |

**The two published sheets overlap.** The 2009–10 sheet repeats the first nine days of December 2010 from the 2010–11 sheet, and concatenating them naively double-counts that month. Non-product lines (postage, Amazon fees, manual adjustments, vouchers, test items) are flagged and kept out of customer features.

### Features ([04_build_customer_features.py](../04_build_customer_features.py))

There is one row per customer per quarterly snapshot (Dec 2010 → Dec 2011):
- 15 behavior features over a 365-day lookback
- outcomes over the next 91 days: whether the customer buys again, and their net revenue

Only data before the snapshot date feeds the features. The last snapshot (2011-12-10) has no outcomes; it's "today".

### Option B: two-axis ([segmentation/two_axis.py](../segmentation/two_axis.py))

- **Value axis** (recency, orders, net revenue) and **behavior axis** (units per order, unit price, lines per order, return rate, Q4 revenue share).
- Each axis is a `Pipeline` of log1p → scale → clip ±3 SD → `BisectingKMeans(bisecting_strategy="largest_cluster")`. That strategy keeps splitting the largest group instead of carving out tiny outlier segments.
- **k is chosen per axis.** The value axis uses out-of-fold R² for next-quarter revenue; the behavior axis uses silhouette. Both require bootstrap stability (ARI ≥ 0.85 on 80% subsamples) and every segment holding ≥ 5% of customers.
- **Surrogate decision trees produce plain-English assignment rules.** They reproduce the actual segments 81–88% of the time (cross-validated).

### Option C: outcome-guided ([segmentation/outcome_guided.py](../segmentation/outcome_guided.py))

- **P(buys again):** `HistGradientBoostingClassifier`, calibrated with `CalibratedClassifierCV(FrozenEstimator(booster), method="temperature")` on a held-out, customer-grouped split.
- **Expected revenue:** `HistGradientBoostingRegressor(loss="poisson")`. Most customers spend £0 next quarter, and spend is non-negative and right-skewed.
- **Segments:** a shallow regression tree trained on the readable features to reproduce the *out-of-fold* expected revenue. Each leaf is a segment, and its path is the rule.
- **Training data:** pooled over every past snapshot, with `GroupKFold` / `GroupShuffleSplit` by customer, because the same customer appears in several quarters.

### Evaluation ([segmentation/evaluation.py](../segmentation/evaluation.py))

Every scheme is fit on one snapshot. Its segments' outcome rates are learned from that snapshot's outcomes, then applied to customers' segments at the *next* snapshot and scored against what actually happened. Metrics:
- buys-again AUC
- Spearman correlation with next-quarter revenue
- top-20% revenue capture, with tied segments sharing the cut-off proportionally
- η² for value and behavior features
- rule fidelity
- segment persistence quarter to quarter
- a forecast-bias check that rank metrics miss

### Monthly re-scoring ([sql/rescore_task.sql](../sql/rescore_task.sql))

The `RESCORE_SEGMENTS_MONTHLY` Task calls `ML.RESCORE_CUSTOMERS` on the 1st of each month. It:
1. rebuilds `MARTS.CUSTOMER_FEATURES` for the snapshot after the latest transaction (the same SQL as step 04, from [segmentation/feature_sql.py](../segmentation/feature_sql.py))
2. scores it with the frozen production models from `@ML.MODELS`
3. writes hybrid rows under `run_id = 'rescore'`, which `CUSTOMER_SEGMENTS_CURRENT` prefers over the production run for the same snapshot

The dataset is static, so every run re-derives 2011-12-10. The procedure reports how many customers got the same segment as the production run; all of them should. If you deployed before re-scoring existed, run `python 05_deploy_procedures.py` and `python 07_run_production.py --view-only`; the second recreates the view without retraining.

**Why not the Snowflake Model Registry?** A custom registry model needs `snowflake-ml-python` in its runtime, and every build of it in Snowflake's Anaconda channel (up to 1.51.0) requires `scikit-learn<1.9`. The models need 1.9.1. Until the channel catches up, the models stay as versioned joblib files on a stage, one folder per run.

## scikit-learn: what's used and why

The code is pinned to **1.9.1, the latest stable release**, which is also in Snowflake's Anaconda channel, so local and in-warehouse code run identical versions.

| Feature | Since | Used for |
|---|---|---|
| `manifold.ClassicalMDS` | 1.8 | The segment map |
| `CalibratedClassifierCV(method="temperature")` | 1.8 | Calibrating P(buys again) |
| `frozen.FrozenEstimator` | 1.6 | Calibrating without refitting the booster on calibration data |
| `cluster.HDBSCAN` | 1.3 | Evidence that there are no natural clusters ([exploration/](../exploration/)) |
| `set_output(transform="pandas")` | 1.2 | Named columns through every preprocessing step |
| `cluster.BisectingKMeans` | 1.1 | Both axes of option B |

The experimental callbacks API (new in 1.9) isn't used: none of the clustering or gradient-boosting estimators support it yet.

**On Windows, Smart App Control may block scikit-learn 1.9.1.** On some machines it refuses to load one of the wheel's DLLs (`_libsvm_sparse`). The pipeline itself isn't affected because it runs in Snowflake. To run the tests there, use GitHub Actions or `python tests/run_in_snowflake.py`. The assistant doesn't use scikit-learn.

## Tests

**Tests:** 27 pipeline tests on synthetic customers with planted structure. They check that the methods recover it, that segment codes are ordered by meaning, that every generated rule is true for the customers it describes, that re-scoring builds the same cells as the hybrid, and the metric edge cases. 140 assistant tests:
- every calculation, checked against pandas on the cases the SQL version got wrong;
- the figure checker;
- the agents with a scripted chat model: plan repair, direction and metric corrections, declines, follow-ups, documentation answers, the writing guards;
- a headless run of the app.

None need Ollama or Snowflake. They run on GitHub Actions (`pytest tests`), or inside Snowflake's own runtime with `python tests/run_in_snowflake.py`.
