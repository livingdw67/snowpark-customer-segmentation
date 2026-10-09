-- Monthly re-scoring of customer segments.
--
-- Prerequisites: 05_deploy_procedures.py (creates ML.RESCORE_CUSTOMERS) and
-- 07_run_production.py (fits the production models the procedure loads from @ML.MODELS).
--
-- Each run rebuilds MARTS.CUSTOMER_FEATURES for the snapshot after the latest transaction,
-- scores it with the frozen production models and writes hybrid rows under run_id 'rescore'.
-- ML.CUSTOMER_SEGMENTS_CURRENT prefers those rows, so BI tools pick up the new month
-- automatically. The dataset is static, so here every run re-derives 2011-12-10 and the
-- procedure's result reports how many customers match the production run (expect all).
-- The Task's owner role needs EXECUTE TASK on the account.

USE SCHEMA CUSTOMER_SEGMENTATION.ML;

CREATE OR REPLACE TASK RESCORE_SEGMENTS_MONTHLY
  WAREHOUSE = COMPUTE_WH                    -- your SNOWFLAKE_WAREHOUSE
  SCHEDULE = 'USING CRON 0 6 1 * * UTC'     -- 06:00 UTC on the 1st of each month
  COMMENT = 'Re-score customers with the frozen production segmentation models'
AS
  CALL RESCORE_CUSTOMERS('production');

ALTER TASK RESCORE_SEGMENTS_MONTHLY RESUME;

-- EXECUTE TASK RESCORE_SEGMENTS_MONTHLY;                       -- run once now
-- SELECT name, state, return_value, error_message
--   FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(TASK_NAME => 'RESCORE_SEGMENTS_MONTHLY'))
--  ORDER BY scheduled_time DESC;
-- ALTER TASK RESCORE_SEGMENTS_MONTHLY SUSPEND;                 -- stop it when you're done
