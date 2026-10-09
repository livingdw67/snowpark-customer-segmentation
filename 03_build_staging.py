"""Clean RAW.TRANSACTIONS into STAGING.TRANSACTIONS and record what each rule removed.

Rules, in order:
  1. The 2009-2010 sheet repeats the first days of the 2010-2011 sheet (December 2010).
     Rows from the earlier sheet on or after the later sheet's first timestamp are dropped.
  2. Exact duplicate rows within a sheet are dropped (same invoice, item, quantity, time, price).
  3. Each line is typed: product sale, product return (invoice prefix C), or non-product
     (postage, fees, manual adjustments, vouchers, test items). Non-product lines are kept but
     flagged, so customer features can use product lines only.
  4. Zero/negative-price non-return lines are stock write-offs, not sales, and are dropped.
Lines without a CUSTOMER_ID stay in staging (they count toward revenue totals) but cannot be
segmented; the customer mart filters them out.
"""
from common import DATABASE, get_session, run_sql

RAW = f"{DATABASE}.RAW.TRANSACTIONS"
STG = f"{DATABASE}.STAGING.TRANSACTIONS"
AUDIT = f"{DATABASE}.STAGING.CLEANING_AUDIT"

# Real products have 5-digit codes (with optional letter suffix); DCGS* and SP1002 are
# webshop products with their own code scheme. Everything else is postage, fees or adjustments.
IS_PRODUCT = "(REGEXP_LIKE(stockcode, '^[0-9]{5}.*') OR stockcode LIKE 'DCGS%' OR stockcode = 'SP1002')"

STAGING_SQL = f"""
CREATE OR REPLACE TABLE {STG} AS
WITH later_sheet_start AS (
    SELECT MIN(invoicedate) AS ts FROM {RAW} WHERE sourcesheet = 'Year 2010-2011'
),
deoverlapped AS (
    SELECT r.*
    FROM {RAW} r, later_sheet_start s
    WHERE NOT (r.sourcesheet = 'Year 2009-2010' AND r.invoicedate >= DATE_TRUNC('day', s.ts))
),
deduped AS (
    SELECT * FROM deoverlapped
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY invoice, stockcode, description, quantity, invoicedate, price, customerid, country
        ORDER BY sourcesheet) = 1
)
SELECT
    invoice                                   AS invoice_id,
    stockcode                                 AS stock_code,
    description,
    quantity,
    price                                     AS unit_price,
    quantity * price                          AS line_amount,
    invoicedate                               AS invoice_ts,
    TO_DATE(invoicedate)                      AS invoice_date,
    customerid::VARCHAR                       AS customer_id,
    country,
    CASE
        WHEN NOT {IS_PRODUCT}                 THEN 'NON_PRODUCT'
        WHEN invoice LIKE 'C%'                THEN 'RETURN'
        ELSE 'SALE'
    END                                       AS line_type,
    sourcesheet                               AS source_sheet
FROM deduped
WHERE NOT (invoice NOT LIKE 'C%' AND (price <= 0 OR quantity <= 0))
"""

AUDIT_SQL = f"""
CREATE OR REPLACE TABLE {AUDIT} AS
WITH s AS (SELECT MIN(invoicedate) AS ts FROM {RAW} WHERE sourcesheet = 'Year 2010-2011'),
counts AS (
    SELECT 1 AS step, 'raw rows' AS stage, (SELECT COUNT(*) FROM {RAW}) AS rows_remaining
    UNION ALL SELECT 2, 'after removing sheet overlap',
        (SELECT COUNT(*) FROM {RAW}, s
         WHERE NOT (sourcesheet = 'Year 2009-2010' AND invoicedate >= DATE_TRUNC('day', s.ts)))
    UNION ALL SELECT 3, 'after removing exact duplicates',
        (SELECT COUNT(*) FROM (
            SELECT DISTINCT invoice, stockcode, description, quantity, invoicedate, price, customerid, country
            FROM {RAW}, s
            WHERE NOT (sourcesheet = 'Year 2009-2010' AND invoicedate >= DATE_TRUNC('day', s.ts))))
    UNION ALL SELECT 4, 'after removing write-offs (staging)', (SELECT COUNT(*) FROM {STG})
    UNION ALL SELECT 5, 'product lines with a customer id',
        (SELECT COUNT(*) FROM {STG} WHERE line_type <> 'NON_PRODUCT' AND customer_id IS NOT NULL)
)
SELECT step, stage, rows_remaining,
       rows_remaining - LAG(rows_remaining) OVER (ORDER BY step) AS change
FROM counts ORDER BY step
"""


def build_staging():
    session = get_session()
    run_sql(session, STAGING_SQL)
    run_sql(session, AUDIT_SQL)
    print(session.table(AUDIT).sort("STEP").to_pandas().to_string(index=False))
    print(session.sql(f"""
        SELECT line_type, COUNT(*) AS lines, ROUND(SUM(line_amount)) AS amount,
               COUNT_IF(customer_id IS NULL) AS lines_without_customer
        FROM {STG} GROUP BY 1 ORDER BY 1
    """).to_pandas().to_string(index=False))
    session.close()


if __name__ == "__main__":
    build_staging()
