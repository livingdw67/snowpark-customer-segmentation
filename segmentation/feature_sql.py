"""SQL for MARTS.CUSTOMER_FEATURES, shared by 04_build_customer_features.py (all snapshots)
and the RESCORE_CUSTOMERS procedure (one new snapshot).

Each snapshot looks back 365 days for behavior features and forward 91 days for outcomes.
Only lines before the snapshot date feed the features.
"""
from .features import DATABASE, FEATURE_TABLE

STAGING = f"{DATABASE}.STAGING.TRANSACTIONS"
LOOKBACK_DAYS = 365
HORIZON_DAYS = 91


def features_select(snapshots, last_data_date: str) -> str:
    """SELECT producing one row per customer per snapshot (outcomes NULL past last_data_date)."""
    STG = STAGING
    values = ", ".join(f"('{d}')" for d in snapshots)
    return f"""
WITH snapshots AS (
    SELECT column1::DATE AS snapshot_date
    FROM VALUES {values}
),
lines AS (
    SELECT * FROM {STG}
    WHERE customer_id IS NOT NULL AND line_type IN ('SALE', 'RETURN')
),
window_lines AS (
    SELECT s.snapshot_date, l.*
    FROM snapshots s
    JOIN lines l
      ON l.invoice_date <  s.snapshot_date
     AND l.invoice_date >= DATEADD(day, -{LOOKBACK_DAYS}, s.snapshot_date)
),
behavior AS (
    SELECT
        snapshot_date,
        customer_id,
        DATEDIFF(day, MAX(IFF(line_type = 'SALE', invoice_date, NULL)), snapshot_date) AS recency_days,
        COUNT(DISTINCT IFF(line_type = 'SALE', invoice_id, NULL))                    AS orders,
        COUNT(DISTINCT IFF(line_type = 'SALE', invoice_date, NULL))                  AS purchase_days,
        COUNT(DISTINCT IFF(line_type = 'SALE', DATE_TRUNC('month', invoice_date), NULL)) AS active_months,
        SUM(IFF(line_type = 'SALE', line_amount, 0))                                  AS gross_revenue,
        -SUM(IFF(line_type = 'RETURN', line_amount, 0))                               AS returned_value,
        SUM(IFF(line_type = 'SALE', quantity, 0))                                     AS units,
        COUNT_IF(line_type = 'SALE')                                                  AS sale_lines,
        COUNT(DISTINCT IFF(line_type = 'SALE', stock_code, NULL))                     AS distinct_products,
        SUM(IFF(line_type = 'SALE' AND MONTH(invoice_date) >= 10, line_amount, 0))    AS q4_revenue,
        DATEDIFF(day, MIN(IFF(line_type = 'SALE', invoice_date, NULL)),
                      MAX(IFF(line_type = 'SALE', invoice_date, NULL)))               AS active_span_days,
        MODE(country)                                                                 AS country
    FROM window_lines
    GROUP BY 1, 2
    HAVING orders > 0
),
first_purchase AS (
    SELECT s.snapshot_date, l.customer_id, MIN(l.invoice_date) AS first_purchase_date
    FROM snapshots s
    JOIN lines l ON l.invoice_date < s.snapshot_date AND l.line_type = 'SALE'
    GROUP BY 1, 2
),
outcomes AS (
    SELECT
        s.snapshot_date,
        l.customer_id,
        SUM(l.line_amount)                                               AS next_net_revenue,
        COUNT(DISTINCT IFF(l.line_type = 'SALE', l.invoice_id, NULL))   AS next_orders
    FROM snapshots s
    JOIN lines l
      ON l.invoice_date >= s.snapshot_date
     AND l.invoice_date <  DATEADD(day, {HORIZON_DAYS}, s.snapshot_date)
    GROUP BY 1, 2
)
SELECT
    b.snapshot_date,
    b.customer_id,
    b.country,
    b.country = 'United Kingdom'                                     AS is_uk,
    -- Recency / frequency / monetary
    b.recency_days,
    b.orders,
    b.purchase_days,
    b.active_months,
    ROUND(b.gross_revenue - b.returned_value, 2)                     AS net_revenue,
    ROUND(b.gross_revenue, 2)                                        AS gross_revenue,
    -- Basket shape (wholesale vs. retail buying)
    ROUND(b.gross_revenue / b.orders, 2)                             AS avg_order_value,
    ROUND(b.units / b.orders, 2)                                     AS avg_units_per_order,
    ROUND(b.sale_lines / b.orders, 2)                                AS avg_lines_per_order,
    ROUND(b.gross_revenue / NULLIF(b.units, 0), 4)                   AS avg_unit_price,
    b.distinct_products,
    -- Returns, seasonality, cadence, tenure
    ROUND(LEAST(b.returned_value / NULLIF(b.gross_revenue, 0), 1), 4) AS return_rate,
    ROUND(b.q4_revenue / NULLIF(b.gross_revenue, 0), 4)               AS q4_revenue_share,
    IFF(b.purchase_days > 1, ROUND(b.active_span_days / (b.purchase_days - 1), 1), NULL)
                                                                     AS avg_days_between_purchases,
    DATEDIFF(day, f.first_purchase_date, b.snapshot_date)            AS tenure_days,
    -- Outcomes over the next {HORIZON_DAYS} days (NULL where the window runs past the data)
    IFF(DATEADD(day, {HORIZON_DAYS - 1}, b.snapshot_date) <= '{last_data_date}',
        COALESCE(o.next_orders, 0) > 0, NULL)                       AS next_active,
    IFF(DATEADD(day, {HORIZON_DAYS - 1}, b.snapshot_date) <= '{last_data_date}',
        COALESCE(o.next_orders, 0), NULL)                           AS next_orders,
    IFF(DATEADD(day, {HORIZON_DAYS - 1}, b.snapshot_date) <= '{last_data_date}',
        ROUND(COALESCE(o.next_net_revenue, 0), 2), NULL)            AS next_net_revenue
FROM behavior b
JOIN first_purchase f USING (snapshot_date, customer_id)
LEFT JOIN outcomes o USING (snapshot_date, customer_id)
"""


def create_features_sql(snapshots, last_data_date: str) -> str:
    return f"CREATE OR REPLACE TABLE {FEATURE_TABLE} AS\n{features_select(snapshots, last_data_date).lstrip()}"


def replace_snapshot_sql(snapshot: str, last_data_date: str) -> list[str]:
    """Statements that rebuild a single snapshot in place (same columns, same order)."""
    return [
        f"DELETE FROM {FEATURE_TABLE} WHERE snapshot_date = '{snapshot}'",
        f"INSERT INTO {FEATURE_TABLE} SELECT * FROM ({features_select([snapshot], last_data_date)})",
    ]
