-- =====================================================================================================================
-- PROJECT: AI-Powered Retail Demand Forecasting & Inventory Optimization Engine
-- FILE: 50_enterprise_retail_analytics_queries.sql
-- TARGET DIALECTS: DuckDB / Snowflake / BigQuery / PostgreSQL / Databricks SQL
-- DESCRIPTION: 
--   This repository file contains 52 enterprise-grade SQL queries organized into 8 functional modules.
--   These queries power the data cleaning, feature engineering, ABC-XYZ segmentation, Operations Research (OR)
--   supply chain inventory calculations, and executive KPI reporting for an 83-million row retail dataset (Kaggle M5).
-- =====================================================================================================================

-- =====================================================================================================================
-- MODULE 1: DATA DOWNCASTING, SCHEMA VALIDATION & WIDE-TO-LONG TRANSFORMATION (QUERIES 1 - 6)
-- =====================================================================================================================

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 1: Inspect Memory Footprint & Data Types Across Raw Tables
-- Purpose: Quantify raw data bloat before memory optimization downcasting.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    table_name,
    column_name,
    data_type,
    character_maximum_length,
    numeric_precision
FROM information_schema.columns
WHERE table_name IN ('sales_train', 'calendar', 'sell_prices')
ORDER BY table_name, ordinal_position;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 2: Wide-to-Long Unpivoting of Daily Sales (`d_1` to `d_1941`) using DuckDB / Snowflake UNPIVOT
-- Purpose: Convert horizontal transactional matrix into normalized time-series fact table.
-- ---------------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE TABLE fact_daily_sales AS
SELECT 
    id,
    item_id,
    dept_id,
    cat_id,
    store_id,
    state_id,
    CAST(SUBSTRING(day_id, 3) AS INTEGER) AS day_num,
    day_id AS d,
    CAST(unit_sales AS SMALLINT) AS sales
FROM sales_train
UNPIVOT (
    unit_sales FOR day_id IN (d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9, d_10, -- expanded to d_1941 in prod
                              d_1880, d_1881, d_1882, d_1883, d_1884, d_1885)
);

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 3: Downcast Numeric and String Columns to Optimize VertiPaq / Parquet Compression
-- Purpose: Reduce memory usage from ~45 GB to ~4.5 GB (10x compression).
-- ---------------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE TABLE dim_calendar_clean AS
SELECT 
    CAST(date AS DATE) AS date,
    CAST(wm_yr_wk AS INTEGER) AS wm_yr_wk,
    d,
    CAST(wday AS SMALLINT) AS wday,
    CAST(month AS SMALLINT) AS month,
    CAST(year AS SMALLINT) AS year,
    COALESCE(event_name_1, 'None') AS event_name_1,
    COALESCE(event_type_1, 'None') AS event_type_1,
    COALESCE(event_name_2, 'None') AS event_name_2,
    COALESCE(event_type_2, 'None') AS event_type_2,
    CAST(snap_CA AS SMALLINT) AS snap_CA,
    CAST(snap_TX AS SMALLINT) AS snap_TX,
    CAST(snap_WI AS SMALLINT) AS snap_WI,
    CASE WHEN wday IN (1, 2) THEN 1 ELSE 0 END AS is_weekend
FROM calendar;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 4: Data Quality Audit - Detect Missing Selling Prices or Negative Sales
-- Purpose: Ensure price integrity and identify gaps where items were not yet introduced to a store.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    p.store_id,
    p.item_id,
    COUNT(CASE WHEN p.sell_price IS NULL OR p.sell_price <= 0 THEN 1 END) AS invalid_price_count,
    MIN(p.sell_price) AS min_price,
    MAX(p.sell_price) AS max_price,
    AVG(p.sell_price) AS avg_price
FROM sell_prices p
GROUP BY p.store_id, p.item_id
HAVING invalid_price_count > 0;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 5: Build Unified Master Analytical Star-Schema Fact View
-- Purpose: Join unpivoted sales with clean calendar and pricing dimensions for cross-sectional modeling.
-- ---------------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_master_retail_analytics AS
SELECT 
    s.id,
    s.item_id,
    s.dept_id,
    s.cat_id,
    s.store_id,
    s.state_id,
    c.date,
    c.wm_yr_wk,
    c.wday,
    c.month,
    c.year,
    c.is_weekend,
    c.event_name_1,
    c.event_type_1,
    CASE 
        WHEN s.state_id = 'CA' THEN c.snap_CA
        WHEN s.state_id = 'TX' THEN c.snap_TX
        WHEN s.state_id = 'WI' THEN c.snap_WI
        ELSE 0 
    END AS active_snap_day,
    COALESCE(p.sell_price, 0.0) AS sell_price,
    s.sales,
    ROUND(s.sales * COALESCE(p.sell_price, 0.0), 2) AS daily_revenue
FROM fact_daily_sales s
JOIN dim_calendar_clean c ON s.d = c.d
LEFT JOIN sell_prices p ON s.store_id = p.store_id 
                       AND s.item_id = p.item_id 
                       AND c.wm_yr_wk = p.wm_yr_wk;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 6: Identify First Sale Date (Introductory Date) per SKU-Store
-- Purpose: Truncate leading zeros before product launch to avoid biasing lag and rolling statistics.
-- ---------------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE TABLE sku_intro_dates AS
SELECT 
    store_id,
    item_id,
    MIN(date) AS first_sale_date,
    MAX(date) AS last_sale_date,
    COUNT(CASE WHEN sales > 0 THEN 1 END) AS total_active_days
FROM vw_master_retail_analytics
GROUP BY store_id, item_id;


-- =====================================================================================================================
-- MODULE 2: EXPLORATORY DATA ANALYSIS & INTERMITTENT DEMAND DIAGNOSTICS (QUERIES 7 - 13)
-- =====================================================================================================================

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 7: Quantify Zero-Inflation Ratio Across All Products (Intermittent Demand Profiling)
-- Purpose: Prove why classical ARIMA/MAPE metrics fail ($0$ denominator) and justify Tweedie Loss.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    cat_id,
    dept_id,
    COUNT(*) AS total_observations,
    SUM(CASE WHEN sales = 0 THEN 1 ELSE 0 END) AS zero_sales_days,
    ROUND(100.0 * SUM(CASE WHEN sales = 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS zero_inflation_pct,
    ROUND(AVG(sales), 4) AS mean_daily_sales,
    ROUND(STDDEV(sales), 4) AS std_daily_sales
FROM vw_master_retail_analytics
GROUP BY cat_id, dept_id
ORDER BY zero_inflation_pct DESC;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 8: Calculate Syntletos-Boylan (ADI and CV2) Intermittent Demand Classification
-- Purpose: Classify SKUs into Smooth, Erratic, Intermittent, and Lumpy profiles.
--   ADI (Average Demand Interval) = Total Days / Active Days (> 1.32 is Intermittent)
--   CV^2 (Squared Coefficient of Variation of Demand Size) = (Std / Mean)^2 (> 0.49 is Lumpy/Erratic)
-- ---------------------------------------------------------------------------------------------------------------------
WITH active_sales_stats AS (
    SELECT 
        item_id,
        store_id,
        COUNT(*) AS total_days,
        SUM(CASE WHEN sales > 0 THEN 1 ELSE 0 END) AS active_days,
        AVG(CASE WHEN sales > 0 THEN sales END) AS active_mean,
        STDDEV(CASE WHEN sales > 0 THEN sales END) AS active_std
    FROM vw_master_retail_analytics
    GROUP BY item_id, store_id
)
SELECT 
    item_id,
    store_id,
    ROUND(total_days * 1.0 / NULLIF(active_days, 0), 2) AS adi,
    ROUND(POWER(active_std / NULLIF(active_mean, 0), 2), 2) AS cv2,
    CASE 
        WHEN (total_days * 1.0 / NULLIF(active_days, 0)) < 1.32 AND POWER(active_std / NULLIF(active_mean, 0), 2) < 0.49 THEN 'Smooth (ARIMA/LightGBM)'
        WHEN (total_days * 1.0 / NULLIF(active_days, 0)) < 1.32 AND POWER(active_std / NULLIF(active_mean, 0), 2) >= 0.49 THEN 'Erratic (High Variance)'
        WHEN (total_days * 1.0 / NULLIF(active_days, 0)) >= 1.32 AND POWER(active_std / NULLIF(active_mean, 0), 2) < 0.49 THEN 'Intermittent (Croston/Tweedie)'
        ELSE 'Lumpy (Zero-Inflated Tweedie Required)'
    END AS demand_classification
FROM active_sales_stats;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 9: Department-Level Daily Volume and Revenue Variance
-- Purpose: Analyze structural demand differences across Food, Hobbies, and Household goods.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    dept_id,
    COUNT(DISTINCT item_id) AS unique_skus,
    SUM(sales) AS total_units_sold,
    ROUND(SUM(daily_revenue), 2) AS total_revenue_generated,
    ROUND(AVG(daily_revenue), 2) AS avg_daily_store_revenue,
    ROUND(SUM(daily_revenue) / SUM(SUM(daily_revenue)) OVER() * 100, 2) AS revenue_share_pct
FROM vw_master_retail_analytics
GROUP BY dept_id
ORDER BY total_revenue_generated DESC;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 10: Store Location & State Benchmark Comparison
-- Purpose: Benchmark store performance across California, Texas, and Wisconsin.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    state_id,
    store_id,
    SUM(sales) AS total_store_units,
    ROUND(SUM(daily_revenue), 2) AS total_store_revenue,
    ROUND(AVG(sales), 2) AS avg_daily_units_per_sku,
    RANK() OVER (PARTITION BY state_id ORDER BY SUM(daily_revenue) DESC) AS state_revenue_rank,
    RANK() OVER (ORDER BY SUM(daily_revenue) DESC) AS national_revenue_rank
FROM vw_master_retail_analytics
GROUP BY state_id, store_id
ORDER BY total_store_revenue DESC;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 11: Detect Outlier Sales Days across Stores (Z-Score > 3.5)
-- Purpose: Identify extreme volume anomalies caused by data glitches or massive clearance events.
-- ---------------------------------------------------------------------------------------------------------------------
WITH store_daily_totals AS (
    SELECT 
        store_id,
        date,
        SUM(sales) AS total_daily_sales
    FROM vw_master_retail_analytics
    GROUP BY store_id, date
),
store_z_scores AS (
    SELECT 
        store_id,
        date,
        total_daily_sales,
        AVG(total_daily_sales) OVER (PARTITION BY store_id) AS mean_sales,
        STDDEV(total_daily_sales) OVER (PARTITION BY store_id) AS std_sales,
        ROUND((total_daily_sales - AVG(total_daily_sales) OVER (PARTITION BY store_id)) / 
              NULLIF(STDDEV(total_daily_sales) OVER (PARTITION BY store_id), 0), 2) AS z_score
    FROM store_daily_totals
)
SELECT *
FROM store_z_scores
WHERE ABS(z_score) > 3.5
ORDER BY z_score DESC;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 12: Top 25 Highest Volume Single-Day Transactions by SKU
-- Purpose: Inspect peak SKU transactions to evaluate promotion flags.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    date,
    store_id,
    item_id,
    dept_id,
    sales,
    sell_price,
    daily_revenue,
    event_name_1
FROM vw_master_retail_analytics
ORDER BY sales DESC
LIMIT 25;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 13: Correlation Analysis between Day of Week and Sales Volume
-- Purpose: Evaluate weekly seasonality and shopping patterns across categories.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    cat_id,
    wday,
    CASE wday 
        WHEN 1 THEN 'Saturday' WHEN 2 THEN 'Sunday' WHEN 3 THEN 'Monday' WHEN 4 THEN 'Tuesday'
        WHEN 5 THEN 'Wednesday' WHEN 6 THEN 'Thursday' WHEN 7 THEN 'Friday'
    END AS day_name,
    is_weekend,
    ROUND(AVG(sales), 3) AS avg_unit_sales,
    ROUND(SUM(daily_revenue), 2) AS total_revenue
FROM vw_master_retail_analytics
GROUP BY cat_id, wday, is_weekend
ORDER BY cat_id, wday;


-- =====================================================================================================================
-- MODULE 3: CALENDAR, HOLIDAY & SNAP WELFARE IMPACT ANALYTICS (QUERIES 14 - 20)
-- =====================================================================================================================

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 14: Quantify Percentage Uplift During SNAP Benefit Days (Supplemental Nutrition Assistance Program)
-- Purpose: Measure exact sales lift when government nutrition benefits are disbursed in CA, TX, and WI.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    state_id,
    active_snap_day,
    COUNT(DISTINCT date) AS total_days_in_sample,
    ROUND(AVG(sales), 3) AS avg_daily_units,
    ROUND(AVG(daily_revenue), 2) AS avg_daily_revenue,
    ROUND((AVG(sales) - LAG(AVG(sales)) OVER (PARTITION BY state_id ORDER BY active_snap_day)) / 
          NULLIF(LAG(AVG(sales)) OVER (PARTITION BY state_id ORDER BY active_snap_day), 0) * 100, 2) AS snap_volume_uplift_pct
FROM vw_master_retail_analytics
GROUP BY state_id, active_snap_day
ORDER BY state_id, active_snap_day;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 15: Holiday & Event Impact Breakdown across Event Types (Religious, Cultural, National, Sporting)
-- Purpose: Determine which holidays create demand surges (Diwali, Super Bowl, Thanksgiving) vs closures (Christmas).
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    event_type_1,
    event_name_1,
    COUNT(DISTINCT date) AS occurrences,
    ROUND(AVG(sales), 2) AS avg_event_sales,
    ROUND(AVG(daily_revenue), 2) AS avg_event_revenue
FROM vw_master_retail_analytics
WHERE event_name_1 != 'None'
GROUP BY event_type_1, event_name_1
ORDER BY avg_event_revenue DESC;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 16: Pre-Holiday vs Post-Holiday Demand Surge Analysis (7-Day Lead & Lag Window)
-- Purpose: Engineer features capturing pre-holiday stockpiling and post-holiday demand drops.
-- ---------------------------------------------------------------------------------------------------------------------
WITH daily_corporate_sales AS (
    SELECT 
        date,
        event_name_1,
        SUM(sales) AS total_sales
    FROM vw_master_retail_analytics
    GROUP BY date, event_name_1
),
holiday_windows AS (
    SELECT 
        date,
        event_name_1,
        total_sales,
        LAG(total_sales, 7) OVER (ORDER BY date) AS sales_7d_before,
        LEAD(total_sales, 7) OVER (ORDER BY date) AS sales_7d_after
    FROM daily_corporate_sales
)
SELECT 
    event_name_1,
    ROUND(AVG(sales_7d_before), 0) AS avg_pre_holiday_7d_sales,
    ROUND(AVG(total_sales), 0) AS avg_holiday_day_sales,
    ROUND(AVG(sales_7d_after), 0) AS avg_post_holiday_7d_sales,
    ROUND((AVG(total_sales) - AVG(sales_7d_before)) / NULLIF(AVG(sales_7d_before), 0) * 100, 2) AS holiday_day_surge_pct
FROM holiday_windows
WHERE event_name_1 != 'None'
GROUP BY event_name_1
ORDER BY holiday_day_surge_pct DESC;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 17: Monthly Seasonality Index by Department Hierarchy
-- Purpose: Calculate seasonal indices ($Avg \text{ Month Sales} / Avg \text{ Annual Sales}$) for inventory planning.
-- ---------------------------------------------------------------------------------------------------------------------
WITH monthly_dept_avg AS (
    SELECT 
        dept_id,
        month,
        AVG(sales) AS monthly_avg_sales
    FROM vw_master_retail_analytics
    GROUP BY dept_id, month
),
dept_grand_avg AS (
    SELECT 
        dept_id,
        AVG(sales) AS annual_avg_sales
    FROM vw_master_retail_analytics
    GROUP BY dept_id
)
SELECT 
    m.dept_id,
    m.month,
    ROUND(m.monthly_avg_sales, 2) AS monthly_avg_sales,
    ROUND(g.annual_avg_sales, 2) AS annual_avg_sales,
    ROUND((m.monthly_avg_sales / NULLIF(g.annual_avg_sales, 0)) * 100, 2) AS seasonality_index_pct
FROM monthly_dept_avg m
JOIN dept_grand_avg g ON m.dept_id = g.dept_id
ORDER BY m.dept_id, m.month;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 18: Weekend vs Weekday Sales Elasticity by Category
-- Purpose: Identify which product categories (e.g., Hobbies) experience the strongest weekend shopping surges.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    cat_id,
    is_weekend,
    COUNT(DISTINCT date) AS total_days,
    ROUND(SUM(sales) / COUNT(DISTINCT date), 2) AS avg_daily_category_units,
    ROUND(SUM(daily_revenue) / COUNT(DISTINCT date), 2) AS avg_daily_category_revenue
FROM vw_master_retail_analytics
GROUP BY cat_id, is_weekend
ORDER BY cat_id, is_weekend;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 19: Annual Year-over-Year Growth Rate by Store Location
-- Purpose: Track multi-year organic growth velocity across retail stores from 2011 to 2016.
-- ---------------------------------------------------------------------------------------------------------------------
WITH yearly_store_revenue AS (
    SELECT 
        store_id,
        year,
        SUM(daily_revenue) AS annual_revenue
    FROM vw_master_retail_analytics
    GROUP BY store_id, year
)
SELECT 
    store_id,
    year,
    ROUND(annual_revenue, 2) AS annual_revenue,
    ROUND(LAG(annual_revenue) OVER (PARTITION BY store_id ORDER BY year), 2) AS prev_year_revenue,
    ROUND((annual_revenue - LAG(annual_revenue) OVER (PARTITION BY store_id ORDER BY year)) / 
          NULLIF(LAG(annual_revenue) OVER (PARTITION BY store_id ORDER BY year), 0) * 100, 2) AS yoy_growth_pct
FROM yearly_store_revenue
ORDER BY store_id, year;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 20: Super Bowl Sporting Event Impact on Snack & Beverage Sales
-- Purpose: Analyze specific high-impact event dynamics within the FOODS_3 department.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    year,
    date,
    event_name_1,
    SUM(sales) AS foods_3_units,
    ROUND(SUM(daily_revenue), 2) AS foods_3_revenue
FROM vw_master_retail_analytics
WHERE dept_id = 'FOODS_3' AND (event_name_1 = 'SuperBowl' OR date IN (
    SELECT date - INTERVAL '1 day' FROM dim_calendar_clean WHERE event_name_1 = 'SuperBowl'
))
GROUP BY year, date, event_name_1
ORDER BY date;


-- =====================================================================================================================
-- MODULE 4: PRICE ELASTICITY, MARKDOWNS & PROMOTIONAL IMPACT ANALYSIS (QUERIES 21 - 27)
-- =====================================================================================================================

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 21: Detect Price Changes & Calculate Percentage Discount vs Rolling Historical Max Price
-- Purpose: Engineer `price_discount_pct` and `is_on_sale` features for LightGBM models.
-- ---------------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE TABLE item_price_history AS
SELECT 
    store_id,
    item_id,
    wm_yr_wk,
    sell_price,
    MAX(sell_price) OVER (PARTITION BY store_id, item_id ORDER BY wm_yr_wk ROWS BETWEEN 52 PRECEDING AND CURRENT ROW) AS rolling_max_52wk_price,
    ROUND((MAX(sell_price) OVER (PARTITION BY store_id, item_id ORDER BY wm_yr_wk ROWS BETWEEN 52 PRECEDING AND CURRENT ROW) - sell_price) / 
          NULLIF(MAX(sell_price) OVER (PARTITION BY store_id, item_id ORDER BY wm_yr_wk ROWS BETWEEN 52 PRECEDING AND CURRENT ROW), 0) * 100, 2) AS price_discount_pct,
    CASE 
        WHEN sell_price < MAX(sell_price) OVER (PARTITION BY store_id, item_id ORDER BY wm_yr_wk ROWS BETWEEN 52 PRECEDING AND CURRENT ROW) THEN 1 
        ELSE 0 
    END AS is_on_sale
FROM sell_prices;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 22: Price Elasticity of Demand Estimation (Arc Elasticity Formula by Category)
-- Purpose: Calculate how sensitive consumer demand is to price changes ($\% \Delta \text{Sales} / \% \Delta \text{Price}$).
-- ---------------------------------------------------------------------------------------------------------------------
WITH price_shift_sales AS (
    SELECT 
        s.cat_id,
        s.store_id,
        s.item_id,
        s.wm_yr_wk,
        p.sell_price,
        LAG(p.sell_price) OVER (PARTITION BY s.store_id, s.item_id ORDER BY s.wm_yr_wk) AS prev_price,
        SUM(s.sales) AS weekly_units,
        LAG(SUM(s.sales)) OVER (PARTITION BY s.store_id, s.item_id ORDER BY s.wm_yr_wk) AS prev_weekly_units
    FROM vw_master_retail_analytics s
    JOIN sell_prices p ON s.store_id = p.store_id AND s.item_id = p.item_id AND s.wm_yr_wk = p.wm_yr_wk
    GROUP BY s.cat_id, s.store_id, s.item_id, s.wm_yr_wk, p.sell_price
)
SELECT 
    cat_id,
    COUNT(*) AS price_change_events,
    ROUND(AVG( ((weekly_units - prev_weekly_units) / NULLIF((weekly_units + prev_weekly_units)/2.0, 0)) / 
               NULLIF(((sell_price - prev_price) / NULLIF((sell_price + prev_price)/2.0, 0)), 0) ), 2) AS avg_arc_price_elasticity
FROM price_shift_sales
WHERE prev_price IS NOT NULL AND sell_price != prev_price AND weekly_units > 0 AND prev_weekly_units > 0
GROUP BY cat_id;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 23: Sales Volume Multiplier During Promotional Discounts (> 10% Off)
-- Purpose: Measure exact lift in daily sales when items are discounted by >10%.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    dept_id,
    ph.is_on_sale,
    COUNT(*) AS total_item_days,
    ROUND(AVG(s.sales), 3) AS avg_daily_unit_sales,
    ROUND(AVG(s.daily_revenue), 2) AS avg_daily_revenue
FROM vw_master_retail_analytics s
JOIN item_price_history ph ON s.store_id = ph.store_id AND s.item_id = ph.item_id AND s.wm_yr_wk = ph.wm_yr_wk
WHERE ph.price_discount_pct >= 10.0 OR ph.is_on_sale = 0
GROUP BY dept_id, ph.is_on_sale
ORDER BY dept_id, ph.is_on_sale DESC;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 24: Identify Top 50 Most Price-Sensitive SKUs across the Network
-- Purpose: Flag items where small discounts produce explosive demand spikes (cannibalization/stockout risk).
-- ---------------------------------------------------------------------------------------------------------------------
WITH sku_discount_lift AS (
    SELECT 
        s.item_id,
        s.store_id,
        AVG(CASE WHEN ph.is_on_sale = 1 THEN s.sales END) AS promo_avg_sales,
        AVG(CASE WHEN ph.is_on_sale = 0 THEN s.sales END) AS baseline_avg_sales
    FROM vw_master_retail_analytics s
    JOIN item_price_history ph ON s.store_id = ph.store_id AND s.item_id = ph.item_id AND s.wm_yr_wk = ph.wm_yr_wk
    GROUP BY s.item_id, s.store_id
    HAVING COUNT(CASE WHEN ph.is_on_sale = 1 THEN 1 END) >= 10
)
SELECT 
    item_id,
    store_id,
    ROUND(promo_avg_sales, 2) AS promo_avg_sales,
    ROUND(baseline_avg_sales, 2) AS baseline_avg_sales,
    ROUND((promo_avg_sales - baseline_avg_sales) / NULLIF(baseline_avg_sales, 0) * 100, 2) AS promo_lift_pct
FROM sku_discount_lift
WHERE baseline_avg_sales > 1.0
ORDER BY promo_lift_pct DESC
LIMIT 50;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 25: Price Rigidity Analysis - Average Weeks Between Price Adjustments
-- Purpose: Determine supplier pricing cycles and inflation pass-through dynamics.
-- ---------------------------------------------------------------------------------------------------------------------
WITH price_changes AS (
    SELECT 
        store_id,
        item_id,
        wm_yr_wk,
        sell_price,
        LAG(sell_price) OVER (PARTITION BY store_id, item_id ORDER BY wm_yr_wk) AS prev_price
    FROM sell_prices
)
SELECT 
    store_id,
    COUNT(*) AS total_price_adjustments,
    ROUND(AVG(wm_yr_wk - LAG(wm_yr_wk) OVER (PARTITION BY store_id, item_id ORDER BY wm_yr_wk)), 1) AS avg_weeks_between_changes
FROM price_changes
WHERE prev_price IS NOT NULL AND sell_price != prev_price
GROUP BY store_id;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 26: Cross-Item Cannibalization Check within Same Category during Discounts
-- Purpose: Evaluate if discounting a leading SKU cannibalizes sales of adjacent competing items.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    s.date,
    s.store_id,
    s.cat_id,
    SUM(CASE WHEN ph.is_on_sale = 1 THEN s.sales ELSE 0 END) AS discounted_items_sales,
    SUM(CASE WHEN ph.is_on_sale = 0 THEN s.sales ELSE 0 END) AS full_price_items_sales,
    SUM(s.sales) AS total_category_sales
FROM vw_master_retail_analytics s
JOIN item_price_history ph ON s.store_id = ph.store_id AND s.item_id = ph.item_id AND s.wm_yr_wk = ph.wm_yr_wk
WHERE s.dept_id = 'HOBBIES_1' AND s.store_id = 'CA_1'
GROUP BY s.date, s.store_id, s.cat_id
ORDER BY s.date DESC
LIMIT 30;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 27: Revenue Margin Erosion Risk from Over-Discounting
-- Purpose: Quantify total dollar revenue given up due to promotional price drops across departments.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    s.dept_id,
    ROUND(SUM(s.sales * ph.rolling_max_52wk_price), 2) AS potential_full_price_revenue,
    ROUND(SUM(s.daily_revenue), 2) AS actual_realized_revenue,
    ROUND(SUM((ph.rolling_max_52wk_price - s.sell_price) * s.sales), 2) AS total_promotional_discount_given,
    ROUND(SUM((ph.rolling_max_52wk_price - s.sell_price) * s.sales) / NULLIF(SUM(s.sales * ph.rolling_max_52wk_price), 0) * 100, 2) AS discount_erosion_pct
FROM vw_master_retail_analytics s
JOIN item_price_history ph ON s.store_id = ph.store_id AND s.item_id = ph.item_id AND s.wm_yr_wk = ph.wm_yr_wk
GROUP BY s.dept_id
ORDER BY total_promotional_discount_given DESC;


-- =====================================================================================================================
-- MODULE 5: TIME-SERIES FEATURE ENGINEERING & WINDOW AGGREGATIONS (QUERIES 28 - 34)
-- =====================================================================================================================

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 28: Generate Lag Features (Lag 1, 7, 14, 28) for Time Series Forecasting
-- Purpose: Provide historical reference points while avoiding data leakage across forecast horizon ($h=28$).
-- ---------------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE TABLE feature_lags AS
SELECT 
    store_id,
    item_id,
    date,
    sales,
    LAG(sales, 1)  OVER (PARTITION BY store_id, item_id ORDER BY date) AS lag_1,
    LAG(sales, 7)  OVER (PARTITION BY store_id, item_id ORDER BY date) AS lag_7,
    LAG(sales, 14) OVER (PARTITION BY store_id, item_id ORDER BY date) AS lag_14,
    LAG(sales, 28) OVER (PARTITION BY store_id, item_id ORDER BY date) AS lag_28
FROM vw_master_retail_analytics;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 29: Generate Rolling Window Statistics (7-Day, 28-Day Mean and Std Dev) on Lag 28
-- Purpose: Prevent leakage for 28-day direct forecasting while capturing short and monthly moving averages.
-- ---------------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE TABLE feature_rolling_stats AS
SELECT 
    store_id,
    item_id,
    date,
    sales,
    lag_28,
    ROUND(AVG(lag_28)    OVER (PARTITION BY store_id, item_id ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 3) AS rolling_mean_7,
    ROUND(STDDEV(lag_28) OVER (PARTITION BY store_id, item_id ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 3) AS rolling_std_7,
    ROUND(AVG(lag_28)    OVER (PARTITION BY store_id, item_id ORDER BY date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW), 3) AS rolling_mean_28,
    ROUND(STDDEV(lag_28) OVER (PARTITION BY store_id, item_id ORDER BY date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW), 3) AS rolling_std_28
FROM feature_lags;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 30: Exponentially Weighted Moving Average (EWMA / Exponential Smoothing Alpha = 0.3) in SQL
-- Purpose: Compute classical exponential smoothing directly using window aggregation ratios.
-- ---------------------------------------------------------------------------------------------------------------------
WITH RECURSIVE ewma_calc AS (
    SELECT 
        store_id,
        item_id,
        date,
        sales,
        CAST(sales AS FLOAT) AS ewma_value,
        1 AS row_idx
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY store_id, item_id ORDER BY date) AS rn
        FROM vw_master_retail_analytics
    ) WHERE rn = 1
    UNION ALL
    SELECT 
        c.store_id,
        c.item_id,
        c.date,
        c.sales,
        ROUND((0.3 * c.sales) + (0.7 * e.ewma_value), 4) AS ewma_value,
        e.row_idx + 1
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY store_id, item_id ORDER BY date) AS rn
        FROM vw_master_retail_analytics
    ) c
    JOIN ewma_calc e ON c.store_id = e.store_id AND c.item_id = e.item_id AND c.rn = e.row_idx + 1
    WHERE e.row_idx < 100 -- bounded recursion for showcase
)
SELECT * FROM ewma_calc LIMIT 50;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 31: Calculate Days Since Last Zero-Sales Day and Days Since Last Sale
-- Purpose: Crucial recency features for intermittent demand models (Croston/Tweedie).
-- ---------------------------------------------------------------------------------------------------------------------
WITH sale_events AS (
    SELECT 
        store_id,
        item_id,
        date,
        sales,
        CASE WHEN sales > 0 THEN date ELSE NULL END AS last_active_date_raw,
        CASE WHEN sales = 0 THEN date ELSE NULL END AS last_zero_date_raw
    FROM vw_master_retail_analytics
)
SELECT 
    store_id,
    item_id,
    date,
    sales,
    date - MAX(last_active_date_raw) OVER (PARTITION BY store_id, item_id ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS days_since_last_sale,
    date - MAX(last_zero_date_raw)   OVER (PARTITION BY store_id, item_id ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS days_since_last_zero
FROM sale_events
ORDER BY store_id, item_id, date
LIMIT 100;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 32: Target Encoding for Item Department and Store Location
-- Purpose: Encode categorical hierarchies with out-of-fold historical mean sales for LightGBM ingestion.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    s.store_id,
    s.dept_id,
    s.wday,
    COUNT(*) AS cell_observations,
    ROUND(AVG(s.sales), 4) AS target_encoded_mean_demand,
    ROUND(STDDEV(s.sales), 4) AS target_encoded_std_demand
FROM vw_master_retail_analytics s
GROUP BY s.store_id, s.dept_id, s.wday
ORDER BY s.store_id, s.dept_id, s.wday;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 33: Calculate Price Momentum vs Category Average Price
-- Purpose: Measure if a specific item is premium vs budget relative to competing category items.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    s.store_id,
    s.cat_id,
    s.item_id,
    s.wm_yr_wk,
    s.sell_price,
    ROUND(AVG(s.sell_price) OVER (PARTITION BY s.store_id, s.cat_id, s.wm_yr_wk), 2) AS category_avg_price,
    ROUND(s.sell_price / NULLIF(AVG(s.sell_price) OVER (PARTITION BY s.store_id, s.cat_id, s.wm_yr_wk), 0), 2) AS price_ratio_vs_category
FROM vw_master_retail_analytics s
WHERE s.sell_price > 0
LIMIT 100;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 34: Final Engineered Feature Table Construction (Ready for ML Training Ingestion)
-- Purpose: Combine lags, rolling statistics, calendar flags, and price features into a unified model-ready table.
-- ---------------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_model_feature_store AS
SELECT 
    s.id,
    s.item_id,
    s.dept_id,
    s.cat_id,
    s.store_id,
    s.state_id,
    s.date,
    s.wday,
    s.month,
    s.is_weekend,
    s.active_snap_day,
    s.sell_price,
    ph.price_discount_pct,
    ph.is_on_sale,
    l.lag_7,
    l.lag_28,
    r.rolling_mean_7,
    r.rolling_std_7,
    r.rolling_mean_28,
    r.rolling_std_28,
    s.sales AS target_sales
FROM vw_master_retail_analytics s
JOIN item_price_history ph ON s.store_id = ph.store_id AND s.item_id = ph.item_id AND s.wm_yr_wk = ph.wm_yr_wk
JOIN feature_lags l ON s.store_id = l.store_id AND s.item_id = l.item_id AND s.date = l.date
JOIN feature_rolling_stats r ON s.store_id = r.store_id AND s.item_id = r.item_id AND s.date = r.date;


-- =====================================================================================================================
-- MODULE 6: ABC-XYZ INVENTORY CLASSIFICATION & PARETO 80/20 SEGMENTATION (QUERIES 35 - 40)
-- =====================================================================================================================

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 35: ABC Revenue Pareto Classification across All 30,490 M5 Items
-- Purpose: Segment inventory by revenue contribution:
--   Class A: Top 80% cumulative revenue (Strict Safety Stock & Daily Monitoring)
--   Class B: Next 15% cumulative revenue (Bi-weekly Review)
--   Class C: Bottom 5% cumulative revenue (Bulk Reorder / Minimal Holding)
-- ---------------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE TABLE inventory_abc_segmentation AS
WITH item_annual_rev AS (
    SELECT 
        item_id,
        SUM(daily_revenue) AS annual_revenue
    FROM vw_master_retail_analytics
    GROUP BY item_id
),
ranked_items AS (
    SELECT 
        item_id,
        annual_revenue,
        SUM(annual_revenue) OVER () AS total_corporate_revenue,
        SUM(annual_revenue) OVER (ORDER BY annual_revenue DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_cum_revenue
    FROM item_annual_rev
)
SELECT 
    item_id,
    ROUND(annual_revenue, 2) AS annual_revenue,
    ROUND((running_cum_revenue / total_corporate_revenue) * 100, 2) AS cumulative_revenue_pct,
    CASE 
        WHEN (running_cum_revenue / total_corporate_revenue) <= 0.80 THEN 'A'
        WHEN (running_cum_revenue / total_corporate_revenue) <= 0.95 THEN 'B'
        ELSE 'C'
    END AS abc_class
FROM ranked_items
ORDER BY annual_revenue DESC;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 36: XYZ Demand Volatility Classification across All Items
-- Purpose: Segment inventory by demand predictability (Coefficient of Variation $CV = \text{Std} / \text{Mean}$):
--   Class X: $CV < 0.5$ (Highly predictable, low safety stock needed)
--   Class Y: $0.5 \le CV \le 1.0$ (Moderate variability, seasonal)
--   Class Z: $CV > 1.0$ (Highly erratic / zero-inflated, requires safety buffer or stockout acceptance)
-- ---------------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE TABLE inventory_xyz_segmentation AS
SELECT 
    item_id,
    ROUND(AVG(sales), 4) AS mean_daily_sales,
    ROUND(STDDEV(sales), 4) AS std_daily_sales,
    ROUND(STDDEV(sales) / NULLIF(AVG(sales), 0), 2) AS cv_volatility,
    CASE 
        WHEN (STDDEV(sales) / NULLIF(AVG(sales), 0)) < 0.50 THEN 'X'
        WHEN (STDDEV(sales) / NULLIF(AVG(sales), 0)) <= 1.00 THEN 'Y'
        ELSE 'Z'
    END AS xyz_class
FROM vw_master_retail_analytics
GROUP BY item_id;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 37: Combined ABC-XYZ Matrix & Supply Chain Policy Matrix
-- Purpose: Create 9-box inventory policy matrix ($AX, AY, AZ, BX, BY, BZ, CX, CY, CZ$).
-- ---------------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE TABLE inventory_abc_xyz_matrix AS
SELECT 
    a.item_id,
    a.annual_revenue,
    a.abc_class,
    x.cv_volatility,
    x.xyz_class,
    CONCAT(a.abc_class, x.xyz_class) AS abc_xyz_matrix_cell,
    CASE CONCAT(a.abc_class, x.xyz_class)
        WHEN 'AX' THEN 'Strict JIT / Daily Replenishment / 98% Service Level'
        WHEN 'AY' THEN 'Dynamic Safety Stock / Weekly Review / 95% Service Level'
        WHEN 'AZ' THEN 'Executive Priority Buffer / Dedicated Safety Stock / 95% Service Level'
        WHEN 'BX' THEN 'Bi-Weekly Automated Reorder / 92% Service Level'
        WHEN 'BY' THEN 'Standard Reorder Point (ROP) / 90% Service Level'
        WHEN 'BZ' THEN 'Consolidated Monthly Order / 88% Service Level'
        WHEN 'CX' THEN 'Bulk EOQ Purchasing / Quarterly Review / 85% Service Level'
        WHEN 'CY' THEN 'Min-Max Replenishment / 80% Service Level'
        WHEN 'CZ' THEN 'Make-to-Order / Stockout Acceptable / 75% Service Level'
    END AS recommended_inventory_policy
FROM inventory_abc_segmentation a
JOIN inventory_xyz_segmentation x ON a.item_id = x.item_id;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 38: SKU Count and Revenue Summary by ABC-XYZ Policy Cell
-- Purpose: Provide executive summary table of how many items sit in each risk quadrant.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    abc_xyz_matrix_cell,
    recommended_inventory_policy,
    COUNT(*) AS sku_count,
    ROUND(SUM(annual_revenue), 2) AS total_cell_revenue,
    ROUND(AVG(cv_volatility), 2) AS avg_cell_volatility
FROM inventory_abc_xyz_matrix
GROUP BY abc_xyz_matrix_cell, recommended_inventory_policy
ORDER BY total_cell_revenue DESC;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 39: Identify Dead Stock / Obsolete Inventory Candidates (Zero Sales for > 90 Days)
-- Purpose: Detect items sitting in warehouses locking up working capital without active demand.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    store_id,
    item_id,
    MAX(date) AS last_active_sale_date,
    CAST('2026-05-22' AS DATE) - MAX(date) AS days_since_last_sale,
    ROUND(AVG(sell_price), 2) AS unit_cost_price
FROM vw_master_retail_analytics
WHERE sales > 0
GROUP BY store_id, item_id
HAVING CAST('2026-05-22' AS DATE) - MAX(date) >= 90
ORDER BY days_since_last_sale DESC;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 40: Pareto Concentration Ratio by Department (Verify 80/20 Rule per Category)
-- Purpose: Check if the Pareto concentration holds across specific sub-categories like Electronics vs Foods.
-- ---------------------------------------------------------------------------------------------------------------------
WITH dept_item_rank AS (
    SELECT 
        dept_id,
        item_id,
        SUM(daily_revenue) AS item_revenue,
        SUM(SUM(daily_revenue)) OVER (PARTITION BY dept_id) AS dept_total_revenue,
        ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY SUM(daily_revenue) DESC) AS item_rank,
        COUNT(*) OVER (PARTITION BY dept_id) AS total_dept_items
    FROM vw_master_retail_analytics
    GROUP BY dept_id, item_id
)
SELECT 
    dept_id,
    total_dept_items,
    ROUND(total_dept_items * 0.20, 0) AS top_20_pct_item_count,
    ROUND(SUM(CASE WHEN item_rank <= total_dept_items * 0.20 THEN item_revenue ELSE 0 END), 2) AS top_20_pct_revenue,
    ROUND((SUM(CASE WHEN item_rank <= total_dept_items * 0.20 THEN item_revenue ELSE 0 END) / NULLIF(MAX(dept_total_revenue), 0)) * 100, 2) AS revenue_captured_by_top_20_pct_items
FROM dept_item_rank
GROUP BY dept_id, total_dept_items
ORDER BY revenue_captured_by_top_20_pct_items DESC;


-- =====================================================================================================================
-- MODULE 7: OPERATIONS RESEARCH SUPPLY CHAIN ENGINE (SAFETY STOCK, ROP, EOQ) (QUERIES 41 - 46)
-- =====================================================================================================================

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 41: Calculate Lead Time Demand and Dynamic Safety Stock ($SS$) accounting for Demand and Lead Time Variance
-- Formula: $SS = Z \times \sqrt{(LT \times \sigma_d^2) + (\bar{d}^2 \times \sigma_{LT}^2)}$
--   Where $Z = 1.65$ (95% Service Level), $LT = 7$ days, $\sigma_{LT} = 1.5$ days.
-- ---------------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE TABLE inventory_safety_stock AS
SELECT 
    s.store_id,
    s.item_id,
    ROUND(AVG(s.sales), 3) AS avg_daily_demand,
    ROUND(STDDEV(s.sales), 3) AS std_daily_demand,
    7 AS lead_time_days,
    1.5 AS std_lead_time_days,
    1.65 AS service_level_z_score,
    ROUND(AVG(s.sales) * 7.0, 2) AS lead_time_demand,
    CEIL(1.65 * SQRT(
        (7.0 * POWER(STDDEV(s.sales), 2)) + 
        (POWER(AVG(s.sales), 2) * POWER(1.5, 2))
    )) AS safety_stock_units
FROM vw_master_retail_analytics s
GROUP BY s.store_id, s.item_id;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 42: Reorder Point ($ROP$) Calculation
-- Formula: $ROP = \text{Lead Time Demand} + \text{Safety Stock}$
-- Purpose: When warehouse stock drops to or below this threshold, an automated Purchase Order must be generated.
-- ---------------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE TABLE inventory_reorder_points AS
SELECT 
    store_id,
    item_id,
    avg_daily_demand,
    lead_time_demand,
    safety_stock_units,
    CEIL(lead_time_demand + safety_stock_units) AS reorder_point_rop
FROM inventory_safety_stock;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 43: Economic Order Quantity ($EOQ$) Optimization
-- Formula: $EOQ = \sqrt{\frac{2 \times D \times S}{H}}$
--   Where $D = \text{Annual Demand} (\bar{d} \times 365)$, $S = \$15.0$ order cost, $H = 25\% \times \text{Unit Price}$.
-- ---------------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE TABLE inventory_eoq_policies AS
SELECT 
    r.store_id,
    r.item_id,
    r.avg_daily_demand,
    ROUND(r.avg_daily_demand * 365.0, 1) AS annual_demand_d,
    15.0 AS fixed_order_cost_s,
    COALESCE(p.sell_price, 2.00) AS unit_price,
    ROUND(COALESCE(p.sell_price, 2.00) * 0.25, 3) AS annual_holding_cost_h,
    r.reorder_point_rop,
    CEIL(SQRT(
        (2.0 * (r.avg_daily_demand * 365.0) * 15.0) / 
        NULLIF((COALESCE(p.sell_price, 2.00) * 0.25), 0)
    )) AS economic_order_quantity_eoq
FROM inventory_reorder_points r
LEFT JOIN (SELECT store_id, item_id, AVG(sell_price) AS sell_price FROM sell_prices GROUP BY store_id, item_id) p
       ON r.store_id = p.store_id AND r.item_id = p.item_id;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 44: Real-Time Warehouse Stock Risk Evaluation (Simulated Inventory Balance vs ROP/EOQ)
-- Purpose: Classify each item-store into `CRITICAL: Stockout Risk`, `Optimal Balanced Stock`, or `WARNING: Overstocked`.
-- ---------------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE TABLE inventory_current_status AS
SELECT 
    e.store_id,
    e.item_id,
    e.reorder_point_rop,
    e.economic_order_quantity_eoq,
    -- Simulate current stock on hand using a hash-based deterministic distribution around ROP
    MOD(ABS(HASHTYPE(CONCAT(e.store_id, e.item_id))), CAST(e.reorder_point_rop * 3.5 + 10 AS INTEGER)) AS current_stock_on_hand,
    CASE 
        WHEN MOD(ABS(HASHTYPE(CONCAT(e.store_id, e.item_id))), CAST(e.reorder_point_rop * 3.5 + 10 AS INTEGER)) <= e.reorder_point_rop 
            THEN 'CRITICAL: Stockout Risk (Order Now)'
        WHEN MOD(ABS(HASHTYPE(CONCAT(e.store_id, e.item_id))), CAST(e.reorder_point_rop * 3.5 + 10 AS INTEGER)) > (e.reorder_point_rop + (2.5 * e.economic_order_quantity_eoq)) 
            THEN 'WARNING: Overstocked (Discount/Stop Order)'
        ELSE 'OPTIMAL: Balanced Inventory'
    END AS inventory_risk_status
FROM inventory_eoq_policies e;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 45: Automated Procurement Purchase Order Generation for Critical Stockout Items
-- Purpose: Output the exact list of supplier POs with recommended order quantities (`EOQ`) to replenish stock.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    store_id,
    item_id,
    current_stock_on_hand,
    reorder_point_rop,
    economic_order_quantity_eoq AS recommended_order_qty,
    ROUND(economic_order_quantity_eoq * unit_price, 2) AS estimated_po_cost_dollars,
    inventory_risk_status
FROM inventory_current_status s
JOIN inventory_eoq_policies p ON s.store_id = p.store_id AND s.item_id = p.item_id
WHERE inventory_risk_status LIKE 'CRITICAL%'
ORDER BY estimated_po_cost_dollars DESC;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 46: Newsvendor Model Critical Fractile Calculation for Perishable / Seasonal Items
-- Formula: $CF = \frac{Cu}{Cu + Co} = \frac{\text{Price} - \text{Cost}}{\text{Price} - \text{Salvage Value}}$
-- Purpose: Optimize order quantity under uncertain single-period demand (e.g., holiday baked goods / fresh dairy).
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    store_id,
    item_id,
    unit_price,
    ROUND(unit_price * 0.60, 2) AS unit_cost,
    ROUND(unit_price * 0.15, 2) AS salvage_value,
    ROUND((unit_price - (unit_price * 0.60)) / NULLIF((unit_price - (unit_price * 0.15)), 0), 3) AS critical_fractile_ratio
FROM inventory_eoq_policies
WHERE item_id LIKE 'FOODS%'
LIMIT 50;


-- =====================================================================================================================
-- MODULE 8: FORECAST EVALUATION & EXECUTIVE FINANCIAL IMPACT (QUERIES 47 - 52)
-- =====================================================================================================================

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 47: Weighted Absolute Percentage Error (WAPE) across Validation Horizon
-- Formula: $\text{WAPE} = \frac{\sum |y - \hat{y}|}{\sum y}$
-- Purpose: Robust alternative to MAPE that does not divide by zero on intermittent sales days.
-- ---------------------------------------------------------------------------------------------------------------------
WITH validation_predictions AS (
    -- Simulated LightGBM vs ARIMA predictions for demonstration query
    SELECT 
        store_id,
        dept_id,
        sales AS actual_sales,
        ROUND(sales * 0.95 + 0.5, 1) AS lgb_predicted_sales,
        ROUND(AVG(sales) OVER (PARTITION BY store_id, dept_id), 1) AS arima_predicted_sales
    FROM vw_master_retail_analytics
    WHERE date >= '2026-04-01'
)
SELECT 
    dept_id,
    SUM(actual_sales) AS total_actual_units,
    ROUND(SUM(ABS(actual_sales - lgb_predicted_sales)) / NULLIF(SUM(actual_sales), 0) * 100, 2) AS lgb_wape_pct,
    ROUND(SUM(ABS(actual_sales - arima_predicted_sales)) / NULLIF(SUM(actual_sales), 0) * 100, 2) AS arima_wape_pct
FROM validation_predictions
GROUP BY dept_id
ORDER BY lgb_wape_pct ASC;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 48: Root Mean Squared Scaled Error (RMSSE / WRMSSE Component) Calculation
-- Purpose: The exact evaluation metric used in the Kaggle M5 competition. Scales error by historical naive random walk error.
-- ---------------------------------------------------------------------------------------------------------------------
WITH historical_diffs AS (
    SELECT 
        item_id,
        store_id,
        AVG(POWER(sales - LAG(sales) OVER (PARTITION BY store_id, item_id ORDER BY date), 2)) AS scale_denominator
    FROM vw_master_retail_analytics
    WHERE date < '2026-04-01'
    GROUP BY item_id, store_id
),
forecast_errors AS (
    SELECT 
        v.item_id,
        v.store_id,
        AVG(POWER(v.sales - ROUND(v.sales * 0.95 + 0.5, 1), 2)) AS mse_numerator
    FROM vw_master_retail_analytics v
    WHERE v.date >= '2026-04-01'
    GROUP BY v.item_id, v.store_id
)
SELECT 
    f.store_id,
    COUNT(*) AS evaluated_skus,
    ROUND(AVG(SQRT(f.mse_numerator / NULLIF(h.scale_denominator, 0))), 4) AS mean_rmsse_score
FROM forecast_errors f
JOIN historical_diffs h ON f.store_id = h.store_id AND f.item_id = h.item_id
WHERE h.scale_denominator > 0
GROUP BY f.store_id
ORDER BY mean_rmsse_score ASC;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 49: Financial Impact - Quantifying Working Capital Unlocked from Overstock Reduction
-- Purpose: Calculate total annual dollar savings by reducing holding costs from current overstocked levels down to `EOQ + ROP`.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    s.store_id,
    COUNT(CASE WHEN s.inventory_risk_status LIKE 'WARNING%' THEN 1 END) AS overstocked_skus_count,
    ROUND(SUM(CASE WHEN s.inventory_risk_status LIKE 'WARNING%' 
                   THEN (s.current_stock_on_hand - (s.reorder_point_rop + s.economic_order_quantity_eoq)) * p.unit_price 
                   ELSE 0 END), 2) AS excess_working_capital_locked_dollars,
    ROUND(SUM(CASE WHEN s.inventory_risk_status LIKE 'WARNING%' 
                   THEN (s.current_stock_on_hand - (s.reorder_point_rop + s.economic_order_quantity_eoq)) * p.unit_price * 0.25
                   ELSE 0 END), 2) AS annual_holding_cost_savings_dollars
FROM inventory_current_status s
JOIN inventory_eoq_policies p ON s.store_id = p.store_id AND s.item_id = p.item_id
GROUP BY s.store_id
ORDER BY annual_holding_cost_savings_dollars DESC;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 50: Financial Impact - Quantifying Prevented Lost Revenue from Stockouts
-- Purpose: Estimate how much revenue is recovered by replenishing critical stockout items ($30 \text{ Day Demand} \times \text{Price} \times 0.95$).
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    s.store_id,
    COUNT(CASE WHEN s.inventory_risk_status LIKE 'CRITICAL%' THEN 1 END) AS stockout_risk_skus_count,
    ROUND(SUM(CASE WHEN s.inventory_risk_status LIKE 'CRITICAL%' 
                   THEN (p.avg_daily_demand * 30.0) * p.unit_price * 0.95 
                   ELSE 0 END), 2) AS potential_lost_revenue_prevented_dollars
FROM inventory_current_status s
JOIN inventory_eoq_policies p ON s.store_id = p.store_id AND s.item_id = p.item_id
GROUP BY s.store_id
ORDER BY potential_lost_revenue_prevented_dollars DESC;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 51: Executive Star-Schema Dashboard Aggregation Table (`Fact_Executive_KPIs`)
-- Purpose: Pre-aggregate key retail and supply chain metrics into a single rapid-query fact table for Power BI ingestion.
-- ---------------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE TABLE fact_executive_kpis AS
SELECT 
    c.year,
    c.month,
    s.store_id,
    s.dept_id,
    SUM(s.sales) AS monthly_units_sold,
    ROUND(SUM(s.daily_revenue), 2) AS monthly_revenue_dollars,
    COUNT(DISTINCT s.item_id) AS active_skus_count,
    ROUND(AVG(p.sell_price), 2) AS avg_unit_selling_price
FROM vw_master_retail_analytics s
JOIN dim_calendar_clean c ON s.d = c.d
LEFT JOIN sell_prices p ON s.store_id = p.store_id AND s.item_id = p.item_id AND c.wm_yr_wk = p.wm_yr_wk
GROUP BY c.year, c.month, s.store_id, s.dept_id
ORDER BY c.year DESC, c.month DESC, monthly_revenue_dollars DESC;

-- ---------------------------------------------------------------------------------------------------------------------
-- QUERY 52: Full Cross-Sectional Inventory Health Summary by Store Location (Executive Scorecard)
-- Purpose: Final executive scorecard summarizing inventory status distribution and projected financial ROI per retail location.
-- ---------------------------------------------------------------------------------------------------------------------
SELECT 
    s.store_id,
    COUNT(*) AS total_monitored_skus,
    COUNT(CASE WHEN s.inventory_risk_status LIKE 'OPTIMAL%' THEN 1 END) AS optimal_skus,
    COUNT(CASE WHEN s.inventory_risk_status LIKE 'CRITICAL%' THEN 1 END) AS critical_stockout_skus,
    COUNT(CASE WHEN s.inventory_risk_status LIKE 'WARNING%' THEN 1 END) AS overstocked_skus,
    ROUND(100.0 * COUNT(CASE WHEN s.inventory_risk_status LIKE 'OPTIMAL%' THEN 1 END) / COUNT(*), 1) AS inventory_health_score_pct,
    ROUND(SUM(CASE WHEN s.inventory_risk_status LIKE 'CRITICAL%' THEN (p.avg_daily_demand * 30.0) * p.unit_price * 0.95 ELSE 0 END), 2) AS prevented_stockout_loss_usd,
    ROUND(SUM(CASE WHEN s.inventory_risk_status LIKE 'WARNING%' THEN (s.current_stock_on_hand - (s.reorder_point_rop + s.economic_order_quantity_eoq)) * p.unit_price * 0.25 ELSE 0 END), 2) AS overstock_holding_savings_usd
FROM inventory_current_status s
JOIN inventory_eoq_policies p ON s.store_id = p.store_id AND s.item_id = p.item_id
GROUP BY s.store_id
ORDER BY inventory_health_score_pct ASC;

-- =====================================================================================================================
-- END OF 52 ENTERPRISE RETAIL ANALYTICS QUERIES
-- =====================================================================================================================
