-- ========================================================================
-- STEP 1: SOURCE - Loads stock quotes, ensuring prices are valid numeric values.
-- ========================================================================
WITH source AS (
  SELECT
    symbol,                                            -- Stock ticker symbol
    TRY_CAST(current_price AS DOUBLE) AS current_price_dbl, -- Safely cast current_price to numeric type
    market_timestamp                                   -- Timestamp of market quote
  FROM {{ ref('silver_clean_stock_quotes') }}          -- Reference to cleaned stock quotes model

  -- Optional: filter out rows with invalid or non-numeric prices
  WHERE TRY_CAST(current_price AS DOUBLE) IS NOT NULL
),

-- =================================================================================================
-- STEP 2: LATEST_DAY - Determines the most recent trading date based on market_timestamp.
-- =================================================================================================
latest_day AS (
  SELECT
    -- Convert latest timestamp to a DATE (if stored as epoch seconds)
    CAST(TO_TIMESTAMP_LTZ(MAX(market_timestamp)) AS DATE) AS max_day
  FROM source
),

-- =================================================================================================
-- STEP 3: LATEST_PRICES - Calculates the average stock price per symbol for that most recent day.
-- =================================================================================================
latest_prices AS (
  SELECT
    symbol,                                            -- Stock ticker symbol
    AVG(current_price_dbl) AS avg_price                -- Average price for the most recent trading day
  FROM source
  JOIN latest_day ld
    -- Match rows from the source that fall on the latest trading date
    ON CAST(TO_TIMESTAMP_LTZ(market_timestamp) AS DATE) = ld.max_day
  GROUP BY symbol
),

-- ==========================================================================================================================
-- STEP 4: ALL_TIME_VOLATILITY - Computes standard deviation (volatility) and relative volatility over all historical data.
-- ==========================================================================================================================
all_time_volatility AS (
  SELECT
    symbol,                                            -- Stock ticker symbol

    -- Absolute volatility across all time (population std. dev.)
    STDDEV_POP(current_price_dbl) AS volatility,             

    -- Relative volatility (normalized by mean price)
    CASE
      WHEN AVG(current_price_dbl) = 0 THEN NULL
      ELSE STDDEV_POP(current_price_dbl) / NULLIF(AVG(current_price_dbl), 0)
    END AS relative_volatility

  FROM source
  GROUP BY symbol
)

-- ========================================================================
-- STEP 5: FINAL OUTPUT - Combine latest-day averages with long-term volatility metrics
-- ========================================================================
SELECT
  lp.symbol,                                          -- Stock ticker symbol
  lp.avg_price,                                       -- Average price on latest day
  v.volatility,                                       -- Historical absolute volatility
  v.relative_volatility                               -- Historical relative volatility
FROM latest_prices lp
JOIN all_time_volatility v 
  ON lp.symbol = v.symbol                             -- Join per symbol
ORDER BY lp.symbol                                    -- Output sorted alphabetically
