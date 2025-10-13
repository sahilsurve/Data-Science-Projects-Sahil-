-- =========================================================================
-- STEP 1: Computes open/close prices per day per stock using window functions.
-- =========================================================================
with enriched as (
    select
        symbol,
        cast(market_timestamp as date) as trade_date,  -- Extract trading date from timestamp
        day_low,                                       -- Daily low price
        day_high,                                      -- Daily high price
        current_price,                                 -- Current price at each timestamp

        -- First price of the day → opening price
        first_value(current_price) over (
            partition by symbol, cast(market_timestamp as date)
            order by market_timestamp
        ) as candle_open,

        -- Last price of the day → closing price
        last_value(current_price) over (
            partition by symbol, cast(market_timestamp as date)
            order by market_timestamp
            rows between unbounded preceding and unbounded following
        ) as candle_close

    from {{ ref('silver_clean_stock_quotes') }}        -- Reference to the cleaned stock quotes table
),

-- ===================================================================================================
-- STEP 2: CANDLES - Aggregates into daily OHLC (open, high, low, close) data and adds a trend line.
-- ===================================================================================================
candles as (
    select
        symbol,
        trade_date as candle_time,                     -- Candle timestamp (daily)
        min(day_low) as candle_low,                    -- Daily minimum price
        max(day_high) as candle_high,                  -- Daily maximum price
        any_value(candle_open) as candle_open,         -- Use representative open (should be same per day)
        any_value(candle_close) as candle_close,       -- Use representative close (should be same per day)
        avg(current_price) as trend_line               -- Average daily price (trend indicator)
    from enriched
    group by symbol, trade_date
),

-- =========================================================================
-- STEP 3: RANKED - Orders daily candles from newest to oldest for each symbol.
-- =========================================================================
ranked as (
    select
        c.*,
        row_number() over (
            partition by symbol
            order by candle_time desc
        ) as rn                                        -- Rank candles by most recent date
    from candles c
)

-- ===================================================================================================
-- STEP 4: FINAL SELECTION - Returns the last 12 days (candles) for each symbol in chronological order.
-- ===================================================================================================
select
    symbol,
    candle_time,
    candle_low,
    candle_high,
    candle_open,
    candle_close,
    trend_line
from ranked
where rn <= 12                                         -- Keep last 12 daily candles
order by symbol, candle_time                          -- Sort output by symbol, then date
