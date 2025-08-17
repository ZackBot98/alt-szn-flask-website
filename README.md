# Alt Season Indicators Dashboard

A real-time dashboard that tracks and analyzes various cryptocurrency market indicators to determine if we're in an "alt season" - a period when altcoins significantly outperform Bitcoin.

## Features

- Real-time market data from CoinGecko API
- Multiple indicator tracking:
  - Bitcoin Dominance
  - ETH/BTC Ratio
  - Fear & Greed Index
  - Bitcoin RSI (14-day)
  - Top 10 Altcoin Performance (30-day)
  - Altcoin Volume Dominance (24h)
  - Altcoin Market Cap Dominance
  - BTC Monthly ROI (30-day)
- Advanced caching system with 60-minute refresh cycles
- Intelligent rate limiting (30 calls/minute)
- Stablecoin exclusion for accurate altcoin metrics
- Server-side caching to respect API limits
- Dark/Light mode support
- Mobile responsive design

## Indicator Logic

### Primary Indicators

1. **Bitcoin Dominance** (< 45%)
   - Measures Bitcoin's market cap as a percentage of total crypto market cap
   - Lower dominance suggests money flowing into altcoins
   - Historical alt seasons typically occur below 45%
   - Data source: CoinGecko `/global` endpoint

2. **ETH/BTC Ratio** (> 0.07)
   - Measures Ethereum's strength against Bitcoin
   - Higher ratio indicates altcoin market strength
   - ETH often leads altcoin movements
   - Data source: CoinGecko `/simple/price` endpoint

3. **Fear & Greed Index** (> 65)
   - Market sentiment indicator from 0-100
   - Above 65 indicates "Greed" or "Extreme Greed"
   - Alt seasons typically occur during high greed periods
   - Data source: `api.alternative.me/fng/`

### Supporting Indicators

4. **BTC Monthly ROI** (< 0%)
   - Bitcoin's 30-day return on investment
   - Negative ROI suggests capital moving from BTC to alts
   - Helps confirm market rotation
   - Data source: CoinGecko market chart endpoint

5. **Top 10 Alts Performance** (> 10%)
   - Average 30-day performance of top 10 altcoins vs BTC
   - **Excludes stablecoins** (USDT, USDC, BUSD, etc.) for accurate altcoin performance
   - Strong outperformance indicates alt season momentum
   - Data source: CoinGecko `/coins/markets` with stablecoin filtering

6. **Altcoin Volume Dominance (24h)** (> 60%)
   - Percentage of total market volume in altcoins over the past 24 hours
   - **Excludes stablecoins** for accurate volume analysis
   - High volume suggests active alt trading and market participation
   - 24h timeframe captures real-time market sentiment
   - Data source: CoinGecko `/coins/markets` with pagination

7. **Bitcoin RSI (14-day)** 
   - Relative Strength Index using 14-period exponential moving average
   - Uses hourly OHLC data for accuracy
   - Standard interpretation: < 30 oversold, > 70 overbought
   - Data source: CoinGecko OHLC endpoint

8. **Altcoin Market Cap Dominance**
   - Ratio of total altcoin market cap to total crypto market cap
   - Excludes Bitcoin from calculation
   - Shows altcoin market share trends
   - Data source: Derived from Bitcoin dominance data

### Alt Season Detection

The dashboard considers it an alt season when **ALL** of the following conditions are met:

- Bitcoin Dominance is below 45%
- ETH/BTC Ratio is above 0.07
- Fear & Greed Index is above 65
- BTC Monthly ROI is below 0%
- Top 10 Alts Performance is above 10%
- Altcoin Volume Dominance is above 60%

## Technical Improvements

### Advanced Caching System
- **Cache Duration**: 60 minutes (increased from 35)
- **Cache Cleanup**: 120 minutes (increased from 60)
- **Stablecoin Cache**: 24-hour cache for stablecoin IDs (they don't change often)
- **Pre-loading**: Cache populated on application startup for better first-visitor experience

### Rate Limiting & API Management
- **Rate Limit**: 30 calls per minute (CoinGecko Demo plan limit)
- **Intelligent Delays**: 500ms between pagination requests, 1s between major operations
- **Monthly Budget**: 10,000 API calls/month with 80% safety margin
- **API Usage Tracking**: Real-time monitoring of monthly API consumption

### Stablecoin Exclusion
- **Comprehensive Filtering**: Uses CoinGecko's category filter to identify all stablecoins
- **Efficient Implementation**: Single stablecoin fetch cached for 24 hours
- **Applied To**: Top 10 Alts Performance and Volume Dominance calculations
- **Benefits**: More accurate altcoin metrics without stablecoin noise

### Data Quality & Validation
- **Robust Error Handling**: Graceful fallbacks for all API failures
- **Data Validation**: Checks for data completeness and quality
- **Safe Calculations**: Prevents division by zero and invalid operations
- **Logging**: Comprehensive error tracking and monitoring

## Data Sources & APIs

### CoinGecko API Endpoints

1. **Bitcoin Dominance**
   ```
   GET /global
   Response: market_cap_percentage.btc
   ```

2. **ETH/BTC Ratio**
   ```
   GET /simple/price
   Params: ids=ethereum,bitcoin&vs_currencies=usd
   Calculation: eth_price / btc_price
   ```

3. **BTC Monthly ROI**
   ```
   GET /coins/bitcoin/market_chart
   Params: vs_currency=usd&days=30&interval=daily
   Calculation: ((end_price - start_price) / start_price) * 100
   ```

4. **Top 10 Alts Performance**
   ```
   GET /coins/markets
   Params: vs_currency=btc&order=market_cap_desc&per_page=20&sparkline=false&price_change_percentage=30d
   Note: Excludes Bitcoin and stablecoins, takes top 10 after filtering
   ```

5. **Altcoin Volume Dominance (24h)**
   ```
   GET /coins/markets (with pagination)
   Params: vs_currency=usd&per_page=250&price_change_percentage=24h&order=market_cap_desc
   Note: Excludes stablecoins, calculates (altcoin_volume / (btc_volume + altcoin_volume)) * 100
   ```

6. **Bitcoin RSI**
   ```
   GET /coins/bitcoin/ohlc
   Params: vs_currency=usd&days=14
   Note: Uses hourly data for 14-period RSI calculation
   ```

7. **Stablecoin Identification**
   ```
   GET /coins/markets
   Params: category=stablecoins&vs_currency=usd&per_page=250&order=market_cap_desc
   Note: Cached for 24 hours, used across multiple indicators
   ```

### External APIs

1. **Fear & Greed Index**
   ```
   GET https://api.alternative.me/fng/
   Response: Current market sentiment (0-100)
   ```

2. **TradingView Charts**
   ```
   Symbols Used:
   - CRYPTOCAP:BTC.D (Bitcoin Dominance)
   - CRYPTOCAP:TOTAL2 (Altcoin Market Cap)
   - BINANCE:ETHBTC (ETH/BTC Ratio)
   - TOTAL2/TOTAL (Altcoin Dominance)
   ```

### Rate Limits & Caching

- **CoinGecko Demo API**: 30 calls/minute, 10,000 calls/month
- **Cache Strategy**: 60-minute refresh cycles with intelligent pre-loading
- **Rate Limiting**: Built-in delays and request tracking
- **API Budget**: 80% of monthly limit used for safety margin

## Requirements

- Python 3.10+
- Flask
- Requests
- python-dotenv
- TradingView Charts (via CDN)
- Tailwind CSS (via CDN)

## Performance & Reliability

- **Cache Hit Rate**: Optimized for high cache hit rates
- **API Efficiency**: Minimal redundant API calls through intelligent caching
- **Error Resilience**: Graceful degradation when APIs are unavailable
- **Startup Performance**: Pre-loaded cache ensures fast first-visitor experience
- **Monthly API Usage**: Typically stays well under 8,000 calls/month

## Links

- [Buy Me a Coffee](https://buymeacoffee.com/thealtsignal)
- [Alt Season Indicators](https://thealtsignal.com/alt-season-indicators)
- [Changelog](https://thealtsignal.com/changelog)
- [Alt Season Telegram Channel](https://t.me/thealtsignal)