# Alt Season Indicators Dashboard

A real-time dashboard that tracks and analyzes various cryptocurrency market indicators to determine if we're in an "alt season" - a period when altcoins significantly outperform Bitcoin.

## Features

- Real-time market data from CoinGecko API
- Multiple indicator tracking:
  - Bitcoin Dominance
  - ETH/BTC Ratio
  - Fear & Greed Index
  - Bitcoin RSI
  - Top 10 Altcoin Performance
  - Volume Dominance
  - Monthly ROI Analysis
- Server-side caching to respect API limits
- Dark/Light mode support
- Mobile responsive design

## Indicator Logic

### Primary Indicators

1. **Bitcoin Dominance** (< 45%)
   - Measures Bitcoin's market cap as a percentage of total crypto market cap
   - Lower dominance suggests money flowing into altcoins
   - Historical alt seasons typically occur below 45%

2. **ETH/BTC Ratio** (> 0.07)
   - Measures Ethereum's strength against Bitcoin
   - Higher ratio indicates altcoin market strength
   - ETH often leads altcoin movements

3. **Fear & Greed Index** (> 65)
   - Market sentiment indicator from 0-100
   - Above 65 indicates "Greed" or "Extreme Greed"
   - Alt seasons typically occur during high greed periods

### Supporting Indicators

4. **BTC Monthly ROI** (< 0%)
   - Bitcoin's 30-day return on investment
   - Negative ROI suggests capital moving from BTC to alts
   - Helps confirm market rotation

5. **Top 10 Alts Performance** (> 10%)
   - Average 30-day performance of top 10 altcoins vs BTC
   - Strong outperformance indicates alt season momentum
   - Excludes Bitcoin from calculation

6. **Volume Dominance** (> 60%)
   - Percentage of total market volume in altcoins
   - High volume suggests active alt trading
   - Confirms market participation

### Alt Season Detection

The dashboard considers it an alt season when: 

- Bitcoin Dominance is below 45%
- ETH/BTC Ratio is above 0.07
- Fear & Greed Index is above 65
- BTC Monthly ROI is below 0%
- Top 10 Alts Performance is above 10%
- Volume Dominance is above 60%

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
   Params: vs_currency=btc&order=market_cap_desc&per_page=11&sparkline=false&price_change_percentage=30d
   Note: Excludes Bitcoin from results
   ```

5. **Volume Dominance**
   ```
   GET /global
   Response: total_volume
   Combined with Bitcoin price for calculation
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

### Rate Limits

- CoinGecko Free API: 10-30 calls/minute
- Alternative.me: No strict limit
- TradingView: Client-side only

### Caching Strategy

- Default cache duration: 35 minutes
- Cache cleanup age: 60 minutes
- API rate limit delay: 1 second between calls
- Server-side caching to respect API limits

## Requirements

- Python 3.10+
- Flask
- Requests
- python-dotenv
- TradingView Charts (via CDN)
- Tailwind CSS (via CDN)