from flask import Flask, jsonify, render_template, send_from_directory
from datetime import datetime, timezone, timedelta
import requests
import time
from functools import wraps
import logging
import os
from dotenv import load_dotenv

# RSI Implementation: Uses standard 14-period exponential moving average method
# Data Source: CoinGecko OHLC endpoint for hourly price data (more accurate than daily)
# Rate Limit Impact: No additional API calls (replaces existing market_chart call)

# Load environment variables
load_dotenv()

# Configuration
class Config:
    COINGECKO_API_URL = "https://api.coingecko.com/api/v3/"
    FEAR_GREED_API = "https://api.alternative.me/fng/"
    API_KEY = os.getenv('COINGECKO_API_KEY')
    GOOGLE_ANALYTICS_ID = os.getenv('GOOGLE_ANALYTICS_ID')
    # More conservative caching to stay well within 10,000 calls/month limit
    CACHE_TIMEOUT = 60  # minutes (increased from 35 to 60)
    CACHE_CLEANUP_AGE = 120  # minutes (increased from 60 to 120)
    # Rate limiting: 30 calls per minute = 1 call every 2 seconds
    API_RATE_LIMIT_DELAY = 2.1  # seconds (slightly over 2 to be safe)
    MAX_REQUESTS_PER_MINUTE = 30
    # API usage tracking
    MAX_MONTHLY_API_CALLS = 10000
    SAFETY_MARGIN = 0.8  # Use only 80% of available calls for safety

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self):
        self.cache = {
            'metadata': {
                'last_refresh': None,
                'next_refresh': None,
                'cache_hits': 0
            }
        }
        # API usage tracking
        self.api_calls_this_month = 0
        self.month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    def get(self, key):
        if key in self.cache:
            return self.cache[key]
        return None

    def set(self, key, value, timestamp):
        self.cache[key] = (value, timestamp)
        self.update_metadata(timestamp)

    def update_metadata(self, timestamp):
        self.cache['metadata'].update({
            'last_refresh': timestamp,
            'next_refresh': timestamp + timedelta(minutes=Config.CACHE_TIMEOUT)
        })

    def track_api_call(self):
        """Track API call usage and check monthly limits"""
        current_time = datetime.now()
        
        # Check if we're in a new month
        if current_time.month != self.month_start.month or current_time.year != self.month_start.year:
            self.api_calls_this_month = 0
            self.month_start = current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        self.api_calls_this_month += 1
        
        # Check if we're approaching the limit
        max_allowed = int(Config.MAX_MONTHLY_API_CALLS * Config.SAFETY_MARGIN)
        if self.api_calls_this_month >= max_allowed:
            return False
        
        return True

    def get_api_usage_stats(self):
        """Get current API usage statistics"""
        max_allowed = int(Config.MAX_MONTHLY_API_CALLS * Config.SAFETY_MARGIN)
        return {
            'calls_used': self.api_calls_this_month,
            'calls_remaining': max_allowed - self.api_calls_this_month,
            'month_start': self.month_start.strftime("%Y-%m-%d"),
            'usage_percentage': (self.api_calls_this_month / max_allowed) * 100
        }
    
    def get_stablecoin_ids(self):
        """Get cached stablecoin IDs or fetch if not available"""
        if 'stablecoin_ids' not in self.cache:
            # Fetch stablecoins once and cache them
            stablecoin_ids = set()
            stablecoin_page = 1
            stablecoin_per_page = 250
            
            while True:
                stablecoin_data = make_coingecko_request('coins/markets', {
                    'vs_currency': 'usd',
                    'category': 'stablecoins',
                    'per_page': stablecoin_per_page,
                    'page': stablecoin_page,
                    'order': 'market_cap_desc'
                })
                
                if not stablecoin_data or len(stablecoin_data) == 0:
                    break
                    
                for coin in stablecoin_data:
                    if 'id' in coin:
                        stablecoin_ids.add(coin['id'])
                
                if len(stablecoin_data) < stablecoin_per_page:
                    break
                    
                stablecoin_page += 1
                time.sleep(0.5)  # Rate limiting
                
                # Safety check to prevent infinite loops
                if stablecoin_page > 50:  # Maximum reasonable number of pages for stablecoins
                    break
            
            # Cache the stablecoin IDs for 24 hours (stablecoins don't change often)
            self.set('stablecoin_ids', stablecoin_ids, datetime.now())
        
        return self.cache['stablecoin_ids'][0]  # Return the set from (value, timestamp)

    def cleanup(self):
        current_time = datetime.now()
        expired_keys = []
        
        # Fix the cache cleanup logic
        for key, value in self.cache.items():
            if key != 'metadata':  # Skip the metadata entry
                try:
                    result, timestamp = value  # Safely unpack
                    if (current_time - timestamp).total_seconds() > Config.CACHE_CLEANUP_AGE * 60:
                        expired_keys.append(key)
                except ValueError:
                    # Skip malformed cache entries
                    expired_keys.append(key)
        
        # Remove expired keys
        for key in expired_keys:
            del self.cache[key]

cache_manager = CacheManager()

class RateLimiter:
    def __init__(self, max_requests_per_minute=30):
        self.max_requests = max_requests_per_minute
        self.requests = []
    
    def can_make_request(self):
        current_time = time.time()
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if current_time - req_time < 60]
        
        if len(self.requests) < self.max_requests:
            self.requests.append(current_time)
            return True
        return False
    
    def wait_if_needed(self):
        while not self.can_make_request():
            wait_time = 60 - (time.time() - self.requests[0])
            if wait_time > 0:
                time.sleep(min(wait_time, 1))  # Wait up to 1 second at a time
            else:
                time.sleep(1)  # Wait 1 second and check again

# Create rate limiter instance after class definition
rate_limiter = RateLimiter(Config.MAX_REQUESTS_PER_MINUTE)

def preload_cache():
    """Pre-load all cached data on app startup to ensure first visitor gets cached data"""
    try:
        # Pre-fetch all the data that will be needed
        get_market_data()
        get_eth_btc_ratio()
        get_bitcoin_rsi()
        get_btc_monthly_roi()
        get_top10_alts_performance()
        get_altcoin_volume_dominance()
        
    except Exception as e:
        pass  # Silently continue if pre-loading fails

def cache_with_timeout(timeout_minutes=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            current_time = datetime.now()
            
            # Update metadata on cache hit
            if cache_key in cache_manager.cache:
                result, timestamp = cache_manager.cache[cache_key]
                if current_time - timestamp < timedelta(minutes=timeout_minutes):
                    cache_manager.cache['metadata']['cache_hits'] += 1
                    return result
            
            # Update metadata on cache miss
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, current_time)
            return result
        return wrapper
    return decorator

class APIError(Exception):
    pass

def make_coingecko_request(endpoint, params=None):
    # Check API usage limits before making request
    if not cache_manager.track_api_call():
        return None
    
    # Wait for rate limiter to allow the request
    rate_limiter.wait_if_needed()
    
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": Config.API_KEY
    }
    
    try:
        response = requests.get(
            f"{Config.COINGECKO_API_URL}{endpoint}",
            headers=headers,
            params=params,
            timeout=10  # Add timeout
        )
        
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()
            
    except requests.RequestException as e:
        return None

@cache_with_timeout(60)
def get_market_data():
    data = make_coingecko_request('global')
    if data and 'data' in data:
        return data['data']
    return {
        'market_cap_percentage': {'btc': 0},
        'total_market_cap': {'usd': 0}
    }

@cache_with_timeout(60)
def get_eth_btc_ratio():
    data = make_coingecko_request('simple/price', {
        'ids': 'ethereum,bitcoin',
        'vs_currencies': 'usd'
    })
    
    if data and 'ethereum' in data and 'bitcoin' in data:
        eth_price = data['ethereum']['usd']
        btc_price = data['bitcoin']['usd']
        return eth_price / btc_price
    return 0

@cache_with_timeout(60)
def get_fear_greed_index():
    try:
        response = requests.get(Config.FEAR_GREED_API, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data or 'data' not in data or not data['data']:
            return {'value': '0', 'value_classification': 'Unknown'}
            
        return {
            'value': str(data['data'][0].get('value', '0')),
            'value_classification': data['data'][0].get('value_classification', 'Unknown')
        }
    except (requests.RequestException, ValueError, KeyError, IndexError) as e:
        return {'value': '0', 'value_classification': 'Unknown'}

@cache_with_timeout(60)
def get_bitcoin_rsi():
    try:
        # Get BTC/USD hourly OHLC data for more accurate RSI calculation
        # Using 14 days = 336 hours, but we need at least 14 periods for RSI
        # API Impact: This replaces the previous market_chart call, so no additional rate limit impact
        data = make_coingecko_request('coins/bitcoin/ohlc', {
            'vs_currency': 'usd',
            'days': '14'
        })
        
        if data and len(data) >= 14:
            # Extract closing prices (index 4 in OHLC data: [timestamp, open, high, low, close])
            prices = [candle[4] for candle in data]
            
            # Validate we have enough data points
            if len(prices) < 14:
                return None
                
            # Validate price data quality
            if any(price <= 0 for price in prices):
                return None
                
            # Calculate price changes
            deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            
            # Separate gains and losses
            gains = [delta if delta > 0 else 0 for delta in deltas]
            losses = [-delta if delta < 0 else 0 for delta in deltas]
            
            # Calculate initial simple moving average for first 14 periods
            initial_avg_gain = sum(gains[:14]) / 14
            initial_avg_loss = sum(losses[:14]) / 14
            
            # Calculate RSI using exponential moving average (standard method)
            # Smoothing factor: k = 2 / (period + 1) = 2 / (14 + 1) = 0.1333
            k = 2 / (14 + 1)
            
            # Initialize EMA values
            ema_gain = initial_avg_gain
            ema_loss = initial_avg_loss
            
            # Calculate EMA for remaining periods
            for i in range(14, len(gains)):
                ema_gain = (gains[i] * k) + (ema_gain * (1 - k))
                ema_loss = (losses[i] * k) + (ema_loss * (1 - k))
            
            # Calculate RSI
            if ema_loss == 0:
                rsi = 100
            else:
                rs = ema_gain / ema_loss
                rsi = 100 - (100 / (1 + rs))
            
            return round(rsi, 2)
            
    except Exception as e:
        return None
    
    return None

# Add this new function
def get_altcoin_dominance():
    try:
        market_data = get_market_data()
        if market_data:
            total_mcap = market_data['total_market_cap']['usd']
            btc_dominance = market_data['market_cap_percentage']['btc']
            
            # Calculate altcoin market cap (exclude Bitcoin)
            altcoin_mcap = total_mcap * (100 - btc_dominance) / 100
            
            # Calculate ratio (altcoin/total)
            ratio = altcoin_mcap / total_mcap
            return ratio
                
    except Exception as e:
        pass
    return None

@cache_with_timeout(60)
def get_btc_monthly_roi():
    try:
        data = make_coingecko_request('coins/bitcoin/market_chart', {
            'vs_currency': 'usd',
            'days': '30',
            'interval': 'daily'
        })
        
        if data and 'prices' in data:
            start_price = data['prices'][0][1]
            end_price = data['prices'][-1][1]
            roi = ((end_price - start_price) / start_price) * 100
            return roi
    except Exception as e:
        pass
    return None

@cache_with_timeout(60)
def get_top10_alts_performance():
    try:
        # Get cached stablecoin IDs
        stablecoin_ids = cache_manager.get_stablecoin_ids()
        
        # Get top coins and filter out BTC and stablecoins
        data = make_coingecko_request('coins/markets', {
            'vs_currency': 'btc',  # Price in BTC to compare against Bitcoin
            'order': 'market_cap_desc',
            'per_page': '20',  # Get more to ensure we have 10 after filtering
            'sparkline': 'false',
            'price_change_percentage': '30d'
        })
        
        if data:
            # Remove Bitcoin and stablecoins, get top 10 actual altcoins
            alts_data = [coin for coin in data 
                        if coin['id'] != 'bitcoin' and coin['id'] not in stablecoin_ids][:10]
            
            if alts_data:
                avg_performance = sum(coin['price_change_percentage_30d_in_currency'] 
                                    for coin in alts_data) / len(alts_data)
                return avg_performance
    except Exception as e:
        pass
    return None

@cache_with_timeout(60)
def get_altcoin_volume_dominance():
    try:
        # Use cached stablecoin IDs instead of fetching again
        stablecoin_ids = cache_manager.get_stablecoin_ids()
        
        # Add delay before starting volume calculations to respect rate limits
        time.sleep(1)  # 1 second delay before starting volume calculations
        
        # Now fetch all coins and calculate volume excluding stablecoins
        total_volume_usd = 0
        bitcoin_volume_usd = 0
        stablecoin_volume_usd = 0
        altcoin_volume_usd = 0
        page = 1
        per_page = 250
        
        while True:
            # Fetch page of coins with volume data
            data = make_coingecko_request('coins/markets', {
                'vs_currency': 'usd',
                'per_page': per_page,
                'page': page,
                'price_change_percentage': '24h',
                'order': 'market_cap_desc'
            })
            
            if not data or len(data) == 0:
                break
                
            # Process each coin in this page
            for coin in data:
                if 'total_volume' in coin and coin['total_volume']:
                    coin_volume = float(coin['total_volume'])
                    coin_id = coin.get('id', '')
                    
                    # Check if this is Bitcoin
                    if coin_id == 'bitcoin':
                        bitcoin_volume_usd = coin_volume
                        total_volume_usd += coin_volume
                    # Check if this is a stablecoin
                    elif coin_id in stablecoin_ids:
                        stablecoin_volume_usd += coin_volume
                        # Don't add to total volume since we're excluding stablecoins
                    else:
                        # This is an altcoin (not Bitcoin, not stablecoin)
                        altcoin_volume_usd += coin_volume
                        total_volume_usd += coin_volume
            
            # If we got less than per_page results, we've reached the end
            if len(data) < per_page:
                break
                
            page += 1
            
            # Add delay between pages to respect rate limits
            time.sleep(0.5)  # 500ms delay between volume calculation page requests
            
            # Safety check to prevent infinite loops
            if page > 100:  # Maximum reasonable number of pages
                break
        
        # Validate we have data
        if total_volume_usd <= 0:
            return None
            
        if bitcoin_volume_usd <= 0:
            return None
        
        # Calculate total volume including Bitcoin and altcoins (excluding stablecoins)
        total_volume_excluding_stablecoins = bitcoin_volume_usd + altcoin_volume_usd
        
        # Calculate altcoin volume dominance percentage (altcoins / (bitcoin + altcoins))
        altcoin_volume_dominance = (altcoin_volume_usd / total_volume_excluding_stablecoins) * 100
        
        # Validate the result
        if altcoin_volume_dominance < 0 or altcoin_volume_dominance > 100:
            return None
        
        return round(altcoin_volume_dominance, 2)
        
    except Exception as e:
        return None

@app.route('/')
def index():
    try:
        cache_manager.cleanup()
        
        # Get and validate all required data
        market_data = get_market_data()
        if not market_data:
            raise APIError("Failed to fetch market data")
        
        bitcoin_rsi = get_bitcoin_rsi()
        altcoin_dominance_ratio = get_altcoin_dominance()
        
        # Calculate indicators
        bitcoin_dominance = market_data['market_cap_percentage']['btc']
        total_market_cap = market_data['total_market_cap']['usd']
        btc_market_cap = bitcoin_dominance * total_market_cap / 100
        altcoin_market_cap = total_market_cap - btc_market_cap
        eth_btc_ratio = get_eth_btc_ratio()
        fear_greed = get_fear_greed_index()
        
        # Get additional indicators
        btc_monthly_roi = get_btc_monthly_roi()
        top10_alts_perf = get_top10_alts_performance()
        altcoin_volume_dominance = get_altcoin_volume_dominance()
        
        # Enhanced alt season detection with safe handling of None values
        is_alt_season = (
            bitcoin_dominance < 45 and          # Traditional indicator
            (eth_btc_ratio or 0) > 0.07 and    # Traditional indicator (safe None handling)
            int(fear_greed.get('value', '0')) > 65 and  # Traditional indicator
            (btc_monthly_roi or 0) < 0 and     # BTC showing weakness (safe None handling)
            (top10_alts_perf or 0) > 10 and    # Top alts outperforming (safe None handling)
            (altcoin_volume_dominance or 0) > 60        # High altcoin volume dominance (safe None handling)
        )
        
        # Add cache metadata
        cache_status = {
            'last_refresh': cache_manager.cache['metadata']['last_refresh'].strftime("%Y-%m-%d %H:%M:%S UTC") if cache_manager.cache['metadata']['last_refresh'] else "Never",
            'next_refresh': cache_manager.cache['metadata']['next_refresh'].strftime("%Y-%m-%d %H:%M:%S UTC") if cache_manager.cache['metadata']['next_refresh'] else "Unknown",
            'minutes_until_refresh': int((cache_manager.cache['metadata']['next_refresh'] - datetime.now()).total_seconds() / 60) if cache_manager.cache['metadata']['next_refresh'] else 0
        }
        
        # Add API usage statistics
        api_usage = cache_manager.get_api_usage_stats()
        
        return render_template('index.html',
                             bitcoin_dominance=bitcoin_dominance or 0,
                             altcoin_market_cap=altcoin_market_cap or 0,
                             eth_btc_ratio=eth_btc_ratio or 0,
                             fear_greed=fear_greed or {'value': '0', 'value_classification': 'Unknown'},
                             is_alt_season=is_alt_season,
                             bitcoin_rsi=bitcoin_rsi or 0,
                             altcoin_dominance_ratio=altcoin_dominance_ratio or 0,
                             btc_monthly_roi=btc_monthly_roi or 0,
                             top10_alts_perf=top10_alts_perf or 0,
                             volume_dominance=altcoin_volume_dominance or 0,
                             last_updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                             cache_status=cache_status,
                             api_usage=api_usage,
                             google_analytics_id=Config.GOOGLE_ANALYTICS_ID)
    except Exception as e:
        return render_template('error.html', error=str(e)), 500

@app.template_filter('number_format')
def number_format_filter(value):
    return "{:,.2f}".format(float(value))

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static', 'images'),
                             'icon.svg', mimetype='image/svg+xml')

@app.route('/changelog')
def changelog():
    return render_template('changelog.html',
                          google_analytics_id=Config.GOOGLE_ANALYTICS_ID)

# Add this route specifically for PythonAnywhere
@app.route('/static/<path:filename>')
def custom_static(filename):
    return send_from_directory('static', filename)

@app.route('/static/images/icon.svg')
def serve_icon():
    return send_from_directory('static/images', 'icon.svg', mimetype='image/svg+xml')

@app.route('/robots.txt')
def robots():
    return send_from_directory('static', 'robots.txt', mimetype='text/plain')

@app.route('/sitemap.xml')
def sitemap():
    return send_from_directory('static', 'sitemap.xml', mimetype='application/xml')

@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json', mimetype='application/json')

if __name__ == '__main__':
    # Pre-load cache on startup
    preload_cache()
    
    # Development - run on local network
    app.run(debug=True, host='0.0.0.0', port=5000)
else:
    # Production - pre-load cache when app starts
    preload_cache()
    
    # Production
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
