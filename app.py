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
    # Updated to 2 pulls per day instead of hourly
    CACHE_TIMEOUT = 720  # minutes (12 hours = 720 minutes)
    CACHE_CLEANUP_AGE = 1440  # minutes (24 hours = 1440 minutes)
    # Rate limiting: 30 calls per minute = 1 call every 2 seconds
    API_RATE_LIMIT_DELAY = 2.1  # seconds (slightly over 2 to be safe)
    MAX_REQUESTS_PER_MINUTE = 30
    # API usage tracking
    MAX_MONTHLY_API_CALLS = 10000
    SAFETY_MARGIN = 0.8  # Use only 80% of available calls for safety
    # Daily pull schedule (UTC times)
    PULL_TIMES = [14, 22]  # 2 PM and 10 PM UTC

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
        self.month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
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
        current_time = datetime.now(timezone.utc)
        
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
            self.set('stablecoin_ids', stablecoin_ids, datetime.now(timezone.utc))
        
        return self.cache['stablecoin_ids'][0]  # Return the set from (value, timestamp)

    def cleanup(self):
        current_time = datetime.now(timezone.utc)
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
            current_time = datetime.now(timezone.utc)
            
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

def is_time_for_pull():
    """Check if it's time for a scheduled data pull based on UTC time"""
    current_utc = datetime.now(timezone.utc)
    current_hour = current_utc.hour
    
    # Check if current hour matches any of our pull times
    if current_hour in Config.PULL_TIMES:
        # Only pull once per hour (avoid multiple pulls in the same hour)
        cache_key = f"pull_completed_{current_utc.strftime('%Y-%m-%d_%H')}"
        if cache_key not in cache_manager.cache:
            logger.info(f"🕐 SCHEDULED PULL: It's {current_hour}:00 UTC - time for data refresh")
            return True
        else:
            logger.info(f"⏭️  PULL SKIPPED: Already completed pull for {current_hour}:00 UTC")
            return False
    
    return False

def mark_pull_completed():
    """Mark that a pull has been completed for the current hour"""
    current_utc = datetime.now(timezone.utc)
    cache_key = f"pull_completed_{current_utc.strftime('%Y-%m-%d_%H')}"
    # Cache for 2 hours to ensure we don't pull again in the same hour
    cache_manager.set(cache_key, True, current_utc)
    logger.info(f"✅ PULL COMPLETED: Marked pull as completed for {current_utc.strftime('%Y-%m-%d %H:00 UTC')}")

def get_next_pull_time():
    """Get the next scheduled pull time"""
    current_utc = datetime.now(timezone.utc)
    current_hour = current_utc.hour
    
    # Find next pull time
    for pull_hour in sorted(Config.PULL_TIMES):
        if pull_hour > current_hour:
            next_pull = current_utc.replace(hour=pull_hour, minute=0, second=0, microsecond=0)
            return next_pull
    
    # If no more pulls today, get first pull tomorrow
    tomorrow = current_utc + timedelta(days=1)
    next_pull = tomorrow.replace(hour=min(Config.PULL_TIMES), minute=0, second=0, microsecond=0)
    return next_pull

def log_api_usage():
    """Log current API usage statistics"""
    usage_stats = cache_manager.get_api_usage_stats()
    logger.info(f"📊 API USAGE: {usage_stats['calls_used']}/{usage_stats['calls_remaining']} calls used this month ({usage_stats['usage_percentage']:.1f}%)")
    
    # Calculate daily average
    days_elapsed = (datetime.now(timezone.utc) - cache_manager.month_start).days + 1
    daily_average = usage_stats['calls_used'] / days_elapsed
    projected_monthly = daily_average * 30
    
    logger.info(f"📈 PROJECTION: {daily_average:.1f} calls/day average, projected {projected_monthly:.0f} calls/month")
    
    if projected_monthly > 8000:
        logger.warning(f"⚠️  WARNING: Projected monthly usage ({projected_monthly:.0f}) exceeds safety limit (8000)")
    else:
        logger.info(f"✅ SAFE: Projected monthly usage ({projected_monthly:.0f}) within safety limit (8000)")

def log_pull_schedule():
    """Log the current pull schedule and next pull time"""
    next_pull = get_next_pull_time()
    time_until_pull = next_pull - datetime.now(timezone.utc)
    hours_until = time_until_pull.total_seconds() / 3600
    
    logger.info(f"🕐 PULL SCHEDULE: Next pull at {next_pull.strftime('%Y-%m-%d %H:%M UTC')} (in {hours_until:.1f} hours)")
    logger.info(f"📅 DAILY PULLS: {Config.PULL_TIMES} UTC ({', '.join([f'{h}:00' for h in Config.PULL_TIMES])})")

def check_health_status():
    """Check the actual health of the system without exposing sensitive data"""
    health_checks = {
        'api_connectivity': False,
        'data_freshness': False,
        'cache_functionality': False,
        'rate_limit_status': False
    }
    
    try:
        # Check API connectivity (test with a simple, safe endpoint)
        test_response = make_coingecko_request('ping')
        health_checks['api_connectivity'] = test_response is not None
        
        # Check data freshness (ensure cache has recent data)
        if cache_manager.cache['metadata']['last_refresh']:
            time_since_refresh = datetime.now(timezone.utc) - cache_manager.cache['metadata']['last_refresh']
            # Data is fresh if refreshed within last 13 hours (allowing 1 hour buffer)
            health_checks['data_freshness'] = time_since_refresh.total_seconds() < (13 * 3600)
        
        # Check cache functionality (ensure cache has data)
        health_checks['cache_functionality'] = len(cache_manager.cache) > 1  # More than just metadata
        
        # Check rate limit status (ensure we're not exceeding limits)
        usage_stats = cache_manager.get_api_usage_stats()
        max_allowed = int(Config.MAX_MONTHLY_API_CALLS * Config.SAFETY_MARGIN)
        health_checks['rate_limit_status'] = usage_stats['calls_used'] < max_allowed
        
        # Determine overall health
        healthy_checks = sum(health_checks.values())
        total_checks = len(health_checks)
        
        if healthy_checks == total_checks:
            return 'healthy', health_checks
        elif healthy_checks >= total_checks * 0.75:  # 75% or more checks pass
            return 'degraded', health_checks
        else:
            return 'unhealthy', health_checks
            
    except Exception as e:
        logger.error(f"❌ HEALTH CHECK ERROR: {str(e)}")
        return 'error', health_checks

@app.route('/')
def index():
    try:
        # Log current status
        logger.info(f"🌐 WEBSITE ACCESS: User accessed website at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Check if it's time for a scheduled pull
        if is_time_for_pull():
            logger.info("🚀 STARTING SCHEDULED DATA PULL")
            log_api_usage()
            
            # Force cache refresh by clearing old data
            cache_manager.cleanup()
            
            # Pre-load fresh data
            try:
                preload_cache()
                logger.info("✅ SCHEDULED PULL: Successfully refreshed all data")
                mark_pull_completed()
            except Exception as e:
                logger.error(f"❌ SCHEDULED PULL FAILED: {str(e)}")
        else:
            logger.info("💾 SERVING CACHED DATA: Using existing cache")
        
        # Log pull schedule
        log_pull_schedule()
        
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
            'minutes_until_refresh': int((cache_manager.cache['metadata']['next_refresh'] - datetime.now(timezone.utc)).total_seconds() / 60) if cache_manager.cache['metadata']['next_refresh'] else 0,
            'next_scheduled_pull': get_next_pull_time().strftime("%Y-%m-%d %H:%M UTC"),
            'pull_schedule': f"{', '.join([f'{h}:00' for h in Config.PULL_TIMES])} UTC"
        }
        
        # Add API usage statistics
        api_usage = cache_manager.get_api_usage_stats()
        
        # Log successful page render
        logger.info(f"✅ PAGE RENDERED: Successfully rendered page with alt season status: {is_alt_season}")
        
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
        logger.error(f"❌ PAGE ERROR: {str(e)}")
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

@app.route('/status')
def status():
    """Status endpoint for monitoring on PythonAnywhere"""
    try:
        current_utc = datetime.now(timezone.utc)
        next_pull = get_next_pull_time()
        time_until_pull = next_pull - current_utc
        hours_until = time_until_pull.total_seconds() / 3600
        
        # Get API usage stats
        api_usage = cache_manager.get_api_usage_stats()
        
        # Calculate daily average and projection
        days_elapsed = (current_utc - cache_manager.month_start).days + 1
        daily_average = api_usage['calls_used'] / days_elapsed
        projected_monthly = daily_average * 30
        
        # Check if it's time for a pull
        should_pull = is_time_for_pull()
        
        # Perform actual health check
        health_status, health_details = check_health_status()
        
        status_data = {
            'timestamp': current_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            'status': health_status,
            'health_checks': {
                'api_connectivity': health_details['api_connectivity'],
                'data_freshness': health_details['data_freshness'],
                'cache_functionality': health_details['cache_functionality'],
                'rate_limit_status': health_details['rate_limit_status']
            },
            'pull_schedule': {
                'daily_times': Config.PULL_TIMES,
                'next_pull': next_pull.strftime("%Y-%m-%d %H:%M UTC"),
                'hours_until_next': round(hours_until, 2),
                'should_pull_now': should_pull,
                'pull_times_formatted': f"{', '.join([f'{h}:00' for h in Config.PULL_TIMES])} UTC"
            },
            'cache': {
                'last_refresh': cache_manager.cache['metadata']['last_refresh'].strftime("%Y-%m-%d %H:%M:%S UTC") if cache_manager.cache['metadata']['last_refresh'] else "Never",
                'next_refresh': cache_manager.cache['metadata']['next_refresh'].strftime("%Y-%m-%d %H:%M:%S UTC") if cache_manager.cache['metadata']['next_refresh'] else "Unknown",
                'cache_hits': cache_manager.cache['metadata']['cache_hits'],
                'timeout_minutes': Config.CACHE_TIMEOUT
            },
            'api_usage': {
                'calls_used': api_usage['calls_used'],
                'calls_remaining': api_usage['calls_remaining'],
                'month_start': api_usage['month_start'],
                'usage_percentage': round(api_usage['usage_percentage'], 1),
                'daily_average': round(daily_average, 1),
                'projected_monthly': round(projected_monthly, 0),
                'safety_status': 'safe' if projected_monthly <= 8000 else 'warning'
            },
            'configuration': {
                'cache_timeout_minutes': Config.CACHE_TIMEOUT,
                'cache_cleanup_age_minutes': Config.CACHE_CLEANUP_AGE,
                'max_requests_per_minute': Config.MAX_REQUESTS_PER_MINUTE,
                'max_monthly_calls': Config.MAX_MONTHLY_API_CALLS,
                'safety_margin': Config.SAFETY_MARGIN
            }
        }
        
        # Log status check with health status
        logger.info(f"📊 STATUS CHECK: Status endpoint accessed at {current_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} - Health: {health_status}")
        
        return jsonify(status_data)
        
    except Exception as e:
        logger.error(f"❌ STATUS ERROR: {str(e)}")
        return jsonify({
            'timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            'status': 'error',
            'error': 'Internal server error'  # Don't expose actual error details
        }), 500

if __name__ == '__main__':
    # Log startup information
    logger.info("🚀 APPLICATION STARTING: Alt Season Dashboard")
    logger.info(f"📅 PULL SCHEDULE: {Config.PULL_TIMES} UTC ({', '.join([f'{h}:00' for h in Config.PULL_TIMES])})")
    logger.info(f"⏰ CACHE TIMEOUT: {Config.CACHE_TIMEOUT} minutes ({Config.CACHE_TIMEOUT/60:.1f} hours)")
    logger.info(f"🔧 API RATE LIMIT: {Config.MAX_REQUESTS_PER_MINUTE} calls/minute")
    logger.info(f"📊 MONTHLY LIMIT: {Config.MAX_MONTHLY_API_CALLS} calls (safety: {int(Config.MAX_MONTHLY_API_CALLS * Config.SAFETY_MARGIN)})")
    
    # Pre-load cache on startup
    logger.info("🔄 PRE-LOADING CACHE: Fetching initial data...")
    preload_cache()
    logger.info("✅ CACHE PRE-LOADED: Initial data fetched successfully")
    
    # Log initial status
    log_pull_schedule()
    log_api_usage()
    
    # Development - run on local network
    logger.info("🌐 STARTING DEVELOPMENT SERVER: http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
else:
    # Production - pre-load cache when app starts
    logger.info("🚀 PRODUCTION STARTUP: Alt Season Dashboard")
    logger.info(f"📅 PULL SCHEDULE: {Config.PULL_TIMES} UTC ({', '.join([f'{h}:00' for h in Config.PULL_TIMES])})")
    logger.info(f"⏰ CACHE TIMEOUT: {Config.CACHE_TIMEOUT} minutes ({Config.CACHE_TIMEOUT/60:.1f} hours)")
    
    # Pre-load cache when app starts
    logger.info("🔄 PRE-LOADING CACHE: Fetching initial data...")
    preload_cache()
    logger.info("✅ CACHE PRE-LOADED: Initial data fetched successfully")
    
    # Log initial status
    log_pull_schedule()
    log_api_usage()
    
    # Production
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
