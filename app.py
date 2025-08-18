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

# Also log to console for immediate visibility
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

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
            try:
                if key == 'metadata':
                    return self.cache[key]
                else:
                    result, timestamp = self.cache[key]
                    return result
            except (ValueError, TypeError, IndexError):
                # Malformed cache entry - remove it
                logger.warning(f"Removing malformed cache entry for key: {key}")
                try:
                    del self.cache[key]
                except KeyError:
                    pass
                return None
        return None

    def set(self, key, value, timestamp):
        try:
            if not isinstance(timestamp, datetime):
                timestamp = datetime.now()
            self.cache[key] = (value, timestamp)
            self.update_metadata(timestamp)
        except Exception as e:
            logger.error(f"Failed to set cache entry for {key}: {str(e)}")
            # Don't let cache errors break the application

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
        
        # Fix: Access the first element of the tuple (value, timestamp)
        try:
            return self.cache['stablecoin_ids'][0]  # Return the set from (value, timestamp)
        except (KeyError, IndexError, TypeError):
            logger.error("Failed to retrieve stablecoin IDs from cache")
            return set()  # Return empty set as fallback

    def cleanup(self):
        current_time = datetime.now()
        expired_keys = []
        
        # Fix the cache cleanup logic
        for key, value in list(self.cache.items()):  # Use list() to avoid modification during iteration
            if key == 'metadata':  # Skip the metadata entry
                continue
                
            try:
                if isinstance(value, tuple) and len(value) == 2:
                    result, timestamp = value  # Safely unpack
                    if isinstance(timestamp, datetime):
                        if (current_time - timestamp).total_seconds() > Config.CACHE_CLEANUP_AGE * 60:
                            expired_keys.append(key)
                    else:
                        # Invalid timestamp - mark for removal
                        expired_keys.append(key)
                else:
                    # Malformed cache entry - mark for removal
                    expired_keys.append(key)
            except (ValueError, TypeError, AttributeError):
                # Skip malformed cache entries
                expired_keys.append(key)
        
        # Remove expired keys
        for key in expired_keys:
            try:
                del self.cache[key]
                logger.debug(f"Removed expired cache entry: {key}")
            except KeyError:
                pass  # Key already removed

    def is_cache_valid(self, key, timeout_minutes=60):
        """Check if a cache entry is valid and not expired"""
        if key not in self.cache:
            return False
            
        try:
            if key == 'metadata':
                return True  # Metadata is always valid
                
            result, timestamp = self.cache[key]
            if not isinstance(timestamp, datetime):
                return False
                
            current_time = datetime.now()
            return (current_time - timestamp).total_seconds() < timeout_minutes * 60
            
        except (ValueError, TypeError, IndexError):
            return False
    
    def get_cache_info(self):
        """Get information about cache state for debugging"""
        try:
            total_entries = len(self.cache)
            metadata_entries = 1 if 'metadata' in self.cache else 0
            data_entries = total_entries - metadata_entries
            
            # Count valid vs expired entries
            current_time = datetime.now()
            valid_entries = 0
            expired_entries = 0
            
            for key, value in self.cache.items():
                if key == 'metadata':
                    continue
                try:
                    if isinstance(value, tuple) and len(value) == 2:
                        result, timestamp = value
                        if isinstance(timestamp, datetime):
                            if (current_time - timestamp).total_seconds() < Config.CACHE_TIMEOUT * 60:
                                valid_entries += 1
                            else:
                                expired_entries += 1
                        else:
                            expired_entries += 1
                    else:
                        expired_entries += 1
                except:
                    expired_entries += 1
            
            return {
                'total_entries': total_entries,
                'metadata_entries': metadata_entries,
                'data_entries': data_entries,
                'valid_entries': valid_entries,
                'expired_entries': expired_entries,
                'cache_timeout_minutes': Config.CACHE_TIMEOUT
            }
        except Exception as e:
            logger.error(f"Failed to get cache info: {str(e)}")
            return {'error': str(e)}

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
            
            # Check cache first
            if cache_key in cache_manager.cache:
                try:
                    result, timestamp = cache_manager.cache[cache_key]
                    if current_time - timestamp < timedelta(minutes=timeout_minutes):
                        # Cache hit - update metadata safely
                        try:
                            cache_manager.cache['metadata']['cache_hits'] += 1
                        except (KeyError, TypeError):
                            pass  # Silently continue if metadata update fails
                        return result
                except (ValueError, TypeError, IndexError):
                    # Malformed cache entry - remove it and continue
                    logger.warning(f"Removing malformed cache entry for {cache_key}")
                    try:
                        del cache_manager.cache[cache_key]
                    except KeyError:
                        pass
            
            # Cache miss - call function and cache result
            try:
                result = func(*args, **kwargs)
                cache_manager.set(cache_key, result, current_time)
                return result
            except Exception as e:
                logger.error(f"Function {func.__name__} failed: {str(e)}")
                return None
        return wrapper
    return decorator

class APIError(Exception):
    pass

def make_coingecko_request(endpoint, params=None):
    # Check API usage limits before making request
    if not cache_manager.track_api_call():
        logger.warning(f"API call limit reached for endpoint: {endpoint}")
        return None
    
    # Wait for rate limiter to allow the request
    rate_limiter.wait_if_needed()
    
    headers = {
        "accept": "application/json"
    }
    
    # Only add API key if it exists
    if Config.API_KEY:
        headers["x-cg-demo-api-key"] = Config.API_KEY
        logger.info(f"Making API request to {endpoint} with API key")
    else:
        logger.warning(f"Making API request to {endpoint} WITHOUT API key - may hit rate limits")
    
    try:
        logger.info(f"Requesting: {Config.COINGECKO_API_URL}{endpoint} with params: {params}")
        response = requests.get(
            f"{Config.COINGECKO_API_URL}{endpoint}",
            headers=headers,
            params=params,
            timeout=15  # Increased timeout
        )
        
        logger.info(f"Response status: {response.status_code} for {endpoint}")
        
        if response.status_code == 429:  # Rate limited
            logger.warning(f"Rate limited for endpoint: {endpoint}")
            time.sleep(5)  # Wait 5 seconds before retrying
            return None
        elif response.status_code == 403:  # Forbidden (API key issues)
            logger.error(f"API access forbidden for endpoint: {endpoint}. Check API key.")
            return None
        elif response.status_code != 200:
            logger.error(f"Bad response for {endpoint}: {response.status_code} - {response.text[:200]}")
            return None
        
        response.raise_for_status()  # Raise exception for other bad status codes
        data = response.json()
        logger.info(f"Successfully got data from {endpoint}: {str(data)[:200]}...")
        return data
            
    except requests.RequestException as e:
        logger.error(f"Request failed for endpoint {endpoint}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error for endpoint {endpoint}: {str(e)}")
        return None

@cache_with_timeout(60)
def get_market_data():
    logger.info("Starting get_market_data()")
    data = make_coingecko_request('global')
    
    if data and 'data' in data:
        logger.info(f"Successfully got market data: BTC dominance = {data['data'].get('market_cap_percentage', {}).get('btc', 'N/A')}%")
        return data['data']
    
    logger.warning("Global market data failed")
    return None

@cache_with_timeout(60)
def get_eth_btc_ratio():
    logger.info("Starting get_eth_btc_ratio()")
    data = make_coingecko_request('simple/price', {
        'ids': 'ethereum,bitcoin',
        'vs_currencies': 'usd'
    })
    
    if data and 'ethereum' in data and 'bitcoin' in data:
        eth_price = data['ethereum']['usd']
        btc_price = data['bitcoin']['usd']
        ratio = eth_price / btc_price
        logger.info(f"Successfully got ETH/BTC ratio: ETH=${eth_price:,.2f}, BTC=${btc_price:,.2f}, ratio={ratio:.5f}")
        return ratio
    
    logger.warning("ETH/BTC ratio failed")
    return None

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
    logger.info("Starting get_btc_monthly_roi()")
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
            logger.info(f"Successfully calculated BTC monthly ROI: start=${start_price:,.2f}, end=${end_price:,.2f}, ROI={roi:.2f}%")
            return roi
        else:
            logger.warning(f"BTC market chart data missing or invalid: {data}")
    except Exception as e:
        logger.error(f"Error calculating BTC monthly ROI: {str(e)}")
    
    logger.warning("BTC monthly ROI calculation failed, returning None")
    return None

@cache_with_timeout(60)
def get_top10_alts_performance():
    logger.info("Starting get_top10_alts_performance()")
    try:
        # Get cached stablecoin IDs
        stablecoin_ids = cache_manager.get_stablecoin_ids()
        logger.info(f"Got {len(stablecoin_ids)} stablecoin IDs from cache")
        
        # Get top coins and filter out BTC and stablecoins
        data = make_coingecko_request('coins/markets', {
            'vs_currency': 'btc',  # Price in BTC to compare against Bitcoin
            'order': 'market_cap_desc',
            'per_page': '20',  # Get more to ensure we have 10 after filtering
            'sparkline': 'false',
            'price_change_percentage': '30d'
        })
        
        if data:
            logger.info(f"Got {len(data)} coins from markets API")
            # Remove Bitcoin and stablecoins, get top 10 actual altcoins
            alts_data = [coin for coin in data 
                        if coin['id'] != 'bitcoin' and coin['id'] not in stablecoin_ids][:10]
            
            logger.info(f"Filtered to {len(alts_data)} altcoins (excluding BTC and stablecoins)")
            
            if alts_data:
                avg_performance = sum(coin['price_change_percentage_30d_in_currency'] 
                                    for coin in alts_data) / len(alts_data)
                logger.info(f"Calculated average performance: {avg_performance:.2f}% for {len(alts_data)} altcoins")
                return avg_performance
            else:
                logger.warning("No altcoins found after filtering")
        else:
            logger.warning("No data received from markets API")
    except Exception as e:
        logger.error(f"Error calculating top 10 alts performance: {str(e)}")
    
    logger.warning("Top 10 alts performance calculation failed, returning None")
    return None

@cache_with_timeout(60)
def get_altcoin_volume_dominance():
    try:
        # Use cached stablecoin IDs instead of fetching again
        stablecoin_ids = cache_manager.get_stablecoin_ids()
        
        # Add delay before starting volume calculations to respect rate limits
        time.sleep(1)  # 1 second delay before starting volume calculations
        
        # Fetch all available pages to get complete volume data
        # Target: 28 calls per minute = 1 call every 2.14 seconds
        total_volume_usd = 0
        bitcoin_volume_usd = 0
        stablecoin_volume_usd = 0
        altcoin_volume_usd = 0
        page = 1
        per_page = 250
        total_pages_processed = 0
        
        logger.info("Starting volume dominance calculation - targeting 28 calls per minute")
        
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
                logger.info(f"Reached end of data at page {page}")
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
            
            total_pages_processed += 1
            
            # Log progress every 5 pages
            if total_pages_processed % 5 == 0:
                logger.info(f"Volume calculation: processed {total_pages_processed} pages, BTC vol: ${bitcoin_volume_usd:,.0f}, Alt vol: ${altcoin_volume_usd:,.0f}")
            
            # If we got less than per_page results, we've reached the end
            if len(data) < per_page:
                logger.info(f"Reached last page with {len(data)} coins (less than {per_page})")
                break
                
            page += 1
            
            # Calculate delay to achieve exactly 28 calls per minute
            # 60 seconds / 28 calls = 2.14 seconds per call
            # Subtract estimated processing time (~0.1 seconds) to get accurate timing
            delay_seconds = (60.0 / 28.0) - 0.1  # 2.04 seconds between calls
            time.sleep(delay_seconds)
            
            # Safety check to prevent infinite loops (but allow many pages)
            if page > 500:  # Increased limit to allow more pages
                logger.warning(f"Reached safety limit of {page} pages, stopping to prevent infinite loop")
                break
        
        logger.info(f"Completed volume calculation: processed {total_pages_processed} pages total")
        
        # Validate we have data
        if total_volume_usd <= 0:
            logger.warning("No volume data found in any pages")
            return None
            
        if bitcoin_volume_usd <= 0:
            logger.warning("No Bitcoin volume data found")
            return None
        
        # Calculate total volume including Bitcoin and altcoins (excluding stablecoins)
        total_volume_excluding_stablecoins = bitcoin_volume_usd + altcoin_volume_usd
        
        # Calculate altcoin volume dominance percentage (altcoins / (bitcoin + altcoins))
        altcoin_volume_dominance = (altcoin_volume_usd / total_volume_excluding_stablecoins) * 100
        
        # Validate the result
        if altcoin_volume_dominance < 0 or altcoin_volume_dominance > 100:
            logger.warning(f"Invalid volume dominance calculated: {altcoin_volume_dominance}%")
            return None
        
        logger.info(f"Successfully calculated volume dominance: {altcoin_volume_dominance:.2f}% from {total_pages_processed} pages")
        return round(altcoin_volume_dominance, 2)
        
    except Exception as e:
        logger.error(f"Error calculating altcoin volume dominance: {str(e)}")
        return None

@app.route('/')
def index():
    try:
        logger.info("=== Starting index route ===")
        
        # Log cache status before cleanup
        cache_info_before = cache_manager.get_cache_info()
        logger.info(f"Cache status before cleanup: {cache_info_before}")
        
        cache_manager.cleanup()
        
        # Log cache status after cleanup
        cache_info_after = cache_manager.get_cache_info()
        logger.info(f"Cache status after cleanup: {cache_info_after}")
        
        # Get and validate all required data
        market_data = get_market_data()
        if not market_data:
            logger.error("Failed to fetch market data - all values will be None")
            # Set all market-related values to None
            bitcoin_dominance = None
            total_market_cap = None
            btc_market_cap = None
            altcoin_market_cap = None
            altcoin_dominance_ratio = None
        else:
            # Calculate indicators
            bitcoin_dominance = market_data['market_cap_percentage']['btc']
            total_market_cap = market_data['total_market_cap']['usd']
            btc_market_cap = bitcoin_dominance * total_market_cap / 100
            altcoin_market_cap = total_market_cap - btc_market_cap
            altcoin_dominance_ratio = get_altcoin_dominance()
        
        eth_btc_ratio = get_eth_btc_ratio()
        fear_greed = get_fear_greed_index()
        bitcoin_rsi = get_bitcoin_rsi()
        
        # Get additional indicators
        btc_monthly_roi = get_btc_monthly_roi()
        top10_alts_perf = get_top10_alts_performance()
        altcoin_volume_dominance = get_altcoin_volume_dominance()
        
        # Log all calculated values
        logger.info(f"Calculated values:")
        logger.info(f"  Bitcoin Dominance: {bitcoin_dominance}%")
        logger.info(f"  Total Market Cap: ${total_market_cap:,.0f}" if total_market_cap else "  Total Market Cap: N/A")
        logger.info(f"  BTC Market Cap: ${btc_market_cap:,.0f}" if btc_market_cap else "  BTC Market Cap: N/A")
        logger.info(f"  Altcoin Market Cap: ${altcoin_market_cap:,.0f}" if altcoin_market_cap else "  Altcoin Market Cap: N/A")
        logger.info(f"  ETH/BTC Ratio: {eth_btc_ratio}")
        logger.info(f"  Fear & Greed: {fear_greed}")
        logger.info(f"  BTC Monthly ROI: {btc_monthly_roi}%")
        logger.info(f"  Top 10 Alts Performance: {top10_alts_perf}%")
        logger.info(f"  Altcoin Volume Dominance: {altcoin_volume_dominance}%")
        logger.info(f"  Bitcoin RSI: {bitcoin_rsi}")
        logger.info(f"  Altcoin Dominance Ratio: {altcoin_dominance_ratio}")
        
        # Enhanced alt season detection with safe handling of None values
        is_alt_season = (
            (bitcoin_dominance or 0) < 45 and          # Traditional indicator
            (eth_btc_ratio or 0) > 0.07 and    # Traditional indicator (safe None handling)
            int(fear_greed.get('value', '0')) > 65 and  # Traditional indicator
            (btc_monthly_roi or 0) < 0 and     # BTC showing weakness (safe None handling)
            (top10_alts_perf or 0) > 10 and    # Top alts outperforming (safe None handling)
            (altcoin_volume_dominance or 0) > 60        # High altcoin volume dominance (safe None handling)
        )
        
        logger.info(f"Alt Season Detection: {is_alt_season}")
        
        # Add cache metadata
        cache_status = {
            'last_refresh': cache_manager.cache['metadata']['last_refresh'].strftime("%Y-%m-%d %H:%M:%S UTC") if cache_manager.cache['metadata']['last_refresh'] else "Never",
            'next_refresh': cache_manager.cache['metadata']['next_refresh'].strftime("%Y-%m-%d %H:%M:%S UTC") if cache_manager.cache['metadata']['next_refresh'] else "Unknown",
            'minutes_until_refresh': int((cache_manager.cache['metadata']['next_refresh'] - datetime.now()).total_seconds() / 60) if cache_manager.cache['metadata']['next_refresh'] else 0
        }
        
        # Add API usage statistics
        api_usage = cache_manager.get_api_usage_stats()
        
        logger.info("=== Rendering template ===")
        return render_template('index.html',
                             bitcoin_dominance=bitcoin_dominance,
                             altcoin_market_cap=altcoin_market_cap,
                             eth_btc_ratio=eth_btc_ratio,
                             fear_greed=fear_greed or {'value': '0', 'value_classification': 'Unknown'},
                             is_alt_season=is_alt_season,
                             bitcoin_rsi=bitcoin_rsi,
                             altcoin_dominance_ratio=altcoin_dominance_ratio,
                             btc_monthly_roi=btc_monthly_roi,
                             top10_alts_perf=top10_alts_perf,
                             volume_dominance=altcoin_volume_dominance,
                             last_updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                             cache_status=cache_status,
                             api_usage=api_usage,
                             google_analytics_id=Config.GOOGLE_ANALYTICS_ID)
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}", exc_info=True)
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
