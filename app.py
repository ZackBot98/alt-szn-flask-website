from flask import Flask, jsonify, render_template, send_from_directory
from datetime import datetime, timezone, timedelta
import requests
import time
from functools import wraps
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
class Config:
    COINGECKO_API_URL = "https://api.coingecko.com/api/v3/"
    FEAR_GREED_API = "https://api.alternative.me/fng/"
    API_KEY = os.getenv('COINGECKO_API_KEY')
    GOOGLE_ANALYTICS_ID = os.getenv('GOOGLE_ANALYTICS_ID')
    CACHE_TIMEOUT = 35  # minutes
    CACHE_CLEANUP_AGE = 60  # minutes
    API_RATE_LIMIT_DELAY = 1  # seconds

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

def cache_with_timeout(timeout_minutes=35):
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
        
        time.sleep(Config.API_RATE_LIMIT_DELAY)
        
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()
            
    except requests.RequestException as e:
        print(f"API Request Error: {str(e)}")
        return None

@cache_with_timeout(35)
def get_market_data():
    data = make_coingecko_request('global')
    if data and 'data' in data:
        return data['data']
    return {
        'market_cap_percentage': {'btc': 0},
        'total_market_cap': {'usd': 0}
    }

@cache_with_timeout(35)
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

@cache_with_timeout(35)
def get_fear_greed_index():
    try:
        response = requests.get(Config.FEAR_GREED_API)
        data = response.json()
        return {
            'value': data['data'][0]['value'],
            'value_classification': data['data'][0]['value_classification']
        }
    except:
        return {'value': '0', 'value_classification': 'Unknown'}

@cache_with_timeout(35)
def get_bitcoin_rsi():
    try:
        # Get BTC/USD price data with RSI indicator
        data = make_coingecko_request('coins/bitcoin/market_chart', {
            'vs_currency': 'usd',
            'days': '14',  # For 14-day RSI
            'interval': 'daily'
        })
        
        if data and 'prices' in data:
            prices = [price[1] for price in data['prices']]
            
            # Calculate RSI
            deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            gains = [delta if delta > 0 else 0 for delta in deltas]
            losses = [-delta if delta < 0 else 0 for delta in deltas]
            
            # Calculate average gains and losses
            avg_gain = sum(gains) / len(gains)
            avg_loss = sum(losses) / len(losses)
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            return rsi
    except Exception as e:
        print(f"Error calculating RSI: {str(e)}")
        return None
    
    return None

# Add this new function
def get_top100_vs_btc():
    try:
        market_data = get_market_data()
        if market_data:
            total_mcap = market_data['total_market_cap']['usd']
            btc_dominance = market_data['market_cap_percentage']['btc']
            
            # Calculate altcoin market cap (TOTAL2)
            altcoin_mcap = total_mcap * (100 - btc_dominance) / 110
            
            # Calculate ratio (TOTAL2/TOTAL)
            ratio = altcoin_mcap / total_mcap
            return ratio
                
    except Exception as e:
        print(f"Error calculating TOTAL2/TOTAL ratio: {str(e)}")
    return None

@cache_with_timeout(35)
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
        print(f"Error calculating BTC monthly ROI: {str(e)}")
    return None

@cache_with_timeout(35)
def get_top10_alts_performance():
    try:
        # Get top 11 coins (including BTC)
        data = make_coingecko_request('coins/markets', {
            'vs_currency': 'btc',  # Price in BTC to compare against Bitcoin
            'order': 'market_cap_desc',
            'per_page': '11',
            'sparkline': 'false',
            'price_change_percentage': '30d'
        })
        
        if data:
            # Remove Bitcoin and get average performance
            alts_data = [coin for coin in data if coin['id'] != 'bitcoin']
            avg_performance = sum(coin['price_change_percentage_30d_in_currency'] 
                                for coin in alts_data) / len(alts_data)
            return avg_performance
    except Exception as e:
        print(f"Error calculating top 10 alts performance: {str(e)}")
    return None

@cache_with_timeout(35)
def get_altcoin_volume_dominance():
    try:
        data = make_coingecko_request('global')
        if data and 'data' in data:
            total_volume = data['data']['total_volume']['usd']
            btc_data = make_coingecko_request('simple/price', {
                'ids': 'bitcoin',
                'vs_currencies': 'usd'
            })
            
            if btc_data and 'bitcoin' in btc_data:
                btc_price = btc_data['bitcoin']['usd']
                btc_volume = data['data']['total_volume']['btc'] * btc_price
                
                altcoin_volume = total_volume - btc_volume
                volume_dominance = (altcoin_volume / total_volume) * 100
                return volume_dominance
    except Exception as e:
        print(f"Error calculating altcoin volume dominance: {str(e)}")
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
        top100_ratio = get_top100_vs_btc()
        
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
        volume_dominance = get_altcoin_volume_dominance()
        
        # Enhanced alt season detection
        is_alt_season = (
            bitcoin_dominance < 45 and          # Traditional indicator
            eth_btc_ratio > 0.07 and           # Traditional indicator
            int(fear_greed['value']) > 65 and  # Traditional indicator
            (btc_monthly_roi or 0) < 0 and     # BTC showing weakness
            (top10_alts_perf or 0) > 10 and    # Top alts outperforming
            (volume_dominance or 0) > 60        # High altcoin volume
        )
        
        # Add cache metadata
        cache_status = {
            'last_refresh': cache_manager.cache['metadata']['last_refresh'].strftime("%Y-%m-%d %H:%M:%S UTC") if cache_manager.cache['metadata']['last_refresh'] else "Never",
            'next_refresh': cache_manager.cache['metadata']['next_refresh'].strftime("%Y-%m-%d %H:%M:%S UTC") if cache_manager.cache['metadata']['next_refresh'] else "Unknown",
            'minutes_until_refresh': int((cache_manager.cache['metadata']['next_refresh'] - datetime.now()).total_seconds() / 60) if cache_manager.cache['metadata']['next_refresh'] else 0
        }
        
        return render_template('index.html',
                             bitcoin_dominance=bitcoin_dominance,
                             altcoin_market_cap=altcoin_market_cap,
                             eth_btc_ratio=eth_btc_ratio,
                             fear_greed=fear_greed,
                             is_alt_season=is_alt_season,
                             bitcoin_rsi=bitcoin_rsi,
                             top100_ratio=top100_ratio,
                             btc_monthly_roi=btc_monthly_roi,
                             top10_alts_perf=top10_alts_perf,
                             volume_dominance=volume_dominance,
                             last_updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                             cache_status=cache_status,
                             google_analytics_id=Config.GOOGLE_ANALYTICS_ID)
    except Exception as e:
        print(f"Error in index route: {str(e)}")
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

if __name__ == '__main__':
    # Development - run on local network
    app.run(debug=True, host='0.0.0.0', port=5000)
else:
    # Production
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
