"""
Real-time feature engineering for fraud detection.
Uses Redis to maintain user transaction history and compute rolling features.
"""
import redis
import os
from datetime import datetime, timedelta
from urllib.parse import urlparse
import json


def get_redis_client():
    """Get Redis client, supporting Railway REDIS_URL or individual vars."""
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        parsed = urlparse(redis_url)
        return redis.Redis(
            host=parsed.hostname,
            port=parsed.port,
            password=parsed.password,
            decode_responses=True,
        )
    else:
        return redis.Redis(
            host=os.environ.get("REDIS_HOST"),
            port=int(os.environ.get("REDIS_PORT")),
            password=os.environ.get("REDIS_PASSWORD"),
            decode_responses=True,
        )


class FeatureEngine:
    """Real-time feature engineering using Redis for state management."""
    
    def __init__(self, redis_client=None):
        self.r = redis_client or get_redis_client()
        self.user_prefix = "user:"
        self.txn_prefix = "txn:"
    
    def _get_user_key(self, user_id):
        return f"{self.user_prefix}{user_id}"
    
    def _get_txn_key(self, transaction_id):
        return f"{self.txn_prefix}{transaction_id}"
    
    def _store_transaction(self, txn):
        """Store transaction in Redis for history tracking."""
        user_id = txn["user_id"]
        transaction_id = txn["transaction_id"]
        timestamp = datetime.fromisoformat(txn["timestamp"])
        
        # Store transaction details
        txn_key = self._get_txn_key(transaction_id)
        self.r.setex(
            txn_key,
            timedelta(days=7).total_seconds(),
            json.dumps(txn)
        )
        
        # Add to user's transaction sorted set (score = timestamp)
        user_key = self._get_user_key(user_id)
        self.r.zadd(user_key, {transaction_id: timestamp.timestamp()})
        
        # Store user's last transaction info
        last_txn_data = {
            "timestamp": txn["timestamp"],
            "country_code": txn["country_code"],
            "merchant_category": txn["merchant_category"],
            "amount": txn["amount"],
        }
        self.r.setex(
            f"{user_key}:last",
            timedelta(days=7).total_seconds(),
            json.dumps(last_txn_data)
        )
        
        # Track seen merchants for user
        merchant_key = f"{user_key}:merchants"
        self.r.sadd(merchant_key, txn["merchant_category"])
        self.r.expire(merchant_key, timedelta(days=7).total_seconds())
    
    def enrich(self, txn):
        """
        Enrich raw transaction with features required by the model.
        
        Args:
            txn: Dictionary with transaction data (transaction_id, user_id, timestamp,
                 amount, merchant_category, payment_method, country_code)
        
        Returns:
            Dictionary with all features needed for fraud prediction
        """
        user_id = txn["user_id"]
        timestamp = datetime.fromisoformat(txn["timestamp"])
        amount = float(txn["amount"])
        user_key = self._get_user_key(user_id)
        
        # Basic temporal features
        hour = timestamp.hour
        is_night = 1 if hour >= 22 or hour <= 6 else 0
        is_weekend = 1 if timestamp.weekday() >= 5 else 0
        
        # Get user's transaction history
        now_ts = timestamp.timestamp()
        one_hour_ago = (timestamp - timedelta(hours=1)).timestamp()
        one_day_ago = (timestamp - timedelta(hours=24)).timestamp()
        seven_days_ago = (timestamp - timedelta(days=7)).timestamp()
        
        # Get transaction IDs in time windows
        txn_ids_1h = self.r.zrangebyscore(
            user_key, one_hour_ago, now_ts, withscores=False
        )
        txn_ids_24h = self.r.zrangebyscore(
            user_key, one_day_ago, now_ts, withscores=False
        )
        txn_ids_7d = self.r.zrangebyscore(
            user_key, seven_days_ago, now_ts, withscores=False
        )
        
        # Calculate transaction counts
        txn_count_1h = len(txn_ids_1h)
        txn_count_24h = len(txn_ids_24h)
        
        # Calculate average amount in last 7 days
        amounts_7d = []
        for txn_id in txn_ids_7d:
            txn_data = self.r.get(self._get_txn_key(txn_id))
            if txn_data:
                txn_obj = json.loads(txn_data)
                amounts_7d.append(float(txn_obj["amount"]))
        
        avg_amount_7d = sum(amounts_7d) / len(amounts_7d) if amounts_7d else amount
        amount_deviation = abs(amount - avg_amount_7d)
        high_amount_flag = 1 if amount > avg_amount_7d * 3 else 0
        
        # Time since last transaction
        last_txn_data = self.r.get(f"{user_key}:last")
        if last_txn_data:
            last_txn = json.loads(last_txn_data)
            last_timestamp = datetime.fromisoformat(last_txn["timestamp"])
            time_since_last_txn = (timestamp - last_timestamp).total_seconds() / 60  # minutes
        else:
            time_since_last_txn = 99999.0
        
        # New merchant flag
        merchant_key = f"{user_key}:merchants"
        new_merchant_flag = 0 if self.r.sismember(merchant_key, txn["merchant_category"]) else 1
        
        # Geo jump flag
        geo_jump = 0
        if last_txn_data:
            last_txn = json.loads(last_txn_data)
            if last_txn["country_code"] != txn["country_code"]:
                geo_jump = 1
        else:
            geo_jump = 0
        
        # Store this transaction AFTER calculating features
        # (so it doesn't count itself in feature calculations)
        self._store_transaction(txn)
        
        return {
            "timestamp": txn["timestamp"],
            "amount": amount,
            "merchant_category": txn["merchant_category"],
            "payment_method": txn["payment_method"],
            "country_code": txn["country_code"],
            "txn_count_1h": txn_count_1h,
            "txn_count_24h": txn_count_24h,
            "avg_amount_7d": round(avg_amount_7d, 2),
            "amount_deviation": round(amount_deviation, 2),
            "time_since_last_txn": round(time_since_last_txn, 2),
            "is_night": is_night,
            "is_weekend": is_weekend,
            "new_merchant_flag": new_merchant_flag,
            "geo_jump": geo_jump,
            "high_amount_flag": high_amount_flag,
        }