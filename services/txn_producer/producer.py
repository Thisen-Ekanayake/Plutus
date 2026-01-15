import os
import time
import random
import numpy as np
from datetime import datetime, UTC
import redis
from urllib.parse import urlparse

# ======================
# Redis connection (supports Railway REDIS_URL or individual vars)
# ======================
def get_redis_client():
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
        # Fallback to individual environment variables
        return redis.Redis(
            host=os.environ.get("REDISHOST"),
            port=int(os.environ.get("REDISPORT")),
            password=os.environ.get("REDISPASSWORD"),
            decode_responses=True,
        )

r = get_redis_client()

STREAM = "transactions.stream"

# ======================
# Synthetic users
# ======================
COUNTRIES = ["US", "UK", "LK", "SG", "DE"]
MERCHANTS = [
    "groceries", "electronics", "travel",
    "fuel", "fashion", "restaurants", "health"
]

users = {
    f"user_{i}": {
        "avg_amount": np.random.lognormal(3, 0.5),
        "home_country": random.choice(COUNTRIES),
    }
    for i in range(5000)
}

# ======================
# Generate transaction
# ======================
def generate_txn():
    user_id = random.choice(list(users.keys()))
    profile = users[user_id]

    amount = np.random.lognormal(np.log(profile["avg_amount"]), 0.6)
    country = (
        random.choice(COUNTRIES)
        if np.random.rand() < 0.05
        else profile["home_country"]
    )

    return {
        "transaction_id": f"txn_{int(time.time() * 1000)}",
        "user_id": user_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "amount": round(amount, 2),
        "merchant_category": random.choice(MERCHANTS),
        "payment_method": random.choice(["card", "online", "wallet"]),
        "country_code": country,
    }

# ======================
# Daemon loop
# ======================
print("Redis producer started")

while True:
    try:
        txn = generate_txn()
        r.xadd(STREAM, txn)
        time.sleep(np.random.exponential(1.5))
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection error: {e}. Retrying in 5 seconds...")
        time.sleep(5)
    except Exception as e:
        print(f"Unexpected error: {e}. Continuing...")
        time.sleep(1)

    print("Added txn:", txn)

    length = r.xlen("transactions.stream")
    print("STREAM LENGTH:", length)

    last = r.xrevrange("transactions.stream", "+", "-", count=3)
    for entry_id, fields in last:
        print(entry_id, fields)