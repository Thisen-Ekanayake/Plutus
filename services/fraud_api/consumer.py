import redis
import os
import json
from datetime import datetime
from urllib.parse import urlparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from feature_engine import FeatureEngine

# ==============================
# Redis connection
# ==============================
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

r = get_redis_client()

# ==============================
# Load model artifacts
# ==============================
BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

try:
    model = joblib.load(ARTIFACTS_DIR / "plutus_xgb.pkl")
    encoders = joblib.load(ARTIFACTS_DIR / "encoders.pkl")
    feature_list = joblib.load(ARTIFACTS_DIR / "feature_list.pkl")
    threshold = joblib.load(ARTIFACTS_DIR / "threshold.pkl")
    print("Model artifacts loaded successfully")
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    raise

# ==============================
# Initialize feature engine
# ==============================
feature_engine = FeatureEngine(r)

# ==============================
# Stream configuration
# ==============================
STREAM = "transactions.stream"
GROUP = "fraud_group"
CONSUMER = f"fraud_instance_{os.environ.get('HOSTNAME', 'default')}"

# ==============================
# Initialize consumer group
# ==============================
try:
    r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    print(f"Created consumer group '{GROUP}'")
except redis.exceptions.ResponseError as e:
    if "BUSYGROUP" in str(e):
        print(f"Consumer group '{GROUP}' already exists")
    else:
        raise

print("Fraud consumer started")
print(f"Stream: {STREAM}")
print(f"Group: {GROUP}")
print(f"Consumer: {CONSUMER}")

# ==============================
# Main consumption loop
# ==============================
while True:
    try:
        messages = r.xreadgroup(
            groupname=GROUP,
            consumername=CONSUMER,
            streams={STREAM: ">"},
            count=10,
            block=5000,
        )

        for _, entries in messages:
            for msg_id, txn_data in entries:
                try:
                    # Parse transaction data
                    txn = dict(txn_data)
                    
                    # Enrich transaction with features
                    enriched = feature_engine.enrich(txn)
                    
                    # Prepare data for model
                    ts = datetime.fromisoformat(enriched["timestamp"])
                    enriched["hour"] = ts.hour
                    
                    # Encode categorical features
                    model_input = enriched.copy()
                    model_input.pop("timestamp")
                    
                    for col, encoder in encoders.items():
                        if col in model_input:
                            if model_input[col] not in encoder.classes_:
                                print(f"Unknown category '{model_input[col]}' for '{col}', skipping")
                                r.xack(STREAM, GROUP, msg_id)
                                continue
                            model_input[col] = encoder.transform([model_input[col]])[0]
                    
                    # Create DataFrame in correct feature order
                    X = pd.DataFrame([[model_input[f] for f in feature_list]], columns=feature_list)
                    
                    # Predict fraud probability
                    prob = model.predict_proba(X)[0][1]
                    pred = int(prob >= threshold)
                    
                    # Calculate latency
                    latency = (
                        datetime.utcnow() - datetime.fromisoformat(txn["timestamp"])
                    ).total_seconds()
                    
                    # Decision logic
                    decision = "ALLOW"
                    if prob >= threshold:
                        decision = "BLOCK"
                    elif prob >= threshold * 0.7:
                        decision = "REVIEW"
                    
                    # Log result
                    print(
                        f"TXN {txn.get('transaction_id', 'unknown')} | "
                        f"User: {txn.get('user_id', 'unknown')} | "
                        f"Amount: ${enriched['amount']:.2f} | "
                        f"Fraud Prob: {prob:.4f} | "
                        f"Decision: {decision} | "
                        f"Latency: {latency:.3f}s"
                    )
                    
                    # Acknowledge message
                    r.xack(STREAM, GROUP, msg_id)
                    
                except Exception as e:
                    print(f"Error processing transaction: {e}")
                    import traceback
                    traceback.print_exc()
                    # Still acknowledge to avoid reprocessing bad messages
                    r.xack(STREAM, GROUP, msg_id)
                    
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection error: {e}")
        print("Retrying in 5 seconds...")
        import time
        time.sleep(5)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        import time
        time.sleep(5)
