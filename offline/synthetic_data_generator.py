import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from tqdm import tqdm

# ==============================
# configuration
# ==============================

N_USERS = 5000
N_TXNS = 150_000
FRAUD_RATE_TARGET = 0.015

START_DATE = datetime(2025, 1, 1)

MERCHANT_CATEGORIES = [
    "groceries", "electronics", "travel", "fuel", "fashion",
    "restaurants", "health", "subscriptions", "gaming", "luxury"
]

PAYMENT_METHODS = ["card", "online", "wallet"]
COUNTRIES = ["US", "UK", "LK", "SG", "DE"]

np.random.seed(42)

# ==============================
# user profiles
# ==============================
users = {
    f"user_{i}": {
        "avg_amount": np.random.lognormal(mean=3, sigma=0.5),
        "home_country": random.choice(COUNTRIES),
        "txn_rate": np.random.uniform(0.5, 3.0)     # txn per day
    }
    for i in range(N_USERS)
}

# ==============================
# generate trasactions
# ==============================
transactions = []
current_time = START_DATE

for i in tqdm(range(N_TXNS), desc="Generating transactions"):
    user_id = random.choice(list(users.keys()))
    profile = users[user_id]

    # advance time
    current_time += timedelta(minutes=np.random.exponential(10))

    amount = np.random.lognormal(mean=np.log(profile["avg_amount"]), sigma=0.6)
    merchant = random.choice(MERCHANT_CATEGORIES)
    payment_method = random.choice(PAYMENT_METHODS)

    country = (
        random.choice(COUNTRIES)
        if np.random.rand() < 0.05
        else profile["home_country"]
    )

    transactions.append({
        "transaction_id": f"txn_{i}",
        "user_id": user_id,
        "timestamp": current_time,
        "amount": round(amount, 2),
        "merchant_category": merchant,
        "payment_method": payment_method,
        "country_code": country
    })

df = pd.DataFrame(transactions)

# ==============================
# sort for real-time simulation
# ==============================
df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

# ==============================
# feature engineering
# ==============================
df["hour"] = df["timestamp"].dt.hour
df["is_night"] = df["hour"].apply(lambda x: 1 if x >= 22 or x <= 6 else 0)
df["is_weekend"] = df["timestamp"].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)


# rolling features
df["txn_count_1h"] = 0
df["txn_count_24h"] = 0
df["avg_amount_7d"] = 0.0
df["time_since_last_txn"] = 0.0
df["new_merchant_flag"] = 0

user_groups = df.groupby("user_id")

for user_id, group in tqdm(user_groups, desc="Computing user features", total=len(user_groups)):
    times = group["timestamp"]
    amounts = group["amount"]
    merchants = group["merchant_category"]

    last_time = None
    seen_merchants = set()

    for idx in group.index:
        t = df.loc[idx, "timestamp"]

        df.loc[idx, "txn_count_1h"] = ((times >= t - timedelta(hours=1)) & (times < t)).sum()
        df.loc[idx, "txn_count_24h"] = ((times >= t - timedelta(hours=24)) & (times < t)).sum()

        last_7d = (times >= t - timedelta(days=7)) & (times < t)
        df.loc[idx, "avg_amount_7d"] = amounts[last_7d].mean() if last_7d.any() else amounts.mean()


        if last_time is None:
            df.loc[idx, "time_since_last_txn"] = 99999
        else:
            df.loc[idx, "time_since_last_txn"] = (t - last_time).seconds
        
        merchant = df.loc[idx, "merchant_category"]
        df.loc[idx, "new_merchant_flag"] = 0 if merchant in seen_merchants else 1
        seen_merchants.add(merchant)

        last_time = t

df["amount_deviation"] = abs(df["amount"] - df["avg_amount_7d"])
df["high_amount_flag"] = (df["amount"] > df["avg_amount_7d"] * 3).astype(int)
df["geo_jump"] = (df["country_code"] != df.groupby("user_id")["country_code"].shift(1)).astype(int)

# ==============================
# fraud injection (conditional)
# ==============================
fraud = []

for _, row in tqdm(df.iterrows(), desc="Injecting fraud", total=len(df)):
    score = 0

    if row["high_amount_flag"] == 1:
        score += 0.4
    if row["txn_count_1h"] > 1:
        score += 0.3
    if row["is_night"] == 1:
        score += 0.2
    if row["geo_jump"] == 1:
        score += 0.3
    if row["new_merchant_flag"] == 1:
        score += 0.2
    
    fraud.append(1 if np.random.rand() < score else 0)

df["fraud_label"] = fraud

# control fraud rate
current_rate = df["fraud_label"].mean()
scaling_factor = FRAUD_RATE_TARGET / current_rate
df.loc[df["fraud_label"] == 1, "fraud_label"] = (
    np.random.rand(df["fraud_label"].sum()) < scaling_factor
).astype(int)

print("Final fraud rate:", df["fraud_label"].mean())

# ==============================
# save dataset
# ==============================
df.to_csv("plutus_transactions.csv", index=False)