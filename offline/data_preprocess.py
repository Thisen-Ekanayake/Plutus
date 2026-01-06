import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# load dataset
df = pd.read_csv("plutus_transactions.csv")

# target
y = df["fraud_label"]

# drop non-features
drop_cols = ["transaction_id", "user_id", "timestamp", "fraud_label"]
X = df.drop(columns=drop_cols)

# categorixcal columns
cat_cols = ["merchant_category", "payment_method", "country_code"]

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# train / validation split (stratified because imbalance)
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# save encoders and feature order
joblib.dump(encoders, "encoders.pkl")
joblib.dump(list(X.columns), "feature_list.pkl")