from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# ==============================
# load artifacts
# ==============================
model = joblib.load("artifacts/plutus_xgb.pkl")
encoders = joblib.load("artifacts/encoders.pkl")
feature_list = joblib.load("artifacts/feature_list.pkl")
threshold = joblib.load("artifacts/threshold.pkl")

app = FastAPI(
    title="Plutus Fraud Detection API",
    description="Real-time credit card fraud detection system",
    version="1.0.0"
)

# ==============================
# input schema
# ==============================
class TransactionInput(BaseModel):
    timestamp: str
    amount: float
    merchant_category: str
    payment_method: str
    country_code: str
    txn_count_1h: int
    txn_count_24h: int
    avg_amount_7d: float
    amount_deviation: float
    time_since_last_txn: float
    is_night: int
    is_weekend: int
    new_merchant_flag: int
    geo_jump: int
    high_amount_flag: int

# ==============================
# health check
# ==============================
@app.get("/health")
def health():
    return {"status": "Plutus is alive"}

# ==============================
# prediction endpoint
# ==============================
@app.post("/predict")
def predict(txn: TransactionInput):
    try:
        data = txn.dict()

        ts =  datetime.fromisoformat(data["timestamp"])
        data["hour"] = ts.hour

        data.pop("timestamp")

        # encode categories
        for col, encoder in encoders.items():
            if data[col] not in encoder.classes_:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown category '{data[col]}' for feature '{col}'"
                )
            data[col] = encoder.transform([data[col]])[0]

        # create dataframe in correct feature order
        X = pd.DataFrame([[data[f] for f in feature_list]], columns=feature_list)

        # predict
        prob = model.predict_proba(X)[0][1]
        pred = int(prob >= threshold)

        decision = "ALLOW"
        if prob >= threshold:
            decision = "BLOCK"
        elif prob >= threshold * 0.7:
            decision = "REVIEW"
        
        return {
            "fraud_probability": round(float(prob), 4),
            "fraud_prediction": pred,
            "decision": decision
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))