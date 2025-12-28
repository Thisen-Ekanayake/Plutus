import streamlit as st
import requests
import pandas as pd
import altair as alt
from datetime import datetime

# ==============================
# config
# ==============================
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Plutus Dashboard", layout="wide")
st.title("Plutus Fraud Detection Dashboard")
st.markdown("Real-time credit card fraud prediction with explainability")

# ==============================
# transaction input form
# ==============================
st.sidebar.header("Enter Transaction Details")

def get_transaction_input():
    timestamp = st.sidebar.date_input("Transaction Date", datetime.now().date())
    time = st.sidebar.time_input("Transaction Time", datetime.now().time())
    timestamp_str = datetime.combine(timestamp, time).isoformat()

    data = {
        "timestamp": timestamp_str,
        "amount": st.sidebar.number_input("Amount", min_value=0.0, value=100.0, step=0.5),
        "merchant_category": st.sidebar.selectbox("Merchant Category",
                                                  [
                                                    "groceries", "electronics", "travel", "fuel", "fashion",
                                                    "restaurants", "health", "subscriptions", "gaming", "luxury"
                                                ]),
        "payment_method": st.sidebar.selectbox("Payment Method", ["card", "online", "wallet"]),
        "country_code": st.sidebar.selectbox("Country Code", ["US", "UK", "LK", "SG", "DE"]),
        "txn_count_1h": st.sidebar.number_input("Txns last 1h", 0, 100, 0),
        "txn_count_24h": st.sidebar.number_input("Txns last 24h", 0, 1000, 0),
        "avg_amount_7d": st.sidebar.number_input("Avg amount last 7d", 0.0, 10000.0, 100.0),
        "amount_deviation": st.sidebar.number_input("Amount deviation", 0.0, 10000.0, 0.0),
        "time_since_last_txn": st.sidebar.number_input("Time since last txn (mins)", 0, 10000, 60),
        "is_night": int(st.sidebar.checkbox("Is night?", False)),
        "is_weekend": int(st.sidebar.checkbox("Is weekend?", False)),
        "new_merchant_flag": int(st.sidebar.checkbox("New Merchant?", False)),
        "geo_jump": int(st.sidebar.checkbox("Geo jump?", False)),
        "high_amount_flag": int(st.sidebar.checkbox("High amount?", False)),
    }
    return data

transaction_data = get_transaction_input()

# ==============================
# predict button
# ==============================
if st.sidebar.button("Predict Fraud"):
    try:
        response = requests.post(API_URL, json=transaction_data)
        result = response.json()

        # ===================================
        # show fraud probability & decision
        # ===================================
        st.subheader("Prediction Result")
        prob = result["fraud_probability"]
        decision = result["decision"]

        color = "green" if decision=="ALLOW" else "yellow" if decision=="REVIEW" else "red"
        st.markdown(f"**Fraud Probability:** {prob*100:.2f}%")
        st.markdown(f"**Decision:** <span style='color:{color};font-weight:bold'>{decision}</span>", unsafe_allow_html=True)

        # ==============================
        # show top shap factors
        # ==============================
        st.subheader("Top Risk Factors")
        factors = result["top_risk_factors"]
        df_factors = pd.DataFrame(factors)
        df_factors["impact_abs"] = df_factors["impact"].abs()

        # horizontal bar chart
        chart = alt.Chart(df_factors).mark_bar().encode(
            x=alt.X("impact_abs:Q", title="Impact (absolute)"),
            y=alt.Y("feature:N", sort='-x'),
            color=alt.Color("impact:Q", scale=alt.Scale(scheme="redblue"), title="Impact Sign"),
            tooltip=["feature", "impact", "effect"]
        ).properties(height=250)

        st.altair_chart(chart, use_container_width=True)

        st.dataframe(df_factors[["feature", "impact", "effect"]])

    except Exception as e:
        st.error(f"Error calling API: {e}")

# ==============================
# simulated past transactions
# ==============================
st.subheader("Recent Transactions (Demo)")
demo_data = pd.DataFrame({
    "timestamp": pd.date_range(end=datetime.now(), periods=5, freq='H'),
    "amount": [120, 450, 300, 50, 700],
    "fraud_probability": [0.05, 0.53, 0.12, 0.02, 0.77],
    "decision": ["ALLOW", "BLOCK", "ALLOW", "ALLOW", "BLOCK"]
})
st.dataframe(demo_data)