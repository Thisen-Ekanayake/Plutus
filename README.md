# Plutus - Real-Time Credit Card Fraud Detection System

Plutus is a machine learning-powered fraud detection system that provides real-time credit card transaction analysis with explainable AI insights. The system uses XGBoost for fraud prediction and SHAP (SHapley Additive exPlanations) for model interpretability.

## 🎯 Features

- **Real-time Fraud Detection**: Fast API-based prediction endpoint for instant transaction analysis
- **Explainable AI**: SHAP-based feature importance to understand why transactions are flagged
- **Interactive Dashboard**: Streamlit-based web interface for easy interaction and visualization
- **Decision Logic**: Three-tier decision system (ALLOW, REVIEW, BLOCK) based on fraud probability
- **Synthetic Data Generation**: Built-in data generator for testing and development
- **Production-Ready API**: FastAPI-based REST API with proper error handling

## 📁 Project Structure

```
Plutus/
├── api/
│   ├── app.py                 # FastAPI application
│   └── artifacts/             # Trained model artifacts
│       ├── plutus_xgb.pkl     # XGBoost model
│       ├── encoders.pkl       # Label encoders for categorical features
│       ├── feature_list.pkl   # Feature order list
│       └── threshold.pkl      # Optimal decision threshold
├── dashboard.py               # Streamlit dashboard
├── data_preprocess.py         # Data preprocessing pipeline
├── model_training.ipynb       # Model training notebook
├── synthetic_data_generator.py # Synthetic transaction data generator
├── plutus_transactions.csv    # Transaction dataset
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd Plutus
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Data Generation

If you need to generate synthetic transaction data:

```bash
python synthetic_data_generator.py
```

This will create `plutus_transactions.csv` with:
- 150,000 transactions from 5,000 users
- ~1.5% fraud rate
- Realistic features including temporal patterns, merchant categories, and behavioral flags

## 🔧 Model Training

### Step 1: Preprocess Data

```bash
python data_preprocess.py
```

This script:
- Loads the transaction dataset
- Encodes categorical features (merchant_category, payment_method, country_code)
- Splits data into train/validation sets (80/20, stratified)
- Saves encoders and feature list to artifacts

### Step 2: Train Model

Open and run `model_training.ipynb`:

1. **Train XGBoost Classifier:**
   - Uses class weights to handle imbalanced data
   - 500 estimators, max depth 6
   - Early stopping based on validation set

2. **Find Optimal Threshold:**
   - Uses Precision-Recall curve
   - Selects threshold with recall ≥ 0.8
   - Saves threshold for production use

3. **Evaluate Model:**
   - Generates classification report
   - Tests multiple threshold values

**Important:** After training, move the artifacts to the API directory:
```bash
mv plutus_xgb.pkl api/artifacts/
mv encoders.pkl api/artifacts/
mv feature_list.pkl api/artifacts/
mv threshold.pkl api/artifacts/
```

## 🌐 API Server

### Start the API

From the project root directory:

```bash
cd api
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Or from the root:
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- **Interactive API Docs:** http://localhost:8000/docs
- **Alternative Docs:** http://localhost:8000/redoc

### Endpoints

#### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "Plutus is alive"
}
```

#### Fraud Prediction
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "timestamp": "2025-01-15T14:30:00",
  "amount": 450.50,
  "merchant_category": "electronics",
  "payment_method": "card",
  "country_code": "US",
  "txn_count_1h": 2,
  "txn_count_24h": 15,
  "avg_amount_7d": 120.0,
  "amount_deviation": 330.5,
  "time_since_last_txn": 45.0,
  "is_night": 0,
  "is_weekend": 0,
  "new_merchant_flag": 1,
  "geo_jump": 0,
  "high_amount_flag": 1
}
```

**Response:**
```json
{
  "fraud_probability": 0.6234,
  "fraud_prediction": 1,
  "decision": "BLOCK",
  "top_risk_factors": [
    {
      "feature": "high_amount_flag",
      "impact": 0.2341,
      "effect": "increase fraud risk"
    },
    {
      "feature": "amount_deviation",
      "impact": 0.1892,
      "effect": "increase fraud risk"
    },
    ...
  ]
}
```

**Decision Logic:**
- `BLOCK`: fraud_probability ≥ threshold
- `REVIEW`: fraud_probability ≥ threshold × 0.7
- `ALLOW`: fraud_probability < threshold × 0.7

## 📈 Dashboard

### Start the Dashboard

In a new terminal:

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Features

1. **Transaction Input Form** (Sidebar):
   - Date and time picker
   - Transaction amount
   - Merchant category (10 options)
   - Payment method (card, online, wallet)
   - Country code (US, UK, LK, SG, DE)
   - Transaction history features
   - Behavioral flags (night, weekend, new merchant, geo jump, high amount)

2. **Prediction Results**:
   - Fraud probability percentage
   - Color-coded decision (ALLOW/REVIEW/BLOCK)
   - Top 5 risk factors with SHAP values
   - Interactive bar chart showing feature impacts

3. **Demo Data**:
   - Sample recent transactions table

## 🔍 Model Details

### Features

The model uses the following features:

**Categorical:**
- `merchant_category`: groceries, electronics, travel, fuel, fashion, restaurants, health, subscriptions, gaming, luxury
- `payment_method`: card, online, wallet
- `country_code`: US, UK, LK, SG, DE

**Numerical:**
- `amount`: Transaction amount
- `hour`: Hour of day (extracted from timestamp)
- `txn_count_1h`: Number of transactions in last hour
- `txn_count_24h`: Number of transactions in last 24 hours
- `avg_amount_7d`: Average transaction amount in last 7 days
- `amount_deviation`: Absolute deviation from 7-day average
- `time_since_last_txn`: Minutes since last transaction

**Binary Flags:**
- `is_night`: 1 if transaction between 22:00-06:00
- `is_weekend`: 1 if transaction on weekend
- `new_merchant_flag`: 1 if first time with this merchant
- `geo_jump`: 1 if country changed from previous transaction
- `high_amount_flag`: 1 if amount > 3× average

### Model Architecture

- **Algorithm:** XGBoost Classifier
- **Objective:** Binary logistic regression
- **Evaluation Metric:** AUC-PR (Area Under Precision-Recall Curve)
- **Class Weighting:** Automatic based on class imbalance
- **Hyperparameters:**
  - n_estimators: 500
  - max_depth: 6
  - learning_rate: 0.05
  - subsample: 0.8
  - colsample_bytree: 0.8

### Explainability

The system uses SHAP TreeExplainer to provide:
- Feature importance scores
- Direction of impact (increases/reduces fraud risk)
- Top 5 risk drivers per prediction

## 🧪 Testing

### Test API with curl

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-01-15T14:30:00",
    "amount": 450.50,
    "merchant_category": "electronics",
    "payment_method": "card",
    "country_code": "US",
    "txn_count_1h": 2,
    "txn_count_24h": 15,
    "avg_amount_7d": 120.0,
    "amount_deviation": 330.5,
    "time_since_last_txn": 45.0,
    "is_night": 0,
    "is_weekend": 0,
    "new_merchant_flag": 1,
    "geo_jump": 0,
    "high_amount_flag": 1
  }'
```

### Test Health Endpoint

```bash
curl http://localhost:8000/health
```

## 📝 Notes

- The API expects categorical values that were seen during training. Unknown categories will return a 400 error.
- The model threshold is optimized for recall ≥ 0.8 to minimize false negatives (missed fraud).
- SHAP values are computed for the positive class (fraud) in binary classification.
- All timestamps should be in ISO format: `YYYY-MM-DDTHH:MM:SS`

## 🛠️ Troubleshooting

### API can't find artifacts
- Ensure you're running the API from the `api/` directory, or
- Check that all `.pkl` files are in `api/artifacts/`

### Dashboard can't connect to API
- Verify the API is running on `http://127.0.0.1:8000`
- Check the `API_URL` in `dashboard.py` matches your API address

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Activate your virtual environment if using one

## 📄 License

This project is provided as-is for educational and demonstration purposes.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

---