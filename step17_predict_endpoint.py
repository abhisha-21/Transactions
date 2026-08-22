"""
Step 3 - Single Transaction Prediction Endpoint
"""

from flask import Flask, request, jsonify
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import logging


print("=" * 70)
print("STEP 3: Flask Fraud Prediction API")
print("=" * 70)


# ============================================================
# INITIALIZE FLASK APP
# ============================================================

app = Flask(__name__)

print("\n[OK] Flask app initialized")


# ============================================================
# SETUP LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Models...")

try:

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    print("  [OK] model.pkl loaded")
    logger.info("Model loaded successfully")

except Exception as e:

    print(f"  [ERROR] Failed to load model: {e}")
    logger.error(f"Failed to load model: {e}")
    raise


# ============================================================
# LOAD SCALER
# ============================================================

try:

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    print("  [OK] scaler.pkl loaded")
    logger.info("Scaler loaded successfully")

except Exception as e:

    print(f"  [ERROR] Failed to load scaler: {e}")
    logger.error(f"Failed to load scaler: {e}")
    raise


# ============================================================
# LOAD FEATURE NAMES
# ============================================================

try:

    feature_names = np.load(
        "feature_names.npy",
        allow_pickle=True
    )

    feature_names = list(feature_names)

    print(
        f"  [OK] feature_names.npy loaded "
        f"({len(feature_names)} features)"
    )

    logger.info(
        f"Feature names loaded ({len(feature_names)} features)"
    )

except Exception as e:

    print(f"  [ERROR] Failed to load feature names: {e}")
    logger.error(f"Failed to load feature names: {e}")
    raise


# ============================================================
# MAPPINGS
# ============================================================

merchants = [
    "swiggy",
    "paytm",
    "flipkart",
    "amazon",
    "uber",
    "oyo",
    "byjus",
    "gpay",
    "airtel",
    "jio",
    "bill_payment",
    "utility",
    "petrol_pump",
    "atm",
    "pos_retail"
]

cities = [
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Hyderabad",
    "Pune",
    "Chennai",
    "Kolkata",
    "Jaipur",
    "Ahmedabad",
    "Surat"
]

networks = [
    "wifi",
    "4g",
    "5g"
]

merchant_to_code = {
    merchant: i
    for i, merchant in enumerate(merchants)
}

city_to_code = {
    city: i
    for i, city in enumerate(cities)
}

network_to_code = {
    network: i
    for i, network in enumerate(networks)
}


print("\nMappings loaded:")
print(f"  Merchants: {len(merchants)}")
print(f"  Cities: {len(cities)}")
print(f"  Networks: {len(networks)}")


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    try:

        logger.info("Health check requested")

        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": True,
            "scaler_loaded": True,
            "features": len(feature_names),
            "merchants": len(merchants),
            "cities": len(cities),
            "networks": len(networks)
        }), 200

    except Exception as e:

        logger.error(f"Health check failed: {e}")

        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


# ============================================================
# INFO ENDPOINT
# ============================================================

@app.route("/info", methods=["GET"])
def info():

    try:

        logger.info("Model info requested")

        return jsonify({
            "model_type": "XGBoost",
            "n_features": len(feature_names),
            "features": feature_names,
            "merchants": merchants,
            "cities": cities,
            "networks": networks,
            "description": (
                "AI Risk Manager - "
                "Credit Card Fraud Detection"
            )
        }), 200

    except Exception as e:

        logger.error(f"Info endpoint failed: {e}")

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# ============================================================
# HELPER FUNCTION: PREPROCESS TRANSACTION
# ============================================================

def preprocess_transaction(
    txn_dict,
    merchant_to_code,
    city_to_code,
    network_to_code,
    feature_names
):
    """
    Convert raw transaction data into model-ready features.
    """

    features = {}

    # ========================================================
    # PCA FEATURES V1-V28
    # ========================================================

    for i in range(1, 29):

        key = f"V{i}"

        value = txn_dict.get(key, 0.0)

        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0

        features[key] = value


    # ========================================================
    # AMOUNT
    # ========================================================

    amount_inr = txn_dict.get("Amount_INR", 1000)

    try:
        amount_inr = float(amount_inr)
    except (TypeError, ValueError):
        amount_inr = 1000.0

    # Prevent invalid negative amount
    amount_inr = max(amount_inr, 0)

    features["Amount_log"] = np.log1p(amount_inr)


    # ========================================================
    # TEMPORAL FEATURES
    # ========================================================

    features["hour_ist"] = txn_dict.get("hour_ist", 12)
    features["day_of_week"] = txn_dict.get("day_of_week", 0)
    features["is_weekend"] = txn_dict.get("is_weekend", 0)
    features["is_night"] = txn_dict.get("is_night", 0)
    features["is_payday"] = txn_dict.get("is_payday", 0)


    # ========================================================
    # VELOCITY FEATURES
    # ========================================================

    features["txn_count_1h"] = txn_dict.get(
        "txn_count_1h",
        2
    )

    features["txn_count_24h"] = txn_dict.get(
        "txn_count_24h",
        8
    )

    features["unique_merchants_24h"] = txn_dict.get(
        "unique_merchants_24h",
        3
    )


    # ========================================================
    # LOCATION
    # ========================================================

    city = txn_dict.get(
        "city",
        "Mumbai"
    )

    features["city_encoded"] = city_to_code.get(
        city,
        0
    )

    features["city_changes_24h"] = txn_dict.get(
        "city_changes_24h",
        0
    )


    # ========================================================
    # DEVICE
    # ========================================================

    features["device_risk_score"] = txn_dict.get(
        "device_risk_score",
        0.3
    )

    network = txn_dict.get(
        "network_type",
        "wifi"
    )

    features["network_encoded"] = network_to_code.get(
        network,
        0
    )


    # ========================================================
    # MERCHANT
    # ========================================================

    merchant = txn_dict.get(
        "merchant_category",
        "swiggy"
    )

    features["merchant_encoded"] = merchant_to_code.get(
        merchant,
        0
    )


    # ========================================================
    # CREATE DATAFRAME IN EXACT FEATURE ORDER
    # ========================================================

    df = pd.DataFrame([features])

    # Make sure every model feature exists
    for feature in feature_names:

        if feature not in df.columns:
            df[feature] = 0.0

    # Keep exactly the same order as training
    df = df[feature_names]

    return df


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ====================================================
        # GET JSON DATA
        # ====================================================

        data = request.get_json(silent=True)

        if not data:

            logger.warning(
                "Invalid request: no JSON body"
            )

            return jsonify({
                "status": "error",
                "error": "Request body must contain JSON"
            }), 400


        # ====================================================
        # CHECK TRANSACTIONS
        # ====================================================

        if "transactions" not in data:

            logger.warning(
                "Invalid request: transactions missing"
            )

            return jsonify({
                "status": "error",
                "error": (
                    'Please provide "transactions" '
                    "in request body"
                )
            }), 400


        transactions = data["transactions"]


        if not isinstance(transactions, list):

            return jsonify({
                "status": "error",
                "error": '"transactions" must be a list'
            }), 400


        if len(transactions) == 0:

            logger.warning(
                "Empty transactions list"
            )

            return jsonify({
                "status": "error",
                "error": (
                    "Transactions list "
                    "cannot be empty"
                )
            }), 400


        logger.info(
            f"Predicting for "
            f"{len(transactions)} transaction(s)"
        )


        # ====================================================
        # PREPROCESS
        # ====================================================

        processed_txns = []

        for txn in transactions:

            if not isinstance(txn, dict):

                return jsonify({
                    "status": "error",
                    "error": (
                        "Each transaction "
                        "must be a JSON object"
                    )
                }), 400

            df = preprocess_transaction(
                txn,
                merchant_to_code,
                city_to_code,
                network_to_code,
                feature_names
            )

            processed_txns.append(df)


        # Combine all transactions
        X = pd.concat(
            processed_txns,
            ignore_index=True
        )


        # ====================================================
        # SCALE
        # ====================================================

        X_scaled = scaler.transform(X)


        # ====================================================
        # PREDICT
        # ====================================================

        predictions = model.predict(X_scaled)

        probabilities = model.predict_proba(
            X_scaled
        )


        # ====================================================
        # FORMAT RESULTS
        # ====================================================

        results = []

        for i, txn in enumerate(transactions):

            prediction = int(predictions[i])

            fraud_probability = float(
                probabilities[i, 1]
            )

            legitimacy_probability = float(
                probabilities[i, 0]
            )

            confidence = float(
                max(probabilities[i])
            )

            result = {
                "transaction_id": txn.get(
                    "id",
                    f"txn_{i}"
                ),

                "prediction": prediction,

                "fraud_probability":
                    fraud_probability,

                "legitimacy_probability":
                    legitimacy_probability,

                "recommendation":
                    "DECLINE"
                    if prediction == 1
                    else "APPROVE",

                "confidence":
                    confidence
            }

            results.append(result)


        logger.info(
            f"Prediction successful for "
            f"{len(transactions)} transaction(s)"
        )


        # ====================================================
        # RETURN RESPONSE
        # ====================================================

        return jsonify({

            "status": "success",

            "timestamp":
                datetime.now().isoformat(),

            "total_transactions":
                len(transactions),

            "results":
                results

        }), 200


    except Exception as e:

        logger.exception(
            "Prediction error"
        )

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message":
            "AI Risk Manager Fraud Detection API",

        "status":
            "running",

        "endpoints": [
            "/",
            "/health",
            "/info",
            "/predict"
        ]
    }), 200


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("FLASK FRAUD DETECTION API READY")
    print("=" * 70)

    print("\nAvailable endpoints:")
    print("  GET  http://localhost:5000/")
    print("  GET  http://localhost:5000/health")
    print("  GET  http://localhost:5000/info")
    print("  POST http://localhost:5000/predict")

    print("\nStarting server...\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )