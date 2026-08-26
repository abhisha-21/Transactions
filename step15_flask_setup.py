"""
Flask API Setup - Load Models & Initialize App
"""

from flask import Flask, jsonify
import numpy as np
import pandas as pd
import pickle
import logging

print("=" * 70)
print("STEP 1: Flask Setup & Load Models")
print("=" * 70)

app = Flask(__name__)

print("\n Flask app initialized")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

logger.info("Starting Flask API Server...")


print("\nLoading Models...")

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    logger.info("Model loaded successfully")
    print("  [OK] model.pkl loaded")

except Exception as e:
    logger.error(f"Failed to load model: {e}")
    print(f"  [ERROR] Error loading model: {e}")
    raise

try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    logger.info("Scaler loaded successfully")
    print("  [OK] scaler.pkl loaded")

except Exception as e:
    logger.error(f"Failed to load scaler: {e}")
    print(f"  [ERROR] Error loading scaler: {e}")
    raise

try:
    feature_names = np.load(
        "feature_names.npy",
        allow_pickle=True
    )

    feature_names = list(feature_names)

    logger.info(
        f"Feature names loaded ({len(feature_names)} features)"
    )

    print(
        f"  [OK] feature_names.npy loaded "
        f"({len(feature_names)} features)"
    )

except Exception as e:
    logger.error(f"Failed to load feature names: {e}")
    print(f"  [ERROR] Error loading features: {e}")
    raise

print("\n All models loaded successfully!")


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

merchant_to_code = {
    merchant: i
    for i, merchant in enumerate(merchants)
}

city_to_code = {
    city: i
    for i, city in enumerate(cities)
}

network_to_code = {
    "wifi": 0,
    "4g": 1,
    "5g": 2
}

print("\nMappings loaded:")
print(f"  Merchants: {len(merchants)}")
print(f"  Cities: {len(cities)}")
print(f"  Networks: {len(network_to_code)}")


app.config["MODEL"] = model
app.config["SCALER"] = scaler
app.config["FEATURE_NAMES"] = feature_names
app.config["MERCHANTS"] = merchants
app.config["CITIES"] = cities
app.config["MERCHANT_TO_CODE"] = merchant_to_code
app.config["CITY_TO_CODE"] = city_to_code
app.config["NETWORK_TO_CODE"] = network_to_code

print("\n Configuration complete!")
print("  Ready for endpoints")


@app.route("/health", methods=["GET"])
def health():
    """
    Check whether the Flask API is running.
    """
    return jsonify({
        "status": "healthy",
        "message": "Fraud detection API is running",
        "model_loaded": True,
        "features": len(feature_names)
    }), 200

@app.route("/info", methods=["GET"])
def info():
    """
    Return basic information about the API and loaded model.
    """
    return jsonify({
        "api": "Fraud Detection API",
        "status": "running",
        "model_loaded": True,
        "feature_count": len(feature_names),
        "merchant_count": len(merchants),
        "city_count": len(cities),
        "network_count": len(network_to_code)
    }), 200



print("\nTest Prediction (using dummy data):")

try:
    # Create dummy transaction
    dummy_X = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )

    # Scale the dummy transaction
    dummy_X_scaled = scaler.transform(dummy_X)

    # Prediction
    dummy_pred = model.predict(dummy_X_scaled)

    # Probability
    dummy_proba = model.predict_proba(dummy_X_scaled)

    print(
        f"  Prediction: {dummy_pred[0]} "
        f"(0=Legit, 1=Fraud)"
    )

    print(
        f"  Probability: {dummy_proba[0]}"
    )

    print("  [OK] Model works!")

except Exception as e:
    print(f"  [ERROR] Test prediction failed: {e}")
    raise

print("\n" + "=" * 70)
print("SETUP COMPLETE - ENDPOINTS READY")


if __name__ == "__main__":

    print("\nStarting Flask server...")
    print("  Health: http://localhost:5000/health")
    print("  Info:   http://localhost:5000/info")
    print("\nPress CTRL+C to stop the server.\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
