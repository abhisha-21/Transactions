from flask import Flask, jsonify
from datetime import datetime
import numpy as np
import pickle
import logging


print("=" * 70)
print("STEP 2: Flask Health Check & Info Endpoints")
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

    print("  [OK] model.pkl loaded")
    logger.info("Model loaded successfully")

except Exception as e:

    print(f"  [ERROR] Failed to load model: {e}")
    logger.error(f"Failed to load model: {e}")
    raise

try:

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    print("  [OK] scaler.pkl loaded")
    logger.info("Scaler loaded successfully")

except Exception as e:

    print(f"  [ERROR] Failed to load scaler: {e}")
    logger.error(f"Failed to load scaler: {e}")
    raise


try:

    feature_names = np.load(
        "feature_names.npy",
        allow_pickle=True
    )

    # Convert NumPy array to normal Python list
    feature_names = list(feature_names)

    print(
        f" feature_names.npy loaded "
        f"({len(feature_names)} features)"
    )

    logger.info(
        f"Feature names loaded ({len(feature_names)} features)"
    )

except Exception as e:

    print(f" Failed to load feature names: {e}")
    logger.error(f"Failed to load feature names: {e}")
    raise


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



@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.

    Returns:
        JSON containing API and model status.
    """

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


@app.route("/info", methods=["GET"])
def info():
    """
    Return information about the fraud detection model.
    """

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


@app.route("/", methods=["GET"])
def home():
    """
    Basic API welcome endpoint.
    """

    return jsonify({
        "message": "AI Risk Manager Fraud Detection API",
        "status": "running",
        "endpoints": [
            "/health",
            "/info"
        ]
    }), 200

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("FLASK API READY")
    print("=" * 70)

    print("\nAvailable endpoints:")
    print("  GET http://localhost:5000/")
    print("  GET http://localhost:5000/health")
    print("  GET http://localhost:5000/info")

    print("\nStarting server...\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
