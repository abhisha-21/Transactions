from flask import request, jsonify
from datetime import datetime
import numpy as np
import pandas as pd
import logging

# Import everything from Step 3
from step17_predict_endpoint import (
    app,
    model,
    scaler,
    feature_names,
    merchant_to_code,
    city_to_code,
    network_to_code,
    preprocess_transaction
)

logger = logging.getLogger(__name__)


@app.route("/batch-predict", methods=["POST"])
def batch_predict():

    try:

        # Get JSON
        data = request.get_json(silent=True)

        if not data:

            return jsonify({
                "status": "error",
                "error": "Request body must contain JSON"
            }), 400


        # Get transactions
        transactions = data.get("transactions")

        if transactions is None:

            return jsonify({
                "status": "error",
                "error": (
                    'Please provide "transactions" '
                    "in request body"
                )
            }), 400


        if not isinstance(transactions, list):

            return jsonify({
                "status": "error",
                "error": '"transactions" must be a list'
            }), 400


        if len(transactions) == 0:

            return jsonify({
                "status": "error",
                "error": "Transactions list cannot be empty"
            }), 400


        logger.info(
            f"Batch predicting for "
            f"{len(transactions)} transactions"
        )


        processed_txns = []

        for i, txn in enumerate(transactions):

            if not isinstance(txn, dict):

                return jsonify({
                    "status": "error",
                    "error": (
                        f"Transaction at index {i} "
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


        # Combine transactions
        X = pd.concat(
            processed_txns,
            ignore_index=True
        )

        X_scaled = scaler.transform(X)

        predictions = model.predict(X_scaled)

        probabilities = model.predict_proba(
            X_scaled
        )

        total_transactions = len(transactions)

        fraud_count = int(
            np.sum(predictions == 1)
        )

        legitimate_count = int(
            np.sum(predictions == 0)
        )

        fraud_rate = (
            fraud_count / total_transactions
        )


        fraud_probabilities = probabilities[:, 1]


        mean_fraud_probability = float(
            np.mean(fraud_probabilities)
        )

        min_fraud_probability = float(
            np.min(fraud_probabilities)
        )

        max_fraud_probability = float(
            np.max(fraud_probabilities)
        )

        std_fraud_probability = float(
            np.std(fraud_probabilities)
        )


        # Statistics for transactions classified as fraud
        fraud_mask = predictions == 1

        if fraud_count > 0:

            avg_fraud_prob = float(
                np.mean(
                    probabilities[fraud_mask, 1]
                )
            )

            max_detected_fraud_prob = float(
                np.max(
                    probabilities[fraud_mask, 1]
                )
            )

        else:

            avg_fraud_prob = 0.0
            max_detected_fraud_prob = 0.0

        return jsonify({

            "status": "success",

            "timestamp":
                datetime.now().isoformat(),

            "total_transactions":
                total_transactions,

            "fraud_detected":
                fraud_count,

            "legitimate_transactions":
                legitimate_count,

            "fraud_rate":
                float(fraud_rate),

            "average_fraud_probability":
                avg_fraud_prob,

            "max_fraud_probability_detected":
                max_detected_fraud_prob,

            "statistics": {

                "mean_fraud_probability":
                    mean_fraud_probability,

                "min_fraud_probability":
                    min_fraud_probability,

                "max_fraud_probability":
                    max_fraud_probability,

                "std_fraud_probability":
                    std_fraud_probability
            }

        }), 200


    except Exception as e:

        logger.exception(
            "Batch prediction error"
        )

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == "__main__":

    print("=" * 70)
    print("STEP 18: BATCH PREDICTION API")
    print("=" * 70)

    print("\nEndpoints:")
    print("  GET  http://localhost:5000/health")
    print("  GET  http://localhost:5000/info")
    print("  POST http://localhost:5000/predict")
    print("  POST http://localhost:5000/batch-predict")

    print("\nStarting server...\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
