"""
Complete Flask API for Fraud Detection
- Health check
- Model info
- Single prediction
- Batch prediction
- Error handling
- Logging
"""

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime
import traceback

# ============================================================
# INITIALIZE
# ============================================================

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

logger.info("="*70)
logger.info("Starting Fraud Detection API")
logger.info("="*70)

# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

feature_names = np.load('feature_names.npy', allow_pickle=True)

# Mappings
merchants = ['swiggy', 'paytm', 'flipkart', 'amazon', 'uber',
             'oyo', 'byjus', 'gpay', 'airtel', 'jio',
             'bill_payment', 'utility', 'petrol_pump', 'atm', 'pos_retail']

cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Pune',
          'Chennai', 'Kolkata', 'Jaipur', 'Ahmedabad', 'Surat']

merchant_to_code = {m: i for i, m in enumerate(merchants)}
city_to_code = {c: i for i, c in enumerate(cities)}
network_to_code = {'wifi': 0, '4g': 1, '5g': 2}

print("✓ Models loaded")

# ============================================================
# HELPER FUNCTION
# ============================================================

def preprocess_transaction(txn_dict):
    """Preprocess transaction to model format"""
    features = {}
    
    # PCA features
    for i in range(1, 29):
        features[f'V{i}'] = txn_dict.get(f'V{i}', 0.0)
    
    # Amount
    features['Amount_log'] = np.log1p(txn_dict.get('Amount_INR', 1000))
    
    # Temporal
    features['hour_ist'] = txn_dict.get('hour_ist', 12)
    features['day_of_week'] = txn_dict.get('day_of_week', 0)
    features['is_weekend'] = txn_dict.get('is_weekend', 0)
    features['is_night'] = txn_dict.get('is_night', 0)
    features['is_payday'] = txn_dict.get('is_payday', 0)
    
    # Velocity
    features['txn_count_1h'] = txn_dict.get('txn_count_1h', 2)
    features['txn_count_24h'] = txn_dict.get('txn_count_24h', 8)
    features['unique_merchants_24h'] = txn_dict.get('unique_merchants_24h', 3)
    
    # Location
    features['city_encoded'] = city_to_code.get(txn_dict.get('city', 'Mumbai'), 0)
    features['city_changes_24h'] = txn_dict.get('city_changes_24h', 0)
    
    # Device
    features['device_risk_score'] = txn_dict.get('device_risk_score', 0.3)
    features['network_encoded'] = network_to_code.get(txn_dict.get('network_type', 'wifi'), 0)
    
    # Merchant
    features['merchant_encoded'] = merchant_to_code.get(txn_dict.get('merchant_category', 'swiggy'), 0)
    
    df = pd.DataFrame([features])
    df = df[feature_names.tolist()]
    return df.values[0]

# ============================================================
# ENDPOINTS
# ============================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model': 'XGBoost',
            'features': len(feature_names)
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    """Model info"""
    return jsonify({
        'model_type': 'XGBoost',
        'n_features': len(feature_names),
        'merchants': merchants,
        'cities': cities,
        'networks': list(network_to_code.keys())
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Single/batch prediction"""
    try:
        data = request.get_json()
        
        if not data or 'transactions' not in data:
            return jsonify({'error': 'Missing transactions'}), 400
        
        transactions = data['transactions']
        if not transactions:
            return jsonify({'error': 'Empty transactions list'}), 400
        
        logger.info(f"Predicting {len(transactions)} transactions")
        
        # Preprocess
        X = np.array([preprocess_transaction(txn) for txn in transactions])
        X_scaled = scaler.transform(X)
        
        # Predict
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Response
        results = []
        for i, txn in enumerate(transactions):
            results.append({
                'transaction_id': txn.get('id', f'txn_{i}'),
                'prediction': int(predictions[i]),
                'fraud_probability': float(probabilities[i, 1]),
                'recommendation': 'DECLINE' if predictions[i] == 1 else 'APPROVE',
                'confidence': float(max(probabilities[i]))
            })
        
        return jsonify({
            'status': 'success',
            'total': len(transactions),
            'results': results
        }), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Batch predict with statistics"""
    try:
        data = request.get_json()
        transactions = data.get('transactions', [])
        
        if not transactions:
            return jsonify({'error': 'Empty transactions'}), 400
        
        logger.info(f"Batch predicting {len(transactions)} transactions")
        
        X = np.array([preprocess_transaction(txn) for txn in transactions])
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        fraud_count = (predictions == 1).sum()
        
        return jsonify({
            'status': 'success',
            'total': len(transactions),
            'fraud_detected': int(fraud_count),
            'fraud_rate': float(fraud_count / len(transactions)),
            'statistics': {
                'mean_fraud_prob': float(probabilities[:, 1].mean()),
                'max_fraud_prob': float(probabilities[:, 1].max()),
                'min_fraud_prob': float(probabilities[:, 1].min())
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Batch error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================
# RUN
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("✅ API READY!")
    print("="*70)
    print("\n🚀 Starting server at http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /health - Health check")
    print("  GET  /info - Model info")
    print("  POST /predict - Single/batch prediction")
    print("  POST /batch-predict - Batch with stats")
    print("\n" + "="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)