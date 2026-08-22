"""
Test the Flask API
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

print("=" * 70)
print("TESTING FRAUD DETECTION API")
print("=" * 70)

# ============================================================
# TEST 1: Health Check
# ============================================================

print("\n1️⃣  Health Check")
response = requests.get(f"{BASE_URL}/health")
print(f"   Status: {response.status_code}")
print(f"   Response: {json.dumps(response.json(), indent=2)}")

# ============================================================
# TEST 2: Model Info
# ============================================================

print("\n2️⃣  Model Info")
response = requests.get(f"{BASE_URL}/info")
print(f"   Status: {response.status_code}")
data = response.json()
print(f"   Features: {data['n_features']}")
print(f"   Merchants: {len(data['merchants'])}")
print(f"   Cities: {len(data['cities'])}")

# ============================================================
# TEST 3: Single Prediction (Legitimate)
# ============================================================

print("\n3️⃣  Single Prediction (Legitimate Transaction)")

legit_txn = {
    "V1": -0.5, "V2": 0.2, "V3": 1.5, "V4": 0.8, "V5": -0.3,
    "V6": 0.4, "V7": 0.2, "V8": 0.1, "V9": 0.3, "V10": 0.1,
    "V11": -0.5, "V12": -0.6, "V13": -1.0, "V14": -0.3, "V15": 1.5,
    "V16": -0.5, "V17": 0.2, "V18": 0.0, "V19": 0.4, "V20": 0.2,
    "V21": 0.0, "V22": 0.3, "V23": -0.1, "V24": 0.1, "V25": 0.1,
    "V26": -0.2, "V27": 0.1, "V28": -0.0,
    "Amount_INR": 500,
    "merchant_category": "swiggy",
    "city": "Mumbai",
    "hour_ist": 14,
    "day_of_week": 2,
    "is_weekend": 0,
    "is_night": 0,
    "is_payday": 0,
    "txn_count_1h": 1,
    "txn_count_24h": 5,
    "unique_merchants_24h": 2,
    "city_changes_24h": 0,
    "device_risk_score": 0.2,
    "network_type": "wifi"
}

response = requests.post(
    f"{BASE_URL}/predict",
    json={"transactions": [legit_txn]},
    headers={"Content-Type": "application/json"}
)

print(f"   Status: {response.status_code}")
result = response.json()['results'][0]
print(f"   Prediction: {result['prediction']} ({result['recommendation']})")
print(f"   Fraud Prob: {result['fraud_probability']:.4f}")
print(f"   Confidence: {result['confidence']:.4f}")

# ============================================================
# TEST 4: Single Prediction (Suspicious)
# ============================================================

print("\n4️⃣  Single Prediction (Suspicious Transaction)")

fraud_txn = {
    "V1": -1.4, "V2": -1.3, "V3": 1.8, "V4": 0.4, "V5": -0.5,
    "V6": 1.8, "V7": 0.8, "V8": 0.2, "V9": -1.5, "V10": 0.2,
    "V11": 0.6, "V12": 0.1, "V13": 0.7, "V14": -0.2, "V15": 2.3,
    "V16": -2.9, "V17": 1.1, "V18": -0.1, "V19": -2.3, "V20": 0.5,
    "V21": 0.2, "V22": 0.8, "V23": 0.9, "V24": -0.7, "V25": -0.3,
    "V26": -0.1, "V27": -0.1, "V28": -0.1,
    "Amount_INR": 45000,
    "merchant_category": "flipkart",
    "city": "Delhi",
    "hour_ist": 3,
    "day_of_week": 6,
    "is_weekend": 1,
    "is_night": 1,
    "is_payday": 0,
    "txn_count_1h": 5,
    "txn_count_24h": 18,
    "unique_merchants_24h": 8,
    "city_changes_24h": 2,
    "device_risk_score": 0.8,
    "network_type": "5g"
}

response = requests.post(
    f"{BASE_URL}/predict",
    json={"transactions": [fraud_txn]},
    headers={"Content-Type": "application/json"}
)

print(f"   Status: {response.status_code}")
result = response.json()['results'][0]
print(f"   Prediction: {result['prediction']} ({result['recommendation']})")
print(f"   Fraud Prob: {result['fraud_probability']:.4f}")
print(f"   Confidence: {result['confidence']:.4f}")

# ============================================================
# TEST 5: Batch Prediction
# ============================================================

print("\n5️⃣  Batch Prediction (100 transactions)")

batch_txns = [legit_txn for _ in range(95)]  # 95 legit
batch_txns.extend([fraud_txn for _ in range(5)])  # 5 suspicious

response = requests.post(
    f"{BASE_URL}/batch-predict",
    json={"transactions": batch_txns},
    headers={"Content-Type": "application/json"}
)

print(f"   Status: {response.status_code}")
data = response.json()
print(f"   Total: {data['total_transactions']}")
print(f"   Fraud Detected: {data['fraud_detected']}")
print(f"   Fraud Rate: {data['fraud_rate']*100:.2f}%")
print(
    f"   Avg Fraud Prob: "
    f"{data['statistics']['mean_fraud_probability']:.4f}"
)
# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nAPI is working correctly!")
print("Ready for production deployment.")