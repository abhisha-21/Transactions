import pandas as pd
import numpy as np

print("=" * 70)
print("STEP 3: Add Indian Merchant Categories")
print("=" * 70)

# Load data with INR amounts
df = pd.read_csv('data_with_inr.csv')
np.random.seed(42)

# 15 realistic Indian merchants
merchants = {
    'swiggy': {'fraud_rate': 0.08, 'min_amt': 100, 'max_amt': 1500},
    'paytm': {'fraud_rate': 0.05, 'min_amt': 50, 'max_amt': 2000},
    'flipkart': {'fraud_rate': 0.12, 'min_amt': 200, 'max_amt': 25000},
    'amazon': {'fraud_rate': 0.10, 'min_amt': 300, 'max_amt': 30000},
    'uber': {'fraud_rate': 0.04, 'min_amt': 50, 'max_amt': 800},
    'oyo': {'fraud_rate': 0.06, 'min_amt': 1000, 'max_amt': 8000},
    'byjus': {'fraud_rate': 0.09, 'min_amt': 5000, 'max_amt': 15000},
    'gpay': {'fraud_rate': 0.03, 'min_amt': 100, 'max_amt': 5000},
    'airtel': {'fraud_rate': 0.02, 'min_amt': 200, 'max_amt': 2000},
    'jio': {'fraud_rate': 0.02, 'min_amt': 100, 'max_amt': 1500},
    'bill_payment': {'fraud_rate': 0.01, 'min_amt': 500, 'max_amt': 10000},
    'utility': {'fraud_rate': 0.01, 'min_amt': 100, 'max_amt': 5000},
    'petrol_pump': {'fraud_rate': 0.03, 'min_amt': 500, 'max_amt': 3000},
    'atm': {'fraud_rate': 0.05, 'min_amt': 1000, 'max_amt': 20000},
    'pos_retail': {'fraud_rate': 0.15, 'min_amt': 50, 'max_amt': 5000},
}

merchant_list = list(merchants.keys())
print(f"\n🏪 Added {len(merchants)} merchant categories:")
for m in merchant_list:
    print(f"  • {m}")

# Randomly assign merchants
df['merchant_category'] = np.random.choice(merchant_list, len(df))

print(f"\nMerchant distribution:")
print(df['merchant_category'].value_counts())

# Save
df.to_csv('data_with_merchants.csv', index=False)
print(f"\n✓ Saved to data_with_merchants.csv")
print("✓ Ready for Step 4!")