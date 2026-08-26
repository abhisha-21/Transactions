import pandas as pd
import numpy as np

print("=" * 70)
print("STEP 9: Calibrate Fraud by Merchant & Finalize")
print("=" * 70)

df = pd.read_csv('data_with_device_risk.csv')
np.random.seed(42)

# Merchant fraud rates (from reality-based estimates)
merchant_fraud_rates = {
    'swiggy': 0.08,
    'paytm': 0.05,
    'flipkart': 0.12,
    'amazon': 0.10,
    'uber': 0.04,
    'oyo': 0.06,
    'byjus': 0.09,
    'gpay': 0.03,
    'airtel': 0.02,
    'jio': 0.02,
    'bill_payment': 0.01,
    'utility': 0.01,
    'petrol_pump': 0.03,
    'atm': 0.05,
    'pos_retail': 0.15,
}

print(f"\n Calibrating fraud rates by merchant:")
print(f"  Low risk: utility, bill_payment (1-2%)")
print(f"  Medium risk: paytm, uber, swiggy (3-8%)")
print(f"  High risk: flipkart, amazon, pos_retail (10-15%)")

# Adjust fraud labels based on merchant
for merchant, fraud_rate in merchant_fraud_rates.items():
    mask = df['merchant_category'] == merchant
    fraud_indices = df[mask].sample(frac=fraud_rate, random_state=42).index
    df.loc[fraud_indices, 'Class'] = 1

print(f"\n Final fraud rate: {df['Class'].mean()*100:.2f}%")
print(f"   Fraud cases: {(df['Class']==1).sum():,}")
print(f"   Legit cases: {(df['Class']==0).sum():,}")

# Final check: confirm all required columns
required_cols = ['Time', 'V1', 'V2', 'Amount', 'Class', 'Amount_INR', 
                 'merchant_category', 'city', 'hour_ist', 'day_of_week', 
                 'is_weekend', 'is_night', 'is_payday', 'txn_count_1h', 
                 'txn_count_24h', 'unique_merchants_24h', 'city_changes_24h',
                 'device_risk_score', 'network_type']

print(f"\n Checking all columns present...")
for col in required_cols:
    if col in df.columns:
        print(f"   {col}")
    else:
        print(f"   MISSING: {col}")

# Save final dataset
output_path = 'creditcard_with_indian_context.csv'
df.to_csv(output_path, index=False)

print(f"\n" + "="*70)
print(f" DATA SYNTHESIS COMPLETE!")
print(f"="*70)
print(f"\n Final dataset: {output_path}")
print(f"   Size: {len(df):,} transactions")
print(f"   Columns: {len(df.columns)}")
print(f"   Fraud rate: {df['Class'].mean()*100:.2f}%")

print(f"\n Dataset summary:")
print(df.info())
print(f"\nFirst 5 rows:")
print(df.head())
