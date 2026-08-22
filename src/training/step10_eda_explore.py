import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("STEP 1: Exploratory Data Analysis (EDA)")
print("=" * 70)

# Load your synthesized dataset
df = pd.read_csv('creditcard_with_indian_context.csv')

print(f"\n📊 Dataset Overview:")
print(f"  Total transactions: {len(df):,}")
print(f"  Total columns: {len(df.columns)}")
print(f"  Date range: {df['datetime_ist'].min()} to {df['datetime_ist'].max()}")

# Class distribution
print(f"\n🎯 Class Distribution (Fraud vs Legitimate):")
fraud_count = (df['Class'] == 1).sum()
legit_count = (df['Class'] == 0).sum()
fraud_rate = df['Class'].mean() * 100

print(f"  Fraud: {fraud_count:,} ({fraud_rate:.2f}%)")
print(f"  Legitimate: {legit_count:,} ({100-fraud_rate:.2f}%)")

# Amount statistics
print(f"\n💰 Amount Statistics (INR):")
print(f"  Mean: ₹{df['Amount_INR'].mean():.2f}")
print(f"  Median: ₹{df['Amount_INR'].median():.2f}")
print(f"  Min: ₹{df['Amount_INR'].min():.2f}")
print(f"  Max: ₹{df['Amount_INR'].max():.2f}")
print(f"  Std Dev: ₹{df['Amount_INR'].std():.2f}")

# Merchant analysis
print(f"\n🏪 Merchant Distribution:")
print(df['merchant_category'].value_counts().head(10))

# City analysis
print(f"\n📍 City Distribution:")
print(df['city'].value_counts())

# Temporal patterns
print(f"\n⏰ Hourly Distribution (IST):")
hourly = df['hour_ist'].value_counts().sort_index()
print(hourly.head(12))

# Fraud by merchant
print(f"\n🚨 Fraud Rate by Merchant:")
fraud_by_merchant = df.groupby('merchant_category')['Class'].agg(['sum', 'count', 'mean'])
fraud_by_merchant.columns = ['fraud_count', 'total', 'fraud_rate']
fraud_by_merchant['fraud_rate'] = fraud_by_merchant['fraud_rate'] * 100
print(fraud_by_merchant.sort_values('fraud_rate', ascending=False).head(10))

# Fraud by time of day
print(f"\n🌙 Fraud Rate by Hour (IST):")
fraud_by_hour = df.groupby('hour_ist')['Class'].mean() * 100
print(fraud_by_hour.sort_values(ascending=False).head(10))

# Fraud by day of week
print(f"\n📅 Fraud Rate by Day of Week:")
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
fraud_by_day = df.groupby('day_of_week')['Class'].mean() * 100
for day_num, fraud_pct in fraud_by_day.items():
    print(f"  {day_names[day_num]}: {fraud_pct:.2f}%")

# Fraud by night vs day
print(f"\n🌞 Fraud Rate (Night vs Day):")
day_fraud = df[df['is_night']==0]['Class'].mean() * 100
night_fraud = df[df['is_night']==1]['Class'].mean() * 100
print(f"  Day (6 AM - 10 PM): {day_fraud:.2f}%")
print(f"  Night (10 PM - 6 AM): {night_fraud:.2f}%")

# Device risk
print(f"\n🔐 Device Risk Score:")
print(f"  Low (<0.3): {(df['device_risk_score'] < 0.3).sum():,}")
print(f"  Medium (0.3-0.7): {((df['device_risk_score'] >= 0.3) & (df['device_risk_score'] < 0.7)).sum():,}")
print(f"  High (>0.7): {(df['device_risk_score'] > 0.7).sum():,}")

# Velocity analysis
print(f"\n📊 Velocity Features:")
print(f"  Txns/hour - Max: {df['txn_count_1h'].max()}, Mean: {df['txn_count_1h'].mean():.2f}")
print(f"  Txns/24h - Max: {df['txn_count_24h'].max()}, Mean: {df['txn_count_24h'].mean():.2f}")
print(f"  Merchants/24h - Max: {df['unique_merchants_24h'].max()}, Mean: {df['unique_merchants_24h'].mean():.2f}")

# Check for missing values
print(f"\n❌ Missing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  None! Dataset is clean.")
else:
    print(missing[missing > 0])

print(f"\n✓ EDA Complete!")