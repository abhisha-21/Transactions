import pandas as pd
import numpy as np

print("=" * 70)
print("STEP 2: Convert Amount to INR")
print("=" * 70)

# Load data from Step 1
df = pd.read_csv('creditcard.csv')

# Exchange rate: 1 USD = 95 INR (approximate)
USD_TO_INR = 95
np.random.seed(42)

print(f"\n Using exchange rate: 1 USD = {USD_TO_INR} INR")

# Convert to INR
df['Amount_INR'] = df['Amount'] * USD_TO_INR

# Add realistic variance (some merchants round, some don't)
variance = np.random.normal(1.0, 0.02, len(df))
df['Amount_INR'] = df['Amount_INR'] * variance

# Ensure no negative amounts
df['Amount_INR'] = df['Amount_INR'].clip(lower=10)

print(f"\nAmount stats (INR):")
print(f"  Min: ₹{df['Amount_INR'].min():.0f}")
print(f"  Max: ₹{df['Amount_INR'].max():.0f}")
print(f"  Mean: ₹{df['Amount_INR'].mean():.0f}")
print(f"  Median: ₹{df['Amount_INR'].median():.0f}")

print(f"\nSample transactions (INR):")
print(df[['Amount', 'Amount_INR']].head(10))

# Save intermediate data
df.to_csv('data_with_inr.csv', index=False)
print(f"\n Saved intermediate data to data_with_inr.csv")

