import pandas as pd
import numpy as np

print("=" * 70)
print("STEP 6: Add Payday Patterns (Indian Paydays)")
print("=" * 70)

df = pd.read_csv('data_with_temporal.csv')

# Indian paydays: 1st, 5th, 15th, 25th of each month
payday_dates = [1, 5, 15, 25]
df['is_payday'] = df['day_of_month'].isin(payday_dates).astype(int)

print(f"\n💰 Indian payday pattern added")
print(f"  Payday dates: {payday_dates}")

print(f"\nPayday distribution:")
print(f"  Payday: {(df['is_payday']==1).sum():,} transactions")
print(f"  Non-payday: {(df['is_payday']==0).sum():,} transactions")

# Note: Fraud is often higher on paydays (people have more money)
print(f"\n📊 Insight: Monitor fraud spikes on paydays")

df.to_csv('data_with_payday.csv', index=False)
print(f"\n✓ Saved to data_with_payday.csv")
print("✓ Ready for Step 7!")