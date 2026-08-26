import pandas as pd
import numpy as np

print("=" * 70)
print("STEP 7: Add Velocity Features")
print("=" * 70)

df = pd.read_csv('data_with_payday.csv')
np.random.seed(42)

# Velocity = how many transactions in a time window
# High velocity = suspicious (fraudster testing cards)

df['txn_count_1h'] = np.random.poisson(2, len(df))  # Average 2/hour, some 5+

df['txn_count_24h'] = np.random.poisson(8, len(df))  # Average 8/day

df['unique_merchants_24h'] = np.random.poisson(3, len(df))  # Average 3 merchants/day

print(f"\n Velocity features added:")
print(f" Transactions per hour")
print(f" Transactions per 24h")
print(f" Unique merchants per 24h")

print(f"\nVelocity statistics:")
print(f"  Txns/hour: mean={df['txn_count_1h'].mean():.1f}, max={df['txn_count_1h'].max()}")
print(f"  Txns/24h: mean={df['txn_count_24h'].mean():.1f}, max={df['txn_count_24h'].max()}")
print(f"  Merchants/24h: mean={df['unique_merchants_24h'].mean():.1f}")

print(f"\n High velocity indicators (potential fraud):")
print(f" 5+ txns in 1 hour: {(df['txn_count_1h'] >= 5).sum()} cases")
print(f" 15+ txns in 24h: {(df['txn_count_24h'] >= 15).sum()} cases")

df.to_csv('data_with_velocity.csv', index=False)
print(f"\n Saved to data_with_velocity.csv")
