import pandas as pd
import numpy as np

print("=" * 70)
print("STEP 8: Add Device & Network Risk Features")
print("=" * 70)

df = pd.read_csv('data_with_velocity.csv')
np.random.seed(42)

# Device risk score (0.0 = trusted, 1.0 = risky)
df['device_risk_score'] = np.random.beta(2, 5, len(df))

# Network type
network_types = ['wifi', '4g', '5g']
df['network_type'] = np.random.choice(network_types, len(df))

# Encode network type numerically
network_mapping = {'wifi': 0, '4g': 1, '5g': 2}
df['network_encoded'] = df['network_type'].map(network_mapping)

print(f"\n Device & Network features added:")
print(f" Device risk score (0.0-1.0)")
print(f" Network type (wifi, 4g, 5g)")

print(f"\nDevice risk distribution:")
print(f"  Low risk (<0.3): {(df['device_risk_score'] < 0.3).sum():,} devices")
print(f"  Medium risk (0.3-0.7): {((df['device_risk_score'] >= 0.3) & (df['device_risk_score'] < 0.7)).sum():,} devices")
print(f"  High risk (>0.7): {(df['device_risk_score'] > 0.7).sum():,} devices")

print(f"\nNetwork type distribution:")
print(df['network_type'].value_counts())

df.to_csv('data_with_device_risk.csv', index=False)
print(f"\n Saved to data_with_device_risk.csv")
print(" Ready for Step 9!")
