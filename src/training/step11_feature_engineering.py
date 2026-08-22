import pandas as pd
import numpy as np

print("=" * 70)
print("STEP 2: Feature Engineering")
print("=" * 70)

# Load data
df = pd.read_csv('creditcard_with_indian_context.csv')

print(f"\n📊 Starting with {len(df.columns)} columns")

# ============================================================
# PART 1: Keep PCA Features (V1-V28)
# ============================================================

pca_features = [f'V{i}' for i in range(1, 29)]
print(f"\n✓ PCA Features: {len(pca_features)} features (V1-V28)")

# ============================================================
# PART 2: Create Amount Feature (Log Transform)
# ============================================================

df['Amount_log'] = np.log1p(df['Amount_INR'])
print(f"✓ Amount_log: Log-transformed amount (for better distribution)")

# ============================================================
# PART 3: Temporal Features
# ============================================================

temporal_features = ['hour_ist', 'day_of_week', 'is_weekend', 'is_night', 'is_payday']
print(f"✓ Temporal Features: {temporal_features}")

# ============================================================
# PART 4: Velocity Features
# ============================================================

velocity_features = ['txn_count_1h', 'txn_count_24h', 'unique_merchants_24h']
print(f"✓ Velocity Features: {velocity_features}")

# ============================================================
# PART 5: Location Features
# ============================================================

location_features = ['city_encoded', 'city_changes_24h']
print(f"✓ Location Features: {location_features}")

# ============================================================
# PART 6: Device & Network Features
# ============================================================

device_features = ['device_risk_score', 'network_encoded']
print(f"✓ Device Features: {device_features}")

# ============================================================
# PART 7: Merchant Encoding
# ============================================================

merchant_encoding = {m: i for i, m in enumerate(df['merchant_category'].unique())}
df['merchant_encoded'] = df['merchant_category'].map(merchant_encoding)
merchant_features = ['merchant_encoded']
print(f"✓ Merchant Feature: {len(merchant_encoding)} unique merchants encoded")

# ============================================================
# COMBINE ALL FEATURES
# ============================================================

all_features = (
    pca_features +           # 28 features
    ['Amount_log'] +          # 1 feature
    temporal_features +       # 5 features
    velocity_features +       # 3 features
    location_features +       # 2 features
    device_features +         # 2 features
    merchant_features         # 1 feature
)

print(f"\n" + "="*70)
print(f"✅ TOTAL FEATURES: {len(all_features)}")
print(f"="*70)
print(f"\nFeature Breakdown:")
print(f"  PCA Features: 28")
print(f"  Amount: 1")
print(f"  Temporal: 5")
print(f"  Velocity: 3")
print(f"  Location: 2")
print(f"  Device: 2")
print(f"  Merchant: 1")
print(f"  TOTAL: {len(all_features)}")

# Prepare X and y
X = df[all_features]
y = df['Class']

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"\nFeature columns saved:")
for i, feat in enumerate(all_features, 1):
    print(f"  {i}. {feat}")

# Save for next step
X.to_csv('X_features.csv', index=False)
y.to_csv('y_target.csv', index=False)

# Save feature list
np.save('feature_columns.npy', np.array(all_features))

print(f"\n✓ X_features.csv - Input features (284807 x {len(all_features)})")
print(f"✓ y_target.csv - Target variable (fraud/legitimate)")
print(f"✓ feature_columns.npy - Feature names for reference")