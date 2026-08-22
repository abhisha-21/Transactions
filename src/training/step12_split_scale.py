import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

print("=" * 70)
print("STEP 3: Train-Test Split & Feature Scaling")
print("=" * 70)

# Load features and target
X = pd.read_csv('X_features.csv')
y = pd.read_csv('y_target.csv').values.ravel()

print(f"\n📊 Before Split:")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  Fraud rate: {y.mean()*100:.2f}%")

# ============================================================
# TRAIN-TEST SPLIT (70-30)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,           # 30% for testing
    random_state=42,         # For reproducibility
    stratify=y               # Keep fraud ratio same in both sets
)

print(f"\n✅ Train-Test Split (70-30, Stratified):")
print(f"  X_train: {X_train.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  y_test shape: {y_test.shape}")

print(f"\n  Train fraud rate: {y_train.mean()*100:.2f}%")
print(f"  Test fraud rate: {y_test.mean()*100:.2f}%")

# ============================================================
# FEATURE SCALING (StandardScaler)
# ============================================================

print(f"\n📊 Scaling Features (StandardScaler):")
print(f"  Mean = 0, Std = 1 for all features")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n  Train sample (first 3 transactions, first 5 features):")
print(f"  Original:")
print(X_train.iloc[:3, :5].values)
print(f"  Scaled:")
print(X_train_scaled[:3, :5])

# Verify scaling
print(f"\n  Verification (scaled train data):")
print(f"    Mean: {X_train_scaled.mean(axis=0)[:5]}")  # Should be ~0
print(f"    Std: {X_train_scaled.std(axis=0)[:5]}")    # Should be ~1

# ============================================================
# SAVE FOR NEXT STEP
# ============================================================

# Save as numpy arrays
np.save('X_train_scaled.npy', X_train_scaled)
np.save('X_test_scaled.npy', X_test_scaled)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# Save scaler for production (important!)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Also save feature names
feature_cols = X_train.columns.tolist()
np.save('feature_names.npy', np.array(feature_cols))

print(f"\n✓ Saved:")
print(f"  X_train_scaled.npy - {X_train_scaled.shape}")
print(f"  X_test_scaled.npy - {X_test_scaled.shape}")
print(f"  y_train.npy - {y_train.shape}")
print(f"  y_test.npy - {y_test.shape}")
print(f"  scaler.pkl - For production use")
print(f"  feature_names.npy - Feature list")

print(f"\n✓ Ready for Model Training!")