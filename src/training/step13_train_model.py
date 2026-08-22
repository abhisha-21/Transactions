import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier

print("=" * 70)
print("STEP 4: Train XGBoost Model")
print("=" * 70)

# Load scaled data
X_train_scaled = np.load('X_train_scaled.npy')
X_test_scaled = np.load('X_test_scaled.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(f"\n📊 Data Loaded:")
print(f"  X_train: {X_train_scaled.shape}")
print(f"  X_test: {X_test_scaled.shape}")
print(f"  Train fraud rate: {y_train.mean()*100:.2f}%")
print(f"  Test fraud rate: {y_test.mean()*100:.2f}%")

# ============================================================
# HANDLE CLASS IMBALANCE
# ============================================================

# Fraud is rare (~5-6%), so penalize false negatives
fraud_count = (y_train == 1).sum()
legit_count = (y_train == 0).sum()
scale_pos_weight = legit_count / fraud_count

print(f"\n⚖️  Class Imbalance Handling:")
print(f"  Fraud cases (1): {fraud_count:,}")
print(f"  Legit cases (0): {legit_count:,}")
print(f"  Scale pos weight: {scale_pos_weight:.2f}")
print(f"  (Penalize missing fraud ~{scale_pos_weight:.0f}x more than false alarms)")

# ============================================================
# CONFIGURE XGBoost
# ============================================================

print(f"\n🤖 XGBoost Configuration:")

model = XGBClassifier(
    n_estimators=200,           # 200 trees
    max_depth=6,                # Tree depth (prevent overfitting)
    learning_rate=0.1,          # Learning rate (0.1 = moderate)
    subsample=0.8,              # Use 80% of samples per tree
    colsample_bytree=0.8,       # Use 80% of features per tree
    scale_pos_weight=scale_pos_weight,  # Handle imbalance
    random_state=42,            # Reproducibility
    eval_metric='logloss',
    verbosity=1                 # Print progress
)

print(f"  n_estimators: 200")
print(f"  max_depth: 6")
print(f"  learning_rate: 0.1")
print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

# ============================================================
# TRAIN THE MODEL
# ============================================================

print(f"\n🔄 Training Model...")
print(f"  This may take 1-2 minutes...")

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=10  # Print every 10 iterations
)

print(f"\n✅ Model Training Complete!")

# ============================================================
# SAVE MODEL
# ============================================================

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"\n✓ Model saved to model.pkl")

# Get feature importance
feature_names = np.load('feature_names.npy')
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(f"\n📊 Top 10 Important Features:")
print(feature_importance_df.head(10))

# Save feature importance
feature_importance_df.to_csv('feature_importance.csv', index=False)

print(f"\n✓ Feature importance saved to feature_importance.csv")
print(f"\n✓ Ready for Model Evaluation!")