import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    roc_auc_score, f1_score, precision_score, recall_score, roc_curve
)

print("=" * 70)
print("STEP 5: Model Evaluation & Visualization")
print("=" * 70)

# Load model and data
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

X_test_scaled = np.load('X_test_scaled.npy')
y_test = np.load('y_test.npy')

print(f"\n✓ Model and test data loaded")

# ============================================================
# MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print(f"\n🎯 Predictions Made")

# ============================================================
# CLASSIFICATION METRICS
# ============================================================

print(f"\n" + "="*70)
print(f"CLASSIFICATION METRICS")
print(f"="*70)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n📊 Key Metrics:")
print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1-Score: {f1:.4f}")
print(f"  ROC-AUC: {roc_auc:.4f}")

print(f"\n📈 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

# ============================================================
# CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(f"\n🔍 Confusion Matrix:")
print(f"  True Negatives (Correct Legit): {tn:,}")
print(f"  False Positives (Wrongly Declined): {fp:,}")
print(f"  False Negatives (Missed Fraud): {fn:,}")
print(f"  True Positives (Caught Fraud): {tp:,}")

# ============================================================
# BUSINESS METRICS
# ============================================================

false_positive_cost = 50  # ₹ per declined transaction
fraud_value = 10000       # ₹ avg fraud amount

fp_cost_total = fp * false_positive_cost
tp_value_prevented = tp * fraud_value

print(f"\n" + "="*70)
print(f"BUSINESS IMPACT (per {len(y_test):,} test transactions)")
print(f"="*70)

print(f"\n💰 Business Metrics:")
print(f"  Fraud Caught: {tp:,} transactions")
print(f"  Fraud Prevented Value: ₹{tp_value_prevented:,.0f}")
print(f"  Declined Legitimate: {fp:,} transactions")
print(f"  False Positive Cost: ₹{fp_cost_total:,.0f}")
print(f"  Net Value: ₹{tp_value_prevented - fp_cost_total:,.0f}")

# ============================================================
# RISK METRICS
# ============================================================

false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"\n⚠️  Risk Metrics:")
print(f"  False Positive Rate: {false_positive_rate*100:.2f}%")
print(f"  False Negative Rate: {false_negative_rate*100:.2f}%")

# ============================================================
# SAVE METRICS
# ============================================================

metrics_dict = {
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'ROC-AUC': roc_auc,
    'True Positives': tp,
    'False Positives': fp,
    'False Negatives': fn,
    'True Negatives': tn,
    'FP Cost (₹)': fp_cost_total,
    'Fraud Prevented (₹)': tp_value_prevented,
    'Net Value (₹)': tp_value_prevented - fp_cost_total
}

metrics_df = pd.DataFrame([metrics_dict])
metrics_df.to_csv('model_metrics.csv', index=False)

print(f"\n✓ Metrics saved to model_metrics.csv")

# ============================================================
# CREATE VISUALIZATIONS
# ============================================================

print(f"\n📊 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('AI Risk Manager: Fraud Detection Model Performance', fontsize=16, fontweight='bold')

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC={roc_auc:.4f})', linewidth=2)
axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curve')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
axes[0, 1].plot(recall_vals, precision_vals, linewidth=2, color='green')
axes[0, 1].set_xlabel('Recall')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision-Recall Curve')
axes[0, 1].grid(alpha=0.3)

# 3. Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')

# 4. Feature Importance (Top 10)
feature_importance = model.feature_importances_
feature_names = np.load('feature_names.npy')
top_indices = np.argsort(feature_importance)[-10:]
axes[1, 1].barh(range(len(top_indices)), feature_importance[top_indices])
axes[1, 1].set_yticks(range(len(top_indices)))
axes[1, 1].set_yticklabels(feature_names[top_indices])
axes[1, 1].set_title('Top 10 Feature Importance')
axes[1, 1].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved model_performance.png")

plt.close()

print(f"\n" + "="*70)
print(f"✅ EVALUATION COMPLETE!")
print(f"="*70)
print(f"\nSummary:")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall: {recall*100:.2f}%")
print(f"  ROC-AUC: {roc_auc:.4f}")
print(f"  Net Business Value: ₹{tp_value_prevented - fp_cost_total:,.0f}")
print(f"\n✓ All files saved (model.pkl, model_metrics.csv, model_performance.png)")