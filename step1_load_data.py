import pandas as pd
import numpy as np

print("=" * 70)
print("STEP 1: Load the Kaggle Dataset")
print("=" * 70)

# Load your credit card data
df = pd.read_csv('creditcard.csv')

print(f"\n Loaded {len(df):,} transactions")
print(f" Columns: {list(df.columns)}")
print(f" Original fraud rate: {df['Class'].mean()*100:.2f}%")
print(f"\nFirst 3 rows:")
print(df.head(3))
print(f"\nDataset info:")
print(df.info())

# Save for next step
print("\n Ready for Step 2!")
