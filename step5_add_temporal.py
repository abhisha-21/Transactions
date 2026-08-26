import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 70)
print("STEP 5: Add Temporal Features (IST Timezone)")
print("=" * 70)

df = pd.read_csv('data_with_cities.csv')
np.random.seed(42)

# Convert Time column to IST datetime
# Assume Time is in seconds from some start point
start_time = datetime(2023, 1, 1, 0, 0, 0)  # IST
df['datetime_ist'] = df['Time'].apply(
    lambda x: start_time + timedelta(seconds=int(x))
)

# Extract temporal features
df['hour_ist'] = df['datetime_ist'].dt.hour  # 0-23 (IST)
df['day_of_week'] = df['datetime_ist'].dt.dayofweek  # 0=Monday, 6=Sunday
df['day_of_month'] = df['datetime_ist'].dt.day  # 1-31

# Binary features
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # Saturday, Sunday
df['is_night'] = ((df['hour_ist'] >= 22) | (df['hour_ist'] <= 5)).astype(int)  # Night = 10 PM - 5 AM

print(f"\n Temporal features added:")
print(f"  • Hour of day (0-23 IST)")
print(f"  • Day of week (0-6)")
print(f"  • Is weekend (0/1)")
print(f"  • Is night (0/1)")

print(f"\nSample timestamps (IST):")
print(df[['Time', 'datetime_ist', 'hour_ist', 'day_of_week']].head(10))

print(f"\nHour distribution:")
print(df['hour_ist'].value_counts().sort_index().head(10))

print(f"\nWeekend vs Weekday:")
print(f"  Weekday: {(df['is_weekend']==0).sum():,} transactions")
print(f"  Weekend: {(df['is_weekend']==1).sum():,} transactions")

df.to_csv('data_with_temporal.csv', index=False)
print(f"\n Saved to data_with_temporal.csv")
