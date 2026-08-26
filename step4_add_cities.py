import pandas as pd
import numpy as np

print("=" * 70)
print("STEP 4: Add Indian Cities & Location Features")
print("=" * 70)

df = pd.read_csv('data_with_merchants.csv')
np.random.seed(42)

# 10 major Indian metro cities
cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Pune', 
          'Chennai', 'Kolkata', 'Jaipur', 'Ahmedabad', 'Surat']

print(f"\n Adding {len(cities)} Indian cities:")
print(f"  {', '.join(cities)}")

# Assign cities randomly
df['city'] = np.random.choice(cities, len(df))

# Encode cities numerically (for modeling later)
city_to_code = {city: idx for idx, city in enumerate(cities)}
df['city_encoded'] = df['city'].map(city_to_code)

# Geographic anomaly: same card used in different cities within 24h
df['city_changes_24h'] = np.random.poisson(0.5, len(df))  
df['city_changes_24h'] = df['city_changes_24h'].clip(upper=3)  # Cap at 3

print(f"\nCity distribution:")
print(df['city'].value_counts())

print(f"\nGeographic anomalies (city changes in 24h):")
print(df['city_changes_24h'].value_counts().sort_index())

df.to_csv('data_with_cities.csv', index=False)
print(f"\n Saved to data_with_cities.csv")
