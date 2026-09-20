from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch Heart Disease dataset
heart_disease = fetch_ucirepo(id=45)

# Features
X = heart_disease.data.features

# Target
y = heart_disease.data.targets

# Combine features and target
df = pd.concat([X, y], axis=1)

# Save CSV
df.to_csv("data/heart_disease.csv", index=False)

print("Heart Disease dataset downloaded successfully!")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)
print("\nFirst 5 rows:")
print(df.head())