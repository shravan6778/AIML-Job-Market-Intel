import pandas as pd

df = pd.read_csv('data/raw/jobs_raw.csv')

print("Shape:", df.shape)
print("\nColumn dtypes:\n", df.dtypes)
print("\nNull counts:\n", df.isnull().sum())
print("\nSample rows:")
print(df.head(5).to_string())