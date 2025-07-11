import pandas as pd

df = pd.read_csv("data/processed/vehicle_features.csv")
print("Label distribution:")
print(df['label'].value_counts())
