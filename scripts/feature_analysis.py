import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Correct local path
df = pd.read_csv("data/processed/vehicle_features.csv")

# Display basic info
df_info = df.describe(include='all')

# Compute correlation matrix
correlation_matrix = df.drop(columns=["label"]).corr()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()
