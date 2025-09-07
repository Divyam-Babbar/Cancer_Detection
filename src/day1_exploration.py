# 📦 Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 📂 Load your dataset
df = pd.read_csv('../data/The_Cancer_data_1500_V2.csv')

# 🔍 Preview the data
print("Preview of data:")
print(df.head(60))

# 📊 Data shape and columns
print("\nShape of data:", df.shape)
print("\nColumns:\n", df.columns)

# ❓ Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# 🎯 Diagnosis (target) column distribution
print("\nDiagnosis distribution:\n", df['Diagnosis'].value_counts())

# 📈 Plot diagnosis distribution
sns.countplot(data=df, x='Diagnosis')
plt.title("Diagnosis Distribution")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
