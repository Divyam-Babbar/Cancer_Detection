# 📦 Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 📂 Load the dataset
df = pd.read_csv('../data/The_Cancer_data_1500_V2.csv')

# 🧹 Clean column names
df.columns = df.columns.str.strip()

# 🧠 Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 📊 Histogram distributions with diagnosis hue
important_features = ['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk',
                      'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']

for feature in important_features:
    subset = df[[feature, 'Diagnosis']].dropna()
    if subset[feature].nunique() < 2 or len(subset) < 2:
        print(f"⚠️ Skipping '{feature}' — not enough data or unique values.")
        continue
    use_kde = pd.api.types.is_numeric_dtype(subset[feature]) and subset[feature].nunique() > 10
    plt.figure(figsize=(6, 4))
    sns.histplot(data=subset, x=feature, hue='Diagnosis', kde=use_kde, palette='Set1')
    plt.title(f'Distribution of {feature} by Diagnosis')
    plt.tight_layout()
    plt.show()

# 📦 Boxplots
for feature in important_features:
    subset = df[[feature, 'Diagnosis']].dropna()
    if subset[feature].nunique() < 2 or len(subset) < 2:
        print(f"⚠️ Skipping boxplot for '{feature}' — not enough data or unique values.")
        continue
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Diagnosis', y=feature, data=subset, palette='Set2')
    plt.title(f'{feature} by Diagnosis')
    plt.tight_layout()
    plt.show()

# 🧠 Correlation with Diagnosis
correlation_with_diagnosis = df.corr()['Diagnosis'].drop('Diagnosis').sort_values(ascending=False)
print("\n📊 Correlation of features with Diagnosis:\n", correlation_with_diagnosis)

# ✅ Train a simple Logistic Regression model
X = df[important_features]
y = df['Diagnosis']

# 🧪 Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 📈 Evaluate
y_pred = model.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
