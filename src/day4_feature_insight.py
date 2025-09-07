# 📦 Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 📂 Load and clean dataset
df = pd.read_csv('../data/The_Cancer_data_1500_V2.csv')
df.columns = df.columns.str.strip()

# 🔄 Encode categorical features
categorical_features = ['Gender', 'Smoking', 'GeneticRisk', 'CancerHistory']
for col in categorical_features:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# 🔢 Standardize numeric features
numerical_features = ['Age', 'BMI', 'PhysicalActivity', 'AlcoholIntake']
df[numerical_features] = StandardScaler().fit_transform(df[numerical_features])

# 🎯 Check target class distribution
sns.countplot(x='Diagnosis', data=df, palette='Set2')
plt.title("Target Class Distribution")
plt.show()

# 📊 1. Correlation with Diagnosis
corr_matrix = df.corr()
diagnosis_corr = corr_matrix['Diagnosis'].drop('Diagnosis').sort_values(ascending=False)
print("\n📈 Features most correlated with Diagnosis:\n", diagnosis_corr)

# 📌 Top 5 features (positively or negatively correlated)
top_features = diagnosis_corr.abs().sort_values(ascending=False).head(5).index.tolist()

# 🔍 2. Pairplot of top features
sns.pairplot(df[top_features + ['Diagnosis']], hue='Diagnosis', palette='Set1')
plt.suptitle("Top Correlated Features with Diagnosis", y=1.02)
plt.show()

# 🎻 3. Violin Plots for top features
for feature in top_features:
    plt.figure(figsize=(6, 4))
    sns.violinplot(x='Diagnosis', y=feature, data=df, palette='muted', inner='quartile')
    plt.title(f'{feature} vs Diagnosis (Violin Plot)')
    plt.tight_layout()
    plt.show()

# 📈 4. Boxplots for top features
for feature in top_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Diagnosis', y=feature, data=df, palette='pastel')
    plt.title(f'{feature} vs Diagnosis (Boxplot)')
    plt.tight_layout()
    plt.show()

# ⚡️ Optional: PCA visualization
from sklearn.decomposition import PCA

X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Diagnosis'] = y.values

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Diagnosis', data=pca_df, palette='Set1')
plt.title("PCA - 2D Projection of Dataset")
plt.show()
