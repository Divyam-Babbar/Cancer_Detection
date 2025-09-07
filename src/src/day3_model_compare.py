# 📦 Model Comparison Script

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Load preprocessed dataset
df = pd.read_csv('../data/The_Cancer_data_1500_V2.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Encode categorical features
from sklearn.preprocessing import LabelEncoder, StandardScaler

categorical_features = ['Gender', 'Smoking', 'GeneticRisk', 'CancerHistory']
for col in categorical_features:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Scale numeric features
numerical_features = ['Age', 'BMI', 'PhysicalActivity', 'AlcoholIntake']
df[numerical_features] = StandardScaler().fit_transform(df[numerical_features])

# Split dataset
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train and evaluate
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "F1 Score": round(f1_score(y_test, y_pred), 4),
        "ROC AUC": round(roc_auc_score(y_test, y_prob), 4)
    })

# Output comparison
results_df = pd.DataFrame(results)
print(results_df)

# Export for Tableau
results_df.to_csv('../data/model_comparison_day3.csv', index=False)
print("\n✅ Model comparison saved to: model_comparison_day3.csv")
