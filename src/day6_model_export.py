# ✅ day6_model_export.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from joblib import dump
import os

# 📂 Load dataset
df = pd.read_csv('../data/The_Cancer_data_1500_V2.csv')
df.columns = df.columns.str.strip()

# 🔄 Encode categorical features
categorical_features = ['Gender', 'Smoking', 'GeneticRisk', 'CancerHistory']
for col in categorical_features:
    df[col] = LabelEncoder().fit_transform(df[col])

# ⚖️ Scale numerical features
numerical_features = ['Age', 'BMI', 'PhysicalActivity', 'AlcoholIntake']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# 🎯 Define X and y
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']  # ✅ this is where 'y' is defined

# 🧪 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# 💾 Export model and scaler
os.makedirs('../models', exist_ok=True)
dump(model, '../models/xgboost_cancer_model.joblib')
dump(scaler, '../models/scaler.joblib')

print("✅ Model and scaler exported successfully.")
