# application.py
import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# Load model and scaler
model = load("../models/xgboost_cancer_model.joblib")
scaler = load("../models/scaler.joblib")

st.title("🧠 Cancer Diagnosis Prediction App")
st.write("Enter patient data to predict the cancer diagnosis.")

# Inputs
age = st.slider("Age", 10, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
smoking = st.selectbox("Smoking", ["Yes", "No"])
genetic_risk = st.selectbox("Genetic Risk", ["Low", "Medium", "High"])
physical_activity = st.slider("Physical Activity (hrs/week)", 0, 20, 3)
alcohol_intake = st.slider("Alcohol Intake (units/week)", 0, 30, 5)
cancer_history = st.selectbox("Family Cancer History", ["Yes", "No"])

# Encode
gender = 1 if gender == "Male" else 0
smoking = 1 if smoking == "Yes" else 0
genetic_risk = {"Low": 0, "Medium": 1, "High": 2}[genetic_risk]
cancer_history = 1 if cancer_history == "Yes" else 0

# Create input DataFrame
input_df = pd.DataFrame([[
    age, bmi, physical_activity, alcohol_intake,
    gender, smoking, genetic_risk, cancer_history
]], columns=[
    'Age', 'BMI', 'PhysicalActivity', 'AlcoholIntake',
    'Gender', 'Smoking', 'GeneticRisk', 'CancerHistory'
])

# Reorder columns to match model training
ordered_cols = ['Age', 'BMI', 'PhysicalActivity', 'AlcoholIntake',
                'Gender', 'Smoking', 'GeneticRisk', 'CancerHistory']
input_df = input_df[ordered_cols]

# Scale numerical features only
numerical = ['Age', 'BMI', 'PhysicalActivity', 'AlcoholIntake']
input_df[numerical] = scaler.transform(input_df[numerical])

# Predict
if st.button("Predict Diagnosis"):
    prediction = model.predict(input_df)[0]
    result = "🧬 Positive (Cancer likely)" if prediction == 1 else "✅ Negative (No cancer)"
    st.success(f"Prediction: {result}")
