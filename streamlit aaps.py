import streamlit as st
import pandas as pd
import joblib

# Load trained model and columns
model = joblib.load("loan_risk_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🏦 Loan Default Risk Prediction")
st.write("Enter customer details")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
credit_history = st.selectbox("Credit History", [1, 0])

dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (months)", min_value=0)

# Prediction
if st.button("Predict"):

    # 1. Raw input dataframe
    input_data = {
        "Gender": gender,
        "Married": married,
        "Education": education,
        "Self_Employed": self_employed,
        "Credit_History": credit_history,
        "Dependents": dependents,
        "Property_Area": property_area,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term
    }

    df = pd.DataFrame([input_data])

    # 2. One-hot encoding (SAME as training)
    df = pd.get_dummies(df)

    # 3. Add missing columns from training
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # 4. Ensure same column order
    df = df[model_columns]

    # 5. Debug check (VERY IMPORTANT)
    st.write("FINAL SHAPE:", df.shape)   # MUST be (1, 19)

    # 6. Prediction
    pred = model.predict(df)[0]

    # 7. Output
    if pred == 1:
        st.success("✅ Low Risk: Loan Approved")
    else:
        st.error("❌ High Risk: Loan Default Possible")
    df = pd.DataFrame([input_data])

# same encoding as training
    df = pd.get_dummies(

        df,
        columns=["Property_Area", "Dependents", "Age_Group"],
        drop_first=False
  )

# add missing columns
    for col in model_columns:
       if col not in df.columns:
        df[col] = 0

# same column order
    df = df[model_columns]

    st.write("FINAL SHAPE:", df.shape)   # MUST be (1, 27)

    pred = model.predict(df)[0]

