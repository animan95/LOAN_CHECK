import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and preprocessor
model = joblib.load("lgbm_credit_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Feature input form
st.title(" Loan Default Risk Predictor")
st.markdown("Enter borrower details to estimate the probability of loan default.")

with st.form("loan_form"):
    loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, max_value=50000, step=500)
    term = st.selectbox("Loan Term", [" 36 months", " 60 months"])
    int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=30.0, step=0.1)
    installment = st.number_input("Monthly Installment ($)", min_value=50.0, max_value=2000.0)
    grade = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
    sub_grade = st.selectbox("Sub Grade", [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)])
    emp_length = st.selectbox("Employment Length", ["< 1 year", "1 year", "2 years", "3 years", "4 years",
                                                     "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"])
    home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, step=1000)
    purpose = st.selectbox("Purpose", ["credit_card", "debt_consolidation", "home_improvement", "major_purchase", "small_business", "car", "medical", "vacation", "other"])
    addr_state = st.selectbox("State", ["CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "NC"])
    dti = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=40.0, step=0.1)
    delinq_2yrs = st.number_input("Delinquencies in Past 2 Years", min_value=0, max_value=10, step=1)
    fico_range_high = st.slider("FICO Score (High)", min_value=600, max_value=850, step=5)
    fico_range_low = st.slider("FICO Score (Low)", min_value=300, max_value=fico_range_high, step=5)
    open_acc = st.number_input("Number of Open Accounts", min_value=0, max_value=50)
    pub_rec = st.number_input("Number of Public Records", min_value=0, max_value=10)
    revol_util = st.slider("Revolving Utilization (%)", min_value=0.0, max_value=100.0, step=0.1)
    total_acc = st.number_input("Total Number of Accounts", min_value=0, max_value=100)

    submit = st.form_submit_button("Predict Risk")

if submit:
    input_data = pd.DataFrame([{
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "installment": installment,
        "grade": grade,
        "sub_grade": sub_grade,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "purpose": purpose,
        "addr_state": addr_state,
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "fico_range_high": fico_range_high,
        "fico_range_low": fico_range_low,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_util": revol_util,
        "total_acc": total_acc
    }])

    # Preprocess and predict
    processed = preprocessor.transform(input_data)
    pred_proba = model.predict(processed)[0]

    # Display result
    st.markdown(f"### 🔎 Estimated Default Probability: **{pred_proba:.2%}**")
    if pred_proba < 0.3:
        st.success("Low Risk")
    elif pred_proba < 0.6:
        st.warning("Moderate Risk")
    else:
        st.error(" High Risk")

