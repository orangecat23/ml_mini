

import streamlit as st
import pandas as pd
import pickle

# Load model, scaler, and label encoders
model = pickle.load(open('financial_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

st.set_page_config(page_title="Personalized Financial Advisor", layout="centered")
st.title("💰 Personalized Financial Advisor (Inflation-Aware)")

st.markdown("This tool estimates your ideal monthly savings based on your personal financial data, adjusted for inflation.")

# --- Example default values ---
default_values = {
    "income": 50000,
    "age": 30,
    "dependents": 2,
    "rent": 12000,
    "loan": 2000,
    "insurance": 1500,
    "groceries": 6000,
    "transport": 2500,
    "eating_out": 1200,
    "entertainment": 1000,
    "utilities": 2000,
    "healthcare": 800,
    "education": 1500,
    "misc": 700,
    "disposable_income": 10000,
    "pot_savings_groceries": 800,
    "pot_savings_transport": 300,
    "pot_savings_eating_out": 200,
    "pot_savings_entertainment": 150,
    "pot_savings_utilities": 250,
    "pot_savings_healthcare": 100,
    "pot_savings_education": 200,
    "pot_savings_misc": 120,
    "pot_savings_insurance": 200,

    
}

# --- User Inputs ---
income = st.number_input("Monthly Income", min_value=0, value=default_values["income"])
age = st.number_input("Age", min_value=18, max_value=100, value=default_values["age"])
dependents = st.number_input("Number of Dependents", min_value=0, value=default_values["dependents"])

occupation = st.selectbox("Occupation", label_encoders['Occupation'].classes_)
city = st.selectbox("City Tier", label_encoders['City_Tier'].classes_)

rent = st.number_input("Monthly Rent", min_value=0, value=default_values["rent"])
loan = st.number_input("Loan Repayment", min_value=0, value=default_values["loan"])
insurance = st.number_input("Insurance", min_value=0, value=default_values["insurance"])
groceries = st.number_input("Groceries", min_value=0, value=default_values["groceries"])
transport = st.number_input("Transport", min_value=0, value=default_values["transport"])
eating_out = st.number_input("Eating Out", min_value=0, value=default_values["eating_out"])
entertainment = st.number_input("Entertainment", min_value=0, value=default_values["entertainment"])
utilities = st.number_input("Utilities", min_value=0, value=default_values["utilities"])
healthcare = st.number_input("Healthcare", min_value=0, value=default_values["healthcare"])
education = st.number_input("Education", min_value=0, value=default_values["education"])
misc = st.number_input("Miscellaneous", min_value=0, value=default_values["misc"])
disposable_income = st.number_input("Disposable Income", min_value=0, value=default_values["disposable_income"])

# Potential savings inputs
pot_savings_groceries = st.number_input("Potential Savings - Groceries", min_value=0.0, value=float(default_values["pot_savings_groceries"]))
pot_savings_transport = st.number_input("Potential Savings - Transport", min_value=0.0, value=float(default_values["pot_savings_transport"]))
pot_savings_eating_out = st.number_input("Potential Savings - Eating Out", min_value=0.0, value=float(default_values["pot_savings_eating_out"]))
pot_savings_entertainment = st.number_input("Potential Savings - Entertainment", min_value=0.0, value=float(default_values["pot_savings_entertainment"]))
pot_savings_utilities = st.number_input("Potential Savings - Utilities", min_value=0.0, value=float(default_values["pot_savings_utilities"]))
pot_savings_healthcare = st.number_input("Potential Savings - Healthcare", min_value=0.0, value=float(default_values["pot_savings_healthcare"]))
pot_savings_education = st.number_input("Potential Savings - Education", min_value=0.0, value=float(default_values["pot_savings_education"]))
pot_savings_misc = st.number_input("Potential Savings - Miscellaneous", min_value=0.0, value=float(default_values["pot_savings_misc"]))
pot_savings_insurance = st.number_input("Potential Savings - Insurance", min_value=0.0, value=float(default_values["pot_savings_insurance"]))


# Inflation slider
inflation_rate = st.slider("Current Inflation Rate (%)", 0.0, 15.0, 6.5)

# --- Predict Button ---
if st.button("💡 Predict Desired Savings"):
    # Encode occupation and city tier
    occ_encoded = label_encoders['Occupation'].transform([occupation])[0]
    city_encoded = label_encoders['City_Tier'].transform([city])[0]

    # Prepare input data in the correct order (26 features total)
    input_data = [[
        income, age, dependents, occ_encoded, city_encoded,
        rent, loan, insurance, groceries, transport,
        eating_out, entertainment, utilities, healthcare, education, misc,
        disposable_income,
        pot_savings_groceries, pot_savings_transport, pot_savings_eating_out,
        pot_savings_entertainment, pot_savings_utilities, pot_savings_healthcare,
        pot_savings_education, pot_savings_misc, pot_savings_insurance
    ]]
   # print("🧪 Number of features passed to model:", len(input_data[0]))

    # Transform and predict
    input_scaled = scaler.transform(input_data)
    predicted_savings = model.predict(input_scaled)[0]
    adjusted_savings = predicted_savings * (1 + inflation_rate / 100)

    # Output 
    #India uses changes in the CPI to measure its rate of inflation.
    st.success(f"Predicted Desired Savings (adjusted for inflation): ₹{adjusted_savings:,.2f}")
