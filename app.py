import streamlit as st
import numpy as np
import pandas as pd
import joblib


model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")


st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter the transaction details below:")


time = st.number_input("Time", value=10000.0)


v_features = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0)
    v_features.append(val)

amount = st.number_input("Amount", value=150.0)


input_data = [time] + v_features + [amount]
input_df = pd.DataFrame([input_data], columns=[
    'Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'])

if st.button("Predict"):
    # Scale input
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Genuine Transaction")
