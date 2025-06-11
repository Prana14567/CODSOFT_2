import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load('final_model.pkl')  # Trained model
scaler = joblib.load('scaler.pkl')      # Scaler for 'Time' and 'Amount' only

# App Title
st.title("🔍 Credit Card Fraud Detection")

# Input for Time and Amount
time = st.number_input("Transaction Time", min_value=0.0)
amount = st.number_input("Transaction Amount", min_value=0.0)

# Inputs for V1 to V28
v_inputs = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0, format="%.5f")
    v_inputs.append(val)

# Prediction Button
if st.button("Predict Fraud"):
    # Create input DataFrame
    input_data = [time] + v_inputs + [amount]
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    input_df = pd.DataFrame([input_data], columns=columns)

    # Scale only 'Time' and 'Amount'
    input_df[['Time', 'Amount']] = scaler.transform(input_df[['Time', 'Amount']])

    # Make prediction
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][prediction]

    # Display result
    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected (Confidence: {confidence:.2%})")
    else:
        st.success(f"✅ Genuine Transaction (Confidence: {confidence:.2%})")
