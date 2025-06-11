import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('final_model.pkl')

st.title("💳 Credit Card Fraud Detection")

st.write("Enter transaction details to predict if it's fraudulent.")

# Input fields
time = st.number_input("⏱ Time (in seconds)", value=0.0, format="%.2f")
amount = st.number_input("💰 Amount", value=0.0, format="%.2f")

# V1 to V28 features
v_features = []
for i in range(1, 29):
    value = st.number_input(f"V{i}", value=0.0, format="%.5f")
    v_features.append(value)

# Prepare input
if st.button("Predict"):
    input_data = [time] + v_features + [amount]  # 30 features total
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    input_df = pd.DataFrame([input_data], columns=columns)

    # Prediction
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][prediction]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected!\nConfidence: {prob:.2%}")
    else:
        st.success(f"✅ Genuine Transaction.\nConfidence: {prob:.2%}")
