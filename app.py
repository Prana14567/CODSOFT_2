import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load('final_model.pkl')  # Make sure this file is in the same directory

# Streamlit App
st.title("💳 Credit Card Fraud Detection")

st.subheader("Enter Transaction Details:")

# Input fields
time = st.number_input("Transaction Time", min_value=0.0, format="%.2f")
amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")

v_features = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0, format="%.5f")
    v_features.append(val)

# When the button is clicked
if st.button("Predict"):
    # Prepare the input data in correct order
    input_data = [time] + v_features + [amount]
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    input_df = pd.DataFrame([input_data], columns=columns)

    # Prediction
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][prediction]

    st.subheader("Result:")
    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected!\nConfidence: {confidence:.2%}")
    else:
        st.success(f"✅ Genuine Transaction.\nConfidence: {confidence:.2%}")
