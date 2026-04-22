import streamlit as st
import pandas as pd
import sys
import os
import datetime

# Add src to path so we can import FraudPredictor
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from predict import FraudPredictor

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

st.title("💳 Real-time Credit Card Fraud Detection")
st.markdown("Enter transaction details below to evaluate the probability of fraud using the trained Machine Learning model.")

# Initialize Predictor
@st.cache_resource
def load_predictor():
    try:
        return FraudPredictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

predictor = load_predictor()

if predictor:
    # Build a user-friendly form
    with st.form("transaction_form"):
        st.subheader("Transaction Information")
        
        col1, col2 = st.columns(2)
        with col1:
            amt = st.number_input("Transaction Amount ($)", min_value=0.01, value=100.0, step=10.0)
            category = st.selectbox("Category", [
                'grocery_pos', 'entertainment', 'shopping_pos', 'misc_pos', 
                'shopping_net', 'gas_transport', 'misc_net', 'grocery_net', 
                'food_dining', 'health_fitness', 'kids_pets', 'travel', 
                'personal_care', 'home'
            ])
            trans_date = st.date_input("Transaction Date", datetime.date.today())
            trans_time = st.time_input("Transaction Time", datetime.datetime.now().time())
            
        with col2:
            gender = st.selectbox("Cardholder Gender", ['M', 'F'])
            dob = st.date_input("Cardholder Date of Birth", datetime.date(1980, 1, 1))
            
        st.subheader("Location Information")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**User Location**")
            lat = st.number_input("Latitude", value=40.7128)
            long = st.number_input("Longitude", value=-74.0060)
            city_pop = st.number_input("City Population", min_value=1, value=100000)
            
        with col4:
            st.markdown("**Merchant Location**")
            merch_lat = st.number_input("Merchant Latitude", value=40.7306)
            merch_long = st.number_input("Merchant Longitude", value=-73.9866)
            
        submitted = st.form_submit_button("Predict Fraud Probability")
        
    if submitted:
        # Construct expected input dictionary
        trans_datetime = datetime.datetime.combine(trans_date, trans_time).strftime('%Y-%m-%d %H:%M:%S')
        input_data = {
            'amt': amt,
            'category': category,
            'trans_date_trans_time': trans_datetime,
            'gender': gender,
            'dob': dob.strftime('%Y-%m-%d'),
            'lat': lat,
            'long': long,
            'city_pop': city_pop,
            'merch_lat': merch_lat,
            'merch_long': merch_long
        }
        
        with st.spinner("Analyzing transaction..."):
            result = predictor.predict(input_data)
            
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if result['is_fraud']:
                st.error("🚨 FRAUDULENT TRANSACTION DETECTED 🚨")
            else:
                st.success("✅ LEGITIMATE TRANSACTION ✅")
                
        with col_res2:
            st.metric("Fraud Probability", f"{result['fraud_probability'] * 100:.2f}%")
            st.metric("Legitimate Probability", f"{result['legitimate_probability'] * 100:.2f}%")
            
else:
    st.warning("Model not found. Please navigate to the `src` directory and run `python train.py` to train and save the model before using the Streamlit app.")
