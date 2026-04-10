import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Page Configuration ---
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Animations and Aesthetics ---
st.markdown("""
    <style>
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #f0f2f6, #ffffff, #e6e9ef, #dbe4f0);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    /* Pulsing effect for the prediction button */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease 0s;
        animation: pulse 2s infinite;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1);
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
        100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("<h1 style='text-align: center; color: #2e4053;'>🩺 Interactive Diabetes Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #5d6d7e;'>Enter your health metrics below to get an instant ML-powered assessment.</p>", unsafe_allow_html=True)
st.divider()

# --- Load the Model ---
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Error: 'model.pkl' not found. Please ensure it is in the same directory as this script.")
    st.stop()

# --- Input Features (Using Columns for layout) ---
st.markdown("### 📊 Patient Health Data")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70, step=1)
    insulin = st.number_input("Insulin (IU/mL)", min_value=0, max_value=900, value=0, step=1)
    pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)

with col2:
    glucose = st.number_input("Glucose Level", min_value=0, max_value=250, value=100, step=1)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30, step=1)

st.write("") # Spacer

# --- Prediction Logic ---
if st.button("🔮 Predict Now"):
    # Create an array of the inputs. Note: The order MUST match the model's training order exactly.
    input_data = np.array([[
        pregnancies, 
        glucose, 
        blood_pressure, 
        skin_thickness, 
        insulin, 
        bmi, 
        pedigree, 
        age
    ]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display Results with Animation
    st.divider()
    st.markdown("### 📝 Results")
    
    if prediction[0] == 1:
        st.warning("🚨 **Prediction:** The model indicates a **high risk** of diabetes. Please consult with a healthcare professional for a proper medical diagnosis.")
    else:
        st.success("🎉 **Prediction:** The model indicates a **low risk** of diabetes. Keep up the healthy lifestyle!")
        st.balloons() # Triggers Streamlit's native animation
