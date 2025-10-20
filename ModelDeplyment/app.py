import streamlit as st
import joblib
import numpy as np
import xgboost as xgb
import os

# App Title
st.set_page_config(page_title="TechTones Sentiment Analyzer", page_icon="🤖")
st.title("TechTones — Sentiment Analyzer for Apple & Google Tweets")

# Load Model
@st.cache_resource
def load_model():
    model_path = "xgboost_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please ensure xgboost_model.pkl is in the same directory.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# User Input
st.subheader("Enter preprocessed feature values (TF-IDF or vectorized features):")
user_input = st.text_input("Comma-separated values (e.g., 0.12, 0.03, 0.56, 0.08):")

if st.button("Predict Sentiment"):
    try:
        # Convert input to numpy array
        features = np.array([float(x.strip()) for x in user_input.split(",")]).reshape(1, -1)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        labels = {0: "Negative 😠", 1: "Neutral 😐", 2: "Positive 😄"}
        st.success(f"**Prediction:** {labels.get(prediction, 'Unknown')}")

        st.write("**Class Probabilities:**")
        st.json({
            "Negative": float(probabilities[0]),
            "Neutral": float(probabilities[1]) if len(probabilities) > 2 else 0.0,
            "Positive": float(probabilities[-1])
        })

    except Exception as e:
        st.error(f"Error: {e}")
