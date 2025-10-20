from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Initialize FastAPI app
app = FastAPI(
    title="TechTones Sentiment API",
    description="Predictive API for classifying sentiment using XGBoost model",
    version="1.0.0"
)

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgboost_model.pkl")
model = joblib.load(MODEL_PATH)

# Define input schema
class ModelInput(BaseModel):
    features: list  # e.g., TF-IDF vector or preprocessed feature list

@app.get("/")
def home():
    return {"message": "Welcome to the TechTones XGBoost Sentiment Prediction API!"}

@app.post("/predict")
def predict(data: ModelInput):
    # Convert input list to numpy array
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0].tolist()

    return {
        "prediction": int(prediction),
        "probabilities": probabilities
    }
