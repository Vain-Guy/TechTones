from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="XGBoost Model API", description="Predictive API for status classification")

# Load model
model = joblib.load("C:\\Users\\lenovo\\OneDrive\\Desktop\\DS\\PROJECTS\\TechTones\\Model Deplyment\\xgboost_model.pkl")

# Define input schema
class ModelInput(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Welcome to the XGBoost Prediction API!"}

@app.post("/predict")
def predict(data: ModelInput):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0].tolist()
    return {
        "prediction": int(prediction),
        "probabilities": probabilities
    }
