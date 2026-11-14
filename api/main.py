import os
import joblib
import pandas as pd
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Churn Inference API", version="1.2.0")

APP_NAME = os.getenv("APP_NAME", "churn-inference-api")
APP_VERSION = os.getenv("APP_VERSION", "v1.2.0")
APP_ENV = os.getenv("APP_ENV", "production")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")

# cached model
_model = None


def get_model():
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception:
            _model = None
    return _model


class CustomerFeatures(BaseModel):
    age: float
    tenure: float
    monthly_charges: float
    total_charges: float


@app.get("/")
def root():
    return {"service": APP_NAME, "status": "running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "app_version": APP_VERSION,
        "environment": APP_ENV
    }


@app.get("/version")
def version():
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "git_commit": "3d2fbcf",
        "dvc_hash": "52aebc1284c804dab986501dcf8c8efe",
        "model_loaded": get_model() is not None
    }


@app.post("/predict")
def predict(features: CustomerFeatures):
    data = pd.DataFrame([features.model_dump()])
    model = get_model()

    if model is not None:
        try:
            prob = float(model.predict_proba(data)[0][1])
            pred = int(prob >= 0.5)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    else:
        # baseline heuristic fallback if weights not loaded
        prob = 0.85 if (features.monthly_charges > 80 and features.tenure < 12) else 0.15
        pred = int(prob >= 0.5)

    return {
        "prediction": pred,
        "churn_probability": prob,
        "churn_risk": "High" if pred == 1 else "Low",
        "model_version": APP_VERSION
    }
