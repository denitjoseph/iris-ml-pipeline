import os
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Iris MLOps Level 2 API")

MODEL_NAME = "IrisRandomForestModel"

# Load latest model version from MLflow Registry
def load_latest_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_uri = f"models:/{MODEL_NAME}/latest"
    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Registry load warning ({e}); checking fallback artifact...")
        if os.path.exists("models/model.pkl"):
            return mlflow.sklearn.load_model("models/model.pkl")
        return None

model = load_latest_model()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(data: IrisInput):
    global model
    if model is None:
        # Try loading once more in case it was newly trained
        model = load_latest_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact unavailable.")

    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
