import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from .pydantic_models import PredictionInput, PredictionOutput
import logging
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="API for predicting customer credit risk for BNPL service.",
    version="1.0.0"
)

MODEL_NAME = "CreditRiskModel-BNPL"
MODEL_ALIAS = "champion-candidate"

client = MlflowClient()
version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
model_uri = f"models:/{MODEL_NAME}/{version}"
logging.info(f"Loading model from {model_uri}")

try:
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise HTTPException(status_code=500, detail="Model loading failed.")

@app.get("/", tags = ['Health Check'])
def read_root():
    """
    Health check endpoint to verify if the API is running.
    """
    return {"message": "Bati Bank Credit Risk API is running."} 


@app.post("/predict", response_model=PredictionOutput, tags=['Prediction'])
def predict(input_data: PredictionInput) -> PredictionOutput:
    
    try:
        df = pd.DataFrame([input_data.dict()])
        prediction_proba = model.predict(df)[0]
        THERESHOLD = 0.5
        is_high_risk = bool(prediction_proba >= THERESHOLD)
        
        return PredictionOutput(
            risk_probability=prediction_proba,
            is_high_risk=is_high_risk
        )
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")