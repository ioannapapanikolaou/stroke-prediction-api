from fastapi import FastAPI
import requests
import json

app = FastAPI()

# The Azure ML Model Scoring Endpoint
AZURE_ML_ENDPOINT = "http://57c1d30e-f22e-414a-811a-0c6f2e757de0.westeurope.azurecontainer.io/score"  

@app.get("/")
def home():
    return {"message": "Welcome to the CatBoost Thrombosis Prediction API!"}

@app.post("/predict")
def predict(data: dict):
    """Accepts input JSON and sends it to Azure ML for prediction"""
    
    headers = {"Content-Type": "application/json"}
    response = requests.post(AZURE_ML_ENDPOINT, json=data, headers=headers)
    
    # Handle errors
    if response.status_code != 200:
        return {"error": "Prediction failed", "details": response.json()}
    
    return response.json()

# Run the API: `uvicorn app:app --host 0.0.0.0 --port 8000`
