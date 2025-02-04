# from fastapi import FastAPI
# import joblib
# import pandas as pd
# import uvicorn
# import requests
# import json

# # Load the deployed model from Azure ML
# MODEL_ENDPOINT = "http://6397adeb-32f3-4c39-88aa-be828b698c83.westeurope.azurecontainer.io/score"

# app = FastAPI()

# @app.get("/")
# def home():
#     return {"message": "CatBoost Thrombosis Prediction API is running!"}

# @app.post("/predict")
# def predict(data: dict):
#     """
#     Receives input JSON, processes it, and returns a prediction from the deployed Azure ML model.
#     """
#     try:
#         # Send data to the Azure ML Model API
#         headers = {"Content-Type": "application/json"}
#         response = requests.post(MODEL_ENDPOINT, data=json.dumps(data), headers=headers)

#         # Return model response
#         return response.json()
    
#     except Exception as e:
#         return {"error": str(e)}

# # Run locally when executed directly
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI
import joblib
import pandas as pd
import uvicorn
import requests
import json

# Load the deployed model from Azure ML
MODEL_ENDPOINT = "http://6397adeb-32f3-4c39-88aa-be828b698c83.westeurope.azurecontainer.io/score"

app = FastAPI()

@app.get("/")
def home():
    return {"message": "CatBoost Thrombosis Prediction API is running!"}

@app.post("/predict")
def predict(data: dict):
    """
    Receives input JSON, processes it, and returns a prediction from the deployed Azure ML model.
    """
    try:
        # Send data to the Azure ML Model API
        headers = {"Content-Type": "application/json"}
        response = requests.post(MODEL_ENDPOINT, data=json.dumps(data), headers=headers)

        # Return model response
        return response.json()
    
    except Exception as e:
        return {"error": str(e)}

# Ensure Azure App Service can correctly start the FastAPI app
def start():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run FastAPI when executed directly
if __name__ == "__main__":
    start()
