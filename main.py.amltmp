# import uvicorn
# import pandas as pd
# from fastapi import FastAPI
# from pydantic import BaseModel
# from catboost import CatBoostClassifier

# # ✅ Initialize FastAPI app
# app = FastAPI()

# # ✅ Load model globally
# model = CatBoostClassifier()
# model.load_model("catboost_model.cbm")  # Ensure path is correct

# # ✅ Define expected input schema
# class PredictionInput(BaseModel):
#     gender: int
#     age: float
#     hypertension: int
#     heart_disease: int
#     ever_married: int
#     work_type: int
#     Residence_type: int
#     avg_glucose_level: float
#     bmi: float
#     smoking_status: int

# @app.post("/predict")
# async def predict(data: PredictionInput):
#     try:
#         # ✅ Convert input to DataFrame
#         input_df = pd.DataFrame([data.dict()])

#         # ✅ Ensure all data is float for CatBoost
#         input_df = input_df.astype(float)

#         # ✅ Make prediction
#         prediction = model.predict(input_df)

#         return {"predictions": prediction.tolist()}

#     except Exception as e:
#         return {"error": str(e)}

# # ✅ Run API locally
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI
import joblib
import pandas as pd
import uvicorn
import requests
import json

# ✅ Load the deployed model from Azure ML
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
        # ✅ Send data to the Azure ML Model API
        headers = {"Content-Type": "application/json"}
        response = requests.post(MODEL_ENDPOINT, data=json.dumps(data), headers=headers)

        # ✅ Return model response
        return response.json()
    
    except Exception as e:
        return {"error": str(e)}

# ✅ Run locally when executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
