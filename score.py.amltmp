import json
import numpy as np
import pandas as pd
from azureml.core.model import Model
from catboost import CatBoostClassifier

# Global variable for the model
model = None

# Load model when the service starts
def init():
    global model
    model_path = Model.get_model_path("catboost-thrombosis-predictor")  # Ensure the model name is correct
    model = CatBoostClassifier()
    model.load_model(model_path)  # Correct method for loading `.cbm` files
    print("✅ Model loaded successfully!")

# Run inference on input data
def run(raw_data):
    try:
        # Parse input JSON data
        data = json.loads(raw_data)

        # Convert to DataFrame
        input_df = pd.DataFrame(data["data"])

        # Ensure categorical columns are strings
        cat_features = ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status"]
        input_df[cat_features] = input_df[cat_features].astype(str)  # ✅ Convert categorical features to strings

        # Make prediction
        prediction = model.predict(input_df)

        return {"predictions": prediction.tolist()}

    except Exception as e:
        return {"error": str(e)}
