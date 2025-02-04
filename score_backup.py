try:
    import joblib
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "joblib"])
    import joblib

import json
import numpy as np
import pandas as pd
import joblib
from azureml.core.model import Model
from catboost import CatBoostClassifier

def init():
    global model
    model_path = Model.get_model_path("catboost-thrombosis-predictor")
    model = CatBoostClassifier()
    model.load_model(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        df = pd.DataFrame(data)
        
        # Convert categorical features to int if necessary
        categorical_features = ["gender", "work_type", "Residence_type", "smoking_status"]
        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str).astype("category")

        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}
