from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from xgboost import XGBClassifier

# Define the FastAPI app
app = FastAPI()

# Load the model from the .joblib file
model_filename = 'gradientOG.joblib'
model = joblib.load(model_filename)

# Define the data model for the input
class SoilFertilityFeatures(BaseModel):
    N: float
    P: float
    K: float
    pH: float
    EC: float
    OC: float
    S: float
    Zn: float
    Fe: float
    Cu: float
    Mn: float
    B: float

# Define the prediction endpoint
@app.post('/predict/')
def predict(features: SoilFertilityFeatures):
    # Convert the input data to a numpy array
    input_data = np.array([[
        features.N, features.P, features.K, features.pH, features.EC,
        features.OC, features.S, features.Zn, features.Fe, features.Cu,
        features.Mn, features.B
    ]])
    
    # Make a prediction using the loaded model
    prediction = model.predict(input_data)[0]
    
    # Map the prediction to fertility levels
    fertility_mapping = {0: 'Low Fertility', 1: 'Medium Fertility', 2: 'High Fertility'}
    fertility_level = fertility_mapping.get(prediction, "Unknown")

    # Return the prediction result
    return {"prediction": fertility_level}

# Root endpoint (optional)
@app.get("/")
def read_root():
    return {"message": "Welcome to the Soil Fertility Prediction API"}
