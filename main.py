from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn

# Load the dataset and model
df = pd.read_csv("Final.csv")  # Ensure this file exists in the working directory
model = joblib.load("trained_model.pkl")

app = FastAPI()

class InputData(BaseModel):
    soil_moisture: float
    N: float
    P: float
    K: float
    soil_pH: float
    land_size: float
    last_crop: str
    crop: str

@app.post("/predict")
def predict(data: InputData):
    # Compute target encoding for last_crop and crop
    last_crop_encoded = df[df["last_crop"] == data.last_crop]["expected_yield"].mean()
    crop_encoded = df[df["crop"] == data.crop]["expected_yield"].mean()

    # Handle cases where encoding results in NaN (unknown category)
    if pd.isna(last_crop_encoded) or pd.isna(crop_encoded):
        raise HTTPException(status_code=400, detail="Unknown last_crop or crop value")

    # Create input feature array
    features = [[
        data.soil_moisture, data.N, data.P, data.K, data.soil_pH, 
        data.land_size, last_crop_encoded, crop_encoded
    ]]

    # Get predictions
    prediction = model.predict(features)[0]
    
    response = {
        "optimal_N": prediction[0],
        "optimal_P": prediction[1],
        "optimal_K": prediction[2],
        "optimal_pH_min": prediction[3],
        "optimal_pH_max": prediction[4],
        "expected_yield": prediction[5]
    }

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
