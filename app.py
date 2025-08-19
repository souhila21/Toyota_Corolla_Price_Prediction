from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib


# Load pipeline
model = joblib.load("car_price_pipeline.pkl")

# Define input schema
class CarFeatures(BaseModel):
    Model: str
    Age_08_04: float
    KM: float
    Fuel_Type: str
    HP: float
    Met_Color: int
    Color: str
    Automatic: int
    CC: float
    Doors: int
    Gears: int
    Quarterly_Tax: float
    Weight: float
    Mfr_Guarantee: int
    BOVAG_Guarantee: int
    Guarantee_Period: int
    ABS: int
    Airbag_1: int
    Airbag_2: int
    Airco: int
    Automatic_airco: int
    Boardcomputer: int
    CD_Player: int
    Central_Lock: int
    Powered_Windows: int
    Power_Steering: int
    Radio: int
    Mistlamps: int
    Sport_Model: int
    Backseat_Divider: int
    Metallic_Rim: int
    Radio_cassette: int
    Parking_Assistant: int
    Tow_Bar: int


app = FastAPI()

@app.post("/predict_batch")
def predict_batch(data: List[CarFeatures]):
    df = pd.DataFrame([item.dict() for item in data])
    preds = model.predict(df)
    return {"predicted_prices": preds.tolist()}

@app.post("/predict")
def predict_price(features: CarFeatures):
    try:
        input_df = pd.DataFrame([features.dict()])
        prediction = model.predict(input_df)
        return {"predicted_price": float(prediction[0])}
    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))