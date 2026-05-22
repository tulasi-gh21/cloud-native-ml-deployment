from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

class InputData(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "ML API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.features])

    return {
        "prediction": int(prediction[0])
    }