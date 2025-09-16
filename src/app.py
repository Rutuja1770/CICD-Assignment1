from fastapi import FastAPI
from src.predict import predict
from src.train import train_model
import os

app = FastAPI()

# Train model if not already trained
if not os.path.exists("model.pkl"):
    train_model()

@app.get("/")
def home():
    return {"message": "ML API is running!"}

@app.get("/predict/{value}")
def get_prediction(value: int):
    result = predict(value)
    return {"input": value, "prediction": result}
