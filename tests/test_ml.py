import os
import joblib
from src.train import train_model
from src.predict import predict

def test_training_creates_model_file():
    model = train_model()
    assert os.path.exists("model.pkl")

def test_prediction_is_correct():
    train_model()
    result = predict(6)
    assert round(result) == 12
