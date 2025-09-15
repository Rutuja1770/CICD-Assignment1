import joblib
import numpy as np

def predict(value):
    model = joblib.load("model.pkl")
    return model.predict(np.array([[value]]))[0]

if __name__ == "__main__":
    print(predict(6))  # should print ~12
