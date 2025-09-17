# src/evaluate.py
import json
import sys
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def evaluate_model():
    # Load dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load trained model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
    }

    # Save metrics report
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("📊 Evaluation Metrics:", metrics)

    # Fail pipeline if accuracy < 80%
    if metrics["accuracy"] < 0.8:
        print("❌ Model accuracy below threshold (80%).")
        sys.exit(1)

if __name__ == "__main__":
    evaluate_model()





