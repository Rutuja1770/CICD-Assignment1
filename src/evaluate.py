import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os

REPORT_PATH = "reports/metrics_report.txt"

def evaluate_model():
    # Load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # Load trained model
    if not os.path.exists("model.pkl"):
        raise FileNotFoundError("Trained model not found. Run train.py first.")

    model = joblib.load("model.pkl")

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    # Save report
    os.makedirs("reports", exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(f"Accuracy: {acc:.2f}\n")
        f.write(f"Precision: {prec:.2f}\n")
        f.write(f"Recall: {rec:.2f}\n")

    print(f"✅ Report saved at {REPORT_PATH}")
    print(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}")

    # Fail if accuracy < 0.80
    if acc < 0.80:
        raise ValueError(f"❌ Accuracy below threshold: {acc:.2f}")

if __name__ == "__main__":
    evaluate_model()
