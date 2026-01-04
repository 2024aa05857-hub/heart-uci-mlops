import joblib
import pandas as pd
from src.config import Paths

def load_model():
    return joblib.load(Paths().model_path)

def predict_one(features: dict) -> dict:
    model = load_model()
    x = pd.DataFrame([features])
    proba = float(model.predict_proba(x)[:, 1][0])
    pred = int(proba >= 0.5)
    return {"risk": pred, "confidence": round(proba, 4)}

if __name__ == "__main__":
    sample = {
        "age": 52, "sex": 1, "cp": 0, "trestbps": 130, "chol": 250, "fbs": 0,
        "restecg": 1, "thalach": 140, "exang": 0, "oldpeak": 1.2,
        "slope": 2, "ca": 0, "thal": 3
    }
    print(predict_one(sample))
