from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200

def test_predict_endpoint():
    payload = {
      "age": 52, "sex": 1, "cp": 0, "trestbps": 130, "chol": 250, "fbs": 0,
      "restecg": 1, "thalach": 140, "exang": 0, "oldpeak": 1.2,
      "slope": 2, "ca": 0, "thal": 3
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "risk" in data and "confidence" in data
