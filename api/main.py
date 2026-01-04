import time
import logging
import joblib
import pandas as pd
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from api.schemas import Patient
from src.config import Paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart-api")

REQUESTS = Counter("api_requests_total", "Total API requests", ["endpoint"])
LATENCY = Histogram("api_latency_seconds", "API latency in seconds", ["endpoint"])

app = FastAPI(title="Heart Disease Risk API", version="1.0.0")

paths = Paths()
model = None

def get_model():
    global model
    if model is None:
        model = joblib.load(paths.model_path)
    return model

@app.on_event("startup")
def _load():
    global model
    model = joblib.load(paths.model_path)
    logger.info("Loaded model from %s", paths.model_path)

@app.get("/health")
def health():
    REQUESTS.labels(endpoint="/health").inc()
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.post("/predict")
def predict(p: Patient):
    start = time.time()
    REQUESTS.labels(endpoint="/predict").inc()

    x = pd.DataFrame([p.model_dump()])
    proba = float(model.predict_proba(x)[:, 1][0])
    pred = int(proba >= 0.5)

    logger.info("prediction=%s proba=%.4f input=%s", pred, proba, p.model_dump())
    LATENCY.labels(endpoint="/predict").observe(time.time() - start)

    return {"risk": pred, "confidence": round(proba, 4)}
