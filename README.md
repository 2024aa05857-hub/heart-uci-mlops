# Heart Disease Risk Prediction – End-to-End MLOps Assignment

Implements the full pipeline required in the assignment:
- Data acquisition + EDA
- Two models + cross-validation metrics
- MLflow experiment tracking
- Reproducible preprocessing pipeline
- CI/CD with GitHub Actions + Pytest
- FastAPI model serving + Docker
- Kubernetes deployment manifests
- Monitoring via Prometheus metrics + request logging

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
python -m src.train
```

## MLflow
```bash
mlflow ui --backend-store-uri ./mlruns
```

## Run API
```bash
uvicorn api.main:app --reload
```

## Predict (sample)
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":0,"trestbps":130,"chol":250,"fbs":0,"restecg":1,"thalach":140,"exang":0,"oldpeak":1.2,"slope":2,"ca":0,"thal":3}'
```

## Tests
```bash
pytest -q
```

## Docker
```bash
docker build -t heart-api:1 .
docker run -p 8000:8000 heart-api:1
```

## Kubernetes
```bash
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
kubectl get pods,svc
```

## Monitoring
Prometheus scrapes `/metrics`. See `monitoring/prometheus.yml`.
