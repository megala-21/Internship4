# Real-Time ML Inference REST API

A production-style REST microservice that serves a trained **champion model** with FastAPI, packaged in Docker.

The model predicts whether a breast-tumour sample is `malignant` or `benign` from 30 numeric features (scikit-learn's Breast Cancer Wisconsin dataset).

## Architecture

```
 train.py                      app.py (FastAPI)                 Docker image
 ┌──────────────────┐          ┌─────────────────────┐          ┌───────────────┐
 │ load dataset     │          │ lifespan: load      │          │ python:3.11   │
 │ 3 candidates     │  saves   │  model + metadata   │  copied  │ pinned deps   │
 │ 5-fold CV AUC    ├─────────►│ POST /predict       ├─────────►│ non-root user │
 │ pick champion    │ model/   │ GET  /health        │  into    │ uvicorn :8000 │
 │ held-out eval    │          │ GET  /model-info    │          │ HEALTHCHECK   │
 └──────────────────┘          └─────────────────────┘          └───────────────┘
```

1. **Training (`train.py`)** compares Logistic Regression, Random Forest and Gradient Boosting using 5-fold cross-validated ROC-AUC on the training split. The best one is the *champion*, refit and evaluated once on a held-out test set.
2. **Artifacts (`model/`)** hold `champion.joblib` and `metadata.json` (feature names, class names, metrics).
3. **Serving (`app.py`)** loads the model once at startup, validates input with Pydantic, and returns class probabilities.
4. **Packaging (`Dockerfile`)** installs pinned dependencies, runs as a non-root user, and exposes a health check.

Current champion: **Logistic Regression** (CV ROC-AUC 0.9935, held-out ROC-AUC 0.9954).

## Project layout

```
├── app.py                 FastAPI service
├── train.py               model selection + training
├── test_app.py            unit tests
├── model/                 champion.joblib, metadata.json
├── requirements.txt       pinned runtime deps
├── requirements-dev.txt   adds pytest, httpx
└── Dockerfile
```

## Quick start

### Local

```bash
pip install -r requirements-dev.txt
python train.py                       # (re)creates model/
uvicorn app:app --reload              # http://localhost:8000
```

### Docker

```bash
python train.py                       # model/ must exist before building
docker build -t ml-inference-api .
docker run -p 8000:8000 ml-inference-api
```

Interactive docs (Swagger UI): http://localhost:8000/docs

## API reference

### `POST /predict`

Request body:

```json
{ "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
               0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
               0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
               0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189] }
```

Response `200 OK`:

```json
{
  "prediction": 0,
  "label": "malignant",
  "probabilities": { "malignant": 0.9999, "benign": 0.0001 }
}
```

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}'
```

| Status | Meaning |
|--------|---------|
| 200 | Prediction returned |
| 422 | Missing field, non-numeric values, or wrong number of features (expects 30) |
| 405 | Wrong HTTP method |

### `GET /health`
Returns `{"status": "ok", "model_loaded": true}`. Used by the Docker `HEALTHCHECK`.

### `GET /model-info`
Returns champion name, metrics, feature names (in required order) and class names.

## Testing

```bash
pytest -q
```

The tests cover health and model-info endpoints, response schema, probabilities summing to 1, a known sample, and 422 / 405 / 404 error codes.

## Design notes

- The model loads once in the FastAPI lifespan handler, not per request, so latency stays low.
- Dependencies are pinned for reproducible builds; the container runs as a non-root user.
- To swap in a different champion, retrain with `train.py` (or replace `model/champion.joblib` and update `metadata.json`) and rebuild the image.
