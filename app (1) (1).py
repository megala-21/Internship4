"""FastAPI microservice serving the champion model."""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent / "model"))

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_DIR / "champion.joblib")
    state["meta"] = json.loads((MODEL_DIR / "metadata.json").read_text())
    yield
    state.clear()


app = FastAPI(
    title="Real-Time ML Inference API",
    description="Serves class probabilities from the champion model.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ..., description="Feature vector, in the order given by GET /model-info"
    )


class PredictResponse(BaseModel):
    prediction: int
    label: str
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    return state["meta"]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    meta = state["meta"]
    expected = len(meta["feature_names"])
    if len(req.features) != expected:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected} features, got {len(req.features)}",
        )
    x = np.array(req.features, dtype=float).reshape(1, -1)
    if not np.isfinite(x).all():
        raise HTTPException(status_code=422, detail="Features must be finite numbers")
    proba = state["model"].predict_proba(x)[0]
    idx = int(np.argmax(proba))
    names = meta["class_names"]
    return PredictResponse(
        prediction=idx,
        label=names[idx],
        probabilities={n: float(p) for n, p in zip(names, proba)},
    )
