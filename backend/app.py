"""FastAPI backend: serves predictions + SHAP explanations and the static 3D frontend.

Run (from the project root):
    uvicorn backend.app:app --reload
then open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .common import MODEL_DIR, ROOT, TARGETS, VESSELS

DISCLAIMER = ("For decision support and educational purposes only. Not a substitute for formal diagnostic "
              "imaging (e.g. coronary angiography or CT angiography) or clinical judgement.")

# --------------------------------------------------------------------------- load artefacts once
if not (MODEL_DIR / "CAD.joblib").exists():
    raise RuntimeError("No trained models found. Run `python -m backend.train` first.")

BUNDLES: dict[str, dict] = {}
for _t in TARGETS:
    _b = joblib.load(MODEL_DIR / f"{_t}.joblib")
    _b["explainer"] = shap.TreeExplainer(_b["pipeline"].named_steps["clf"])
    BUNDLES[_t] = _b

SCHEMA: list[dict] = json.loads((MODEL_DIR / "schema.json").read_text())
METRICS: dict = json.loads((MODEL_DIR / "metrics.json").read_text())
SAMPLES: list = json.loads((MODEL_DIR / "samples.json").read_text())
FEATURE_COLS: list[str] = BUNDLES["CAD"]["feature_cols"]
FIELD_BY_NAME = {f["name"]: f for f in SCHEMA}

app = FastAPI(title="Cardiovascular Risk 3D", version="1.0")


# --------------------------------------------------------------------------- helpers
class PredictIn(BaseModel):
    features: dict[str, Any]


def build_frame(features: dict[str, Any]) -> pd.DataFrame:
    """One-row DataFrame in training column order. Blank / unknown values become NaN and are imputed."""
    row: dict[str, Any] = {}
    for col in FEATURE_COLS:
        raw = features.get(col)
        field = FIELD_BY_NAME.get(col)
        if raw is None or raw == "" or field is None:
            row[col] = np.nan
        elif field["type"] == "category":
            row[col] = str(raw)
        else:
            try:
                row[col] = float(raw)
            except (TypeError, ValueError):
                raise HTTPException(status_code=422, detail=f"Invalid numeric value for '{col}': {raw!r}")
    return pd.DataFrame([row], columns=FEATURE_COLS)


def band(p: float) -> str:
    return "low" if p < 0.33 else "moderate" if p < 0.66 else "high"


def explain(target: str, X: pd.DataFrame) -> tuple[float, dict]:
    bundle = BUNDLES[target]
    prep = bundle["pipeline"].named_steps["prep"]
    clf = bundle["pipeline"].named_steps["clf"]

    Xt = np.asarray(prep.transform(X), dtype=float)
    names = list(prep.get_feature_names_out())
    prob = float(clf.predict_proba(Xt)[0, 1])

    sv = bundle["explainer"].shap_values(Xt, check_additivity=False)
    if isinstance(sv, list):          # older SHAP: [class0, class1]
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # newer SHAP: (samples, features, classes)
        sv = sv[:, :, -1]
    vals = sv[0]
    base = float(np.ravel(bundle["explainer"].expected_value)[-1])

    total = float(np.abs(vals).sum()) or 1.0
    order = np.argsort(-np.abs(vals))
    contribs = [
        {
            "feature": names[i],
            "value": float(Xt[0, i]),
            "shap": round(float(vals[i]), 5),
            "share": round(float(abs(vals[i]) / total * 100), 2),
            "direction": "increases" if vals[i] > 0 else "decreases",
        }
        for i in order
    ]
    return prob, {"base_value": round(base, 4), "contributions": contribs}


# --------------------------------------------------------------------------- API
@app.get("/api/health")
def health():
    return {"status": "ok", "targets": TARGETS}


@app.get("/api/schema")
def schema():
    return {"features": SCHEMA, "disclaimer": DISCLAIMER}


@app.get("/api/metrics")
def metrics():
    return METRICS


@app.get("/api/samples")
def samples():
    return SAMPLES


@app.post("/api/predict")
def predict(req: PredictIn):
    X = build_frame(req.features)
    probs: dict[str, float] = {}
    explanations: dict[str, dict] = {}
    for t in TARGETS:
        probs[t], explanations[t] = explain(t, X)
    return {
        "cad": {"probability": round(probs["CAD"], 4), "label": int(probs["CAD"] >= 0.5), "band": band(probs["CAD"])},
        "vessels": {
            v: {"probability": round(probs[v], 4), "label": int(probs[v] >= 0.5), "band": band(probs[v])}
            for v in VESSELS
        },
        "explanations": explanations,
        "models": {t: BUNDLES[t]["model_name"] for t in TARGETS},
        "disclaimer": DISCLAIMER,
    }


# Frontend (must be mounted last so /api/* routes win)
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
