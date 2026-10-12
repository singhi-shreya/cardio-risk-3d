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
from .field_info import GROUPS, enrich

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

SCHEMA: list[dict] = enrich(json.loads((MODEL_DIR / "schema.json").read_text()))
METRICS: dict = json.loads((MODEL_DIR / "metrics.json").read_text())
SAMPLES: list = json.loads((MODEL_DIR / "samples.json").read_text())
FEATURE_COLS: list[str] = BUNDLES["CAD"]["feature_cols"]
FIELD_BY_NAME = {f["name"]: f for f in SCHEMA}

# Reference patients for the "patients like this" feature (created by `python -m backend.train`)
REF = np.load(MODEL_DIR / "reference.npz") if (MODEL_DIR / "reference.npz").exists() else None
_CAD_PREP = BUNDLES["CAD"]["pipeline"].named_steps["prep"]


def _owner(col: str) -> str | None:
    """Which original input column a (possibly one-hot) model column came from."""
    best = None
    for c in FEATURE_COLS:
        if (col == c or col.startswith(c + "_")) and (best is None or len(c) > len(best)):
            best = c
    return best


OWNERS = [_owner(n) for n in _CAD_PREP.get_feature_names_out()]

app = FastAPI(title="Cardiovascular Risk 3D", version="2.0")


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


def similar_patients(X: pd.DataFrame, provided: set[str], k: int = 5) -> dict | None:
    """The k most similar dataset patients, compared only on the details the user actually provided."""
    if REF is None:
        return None
    xt = np.asarray(_CAD_PREP.transform(X), dtype=float)[0]
    z = (xt - REF["mean"]) / REF["std"]
    mask = np.array([o in provided for o in OWNERS])
    if not mask.any():
        mask[:] = True
    dist = np.sqrt(((REF["Z"][:, mask] - z[mask]) ** 2).sum(axis=1))
    idx = np.argsort(dist)[:k]
    Y = REF["Y"][idx]
    return {"k": int(len(idx)), "labels": {t: [int(v) for v in Y[:, j]] for j, t in enumerate(TARGETS)}}


# --------------------------------------------------------------------------- API
@app.get("/api/health")
def health():
    return {"status": "ok", "targets": TARGETS}


@app.get("/api/schema")
def schema():
    return {"features": SCHEMA, "groups": GROUPS, "disclaimer": DISCLAIMER}


@app.get("/api/metrics")
def metrics():
    return METRICS


@app.get("/api/samples")
def samples():
    return SAMPLES


@app.post("/api/predict")
def predict(req: PredictIn):
    X = build_frame(req.features)
    provided = {k for k, v in req.features.items() if k in FIELD_BY_NAME and v not in (None, "")}
    probs: dict[str, float] = {}
    explanations: dict[str, dict] = {}
    for t in TARGETS:
        probs[t], explanations[t] = explain(t, X)
    share = len(provided) / max(len(SCHEMA), 1)
    return {
        "cad": {"probability": round(probs["CAD"], 4), "label": int(probs["CAD"] >= 0.5), "band": band(probs["CAD"])},
        "vessels": {
            v: {"probability": round(probs[v], 4), "label": int(probs[v] >= 0.5), "band": band(probs[v])}
            for v in VESSELS
        },
        "explanations": explanations,
        "similar": similar_patients(X, provided),
        "coverage": {"provided": len(provided), "total": len(SCHEMA),
                     "level": "limited" if share < 0.25 else "moderate" if share < 0.6 else "good"},
        "models": {t: BUNDLES[t]["model_name"] for t in TARGETS},
        "disclaimer": DISCLAIMER,
    }


# Frontend (must be mounted last so /api/* routes win)
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
