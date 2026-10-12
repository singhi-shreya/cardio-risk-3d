"""Train the CAD + LAD / LCX / RCA classifiers.

Usage (from the project root):
    python -m backend.train

For each target we
  1. split 80/20 (stratified) into train / hold-out test,
  2. compare Random Forest vs Gradient Boosting with 5-fold CV ROC-AUC on the training part,
  3. report hold-out metrics AND out-of-fold (5-fold CV over all data) metrics,
  4. refit the winner on ALL data and save it to models/<TARGET>.joblib.
Leakage columns (LAD, LCX, RCA, Cath) are removed from the inputs in common.split_xy().
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .common import (LEAKAGE_COLS, MODEL_DIR, TARGETS, build_schema, find_dataset, load_raw, make_preprocessor,
                     split_xy)

SEED = 42
warnings.filterwarnings("ignore")


def build_pipeline(X: pd.DataFrame, name: str) -> Pipeline:
    if name == "random_forest":
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced",
                                     random_state=SEED, n_jobs=-1)
    elif name == "gradient_boosting":
        clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8,
                                         random_state=SEED)
    else:
        raise ValueError(name)
    return Pipeline([("prep", make_preprocessor(X)), ("clf", clf)])


def evaluate(y_true, prob, thr: float = 0.5) -> dict:
    pred = (np.asarray(prob) >= thr).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, prob)
    except ValueError:
        out["roc_auc"] = float("nan")
    return {k: round(float(v), 4) for k, v in out.items()}


def main() -> None:
    path = find_dataset()
    df = load_raw(path)
    X, Y = split_xy(df)
    MODEL_DIR.mkdir(exist_ok=True)
    print(f"Dataset: {path.name}  rows={len(df)}  input features={X.shape[1]}")
    print(f"Removed leakage columns: {sorted(c for c in df.columns if c.lower() in LEAKAGE_COLS)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    report = {
        "dataset": {"file": path.name, "rows": int(len(df)), "n_features": int(X.shape[1]),
                    "removed_leakage_columns": sorted(c for c in df.columns if c.lower() in LEAKAGE_COLS),
                    "positives": {t: int(Y[t].sum()) for t in TARGETS}},
        "targets": {},
    }

    for target in TARGETS:
        y = Y[target]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

        cv_auc = {}
        for name in ("random_forest", "gradient_boosting"):
            cv_auc[name] = float(cross_val_score(build_pipeline(X, name), X_tr, y_tr, cv=cv,
                                                 scoring="roc_auc").mean())
        best = max(cv_auc, key=cv_auc.get)

        holdout_model = build_pipeline(X, best).fit(X_tr, y_tr)
        holdout = evaluate(y_te, holdout_model.predict_proba(X_te)[:, 1])
        oof = evaluate(y, cross_val_predict(build_pipeline(X, best), X, y, cv=cv, method="predict_proba")[:, 1])

        final = build_pipeline(X, best).fit(X, y)  # deployed model uses all available data
        if target == "CAD":
            cad_final = final
        joblib.dump({"pipeline": final, "feature_cols": list(X.columns), "model_name": best, "target": target},
                    MODEL_DIR / f"{target}.joblib", compress=3)

        report["targets"][target] = {"model": best, "cv_auc_by_model": {k: round(v, 4) for k, v in cv_auc.items()},
                                     "holdout": holdout, "cv_oof": oof,
                                     "n_train": int(len(X_tr)), "n_test": int(len(X_te))}
        print(f"\n[{target}] best={best}  CV-AUC {({k: round(v, 3) for k, v in cv_auc.items()})}")
        print(f"   hold-out : {holdout}")
        print(f"   5-fold OOF: {oof}")

    # Reference table for "patients like this": every patient, preprocessed and standardised
    Xt = np.asarray(cad_final.named_steps["prep"].transform(X), dtype=float)
    mean, std = Xt.mean(axis=0), Xt.std(axis=0)
    std[std == 0] = 1.0
    np.savez_compressed(MODEL_DIR / "reference.npz", Z=(Xt - mean) / std, mean=mean, std=std,
                        Y=Y[TARGETS].to_numpy())

    # Form schema + a few real patients for the "load sample patient" button in the UI
    (MODEL_DIR / "schema.json").write_text(json.dumps(build_schema(X), indent=2))
    (MODEL_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    idx = df.sample(min(30, len(df)), random_state=SEED).index
    samples = []
    for i in idx:
        feats = {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in X.loc[i].items()}
        samples.append({"features": feats, "truth": {t: int(Y.loc[i, t]) for t in TARGETS}})
    (MODEL_DIR / "samples.json").write_text(json.dumps(samples))
    print(f"\nSaved models, schema.json, metrics.json, samples.json, reference.npz to {MODEL_DIR}")


if __name__ == "__main__":
    main()
