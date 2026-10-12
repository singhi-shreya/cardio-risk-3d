"""Shared helpers: paths, dataset loading, target construction, form schema, preprocessing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

VESSELS = ["LAD", "LCX", "RCA"]
TARGETS = ["CAD"] + VESSELS

# Columns that must NEVER be model inputs (target leakage), compared case-insensitively.
LEAKAGE_COLS = {"lad", "lcx", "rca", "cath", "cad"}

# String labels that mean "positive" (CAD present / vessel stenosed).
POSITIVE_TOKENS = {"cad", "y", "yes", "1", "1.0", "true", "stenotic", "stenosis", "abnormal", "positive"}


# --------------------------------------------------------------------------- data
def find_dataset() -> Path:
    """Return the first .csv/.xlsx in data/ (files starting with 'demo' are only a fallback)."""
    files = sorted(p for p in DATA_DIR.glob("*") if p.suffix.lower() in {".csv", ".xlsx", ".xls"})
    if not files:
        raise FileNotFoundError(
            f"No dataset found in {DATA_DIR}. Download the Z-Alizadeh Sani extension dataset "
            "(UCI / Kaggle) and put the .csv or .xlsx file there, or run: python data/make_demo_data.py"
        )
    real = [p for p in files if not p.stem.lower().startswith("demo")]
    return (real or files)[0]


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or find_dataset()
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.lower().str.startswith("unnamed")]
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            col = df[c].astype(str).str.strip()
            col = col.where(~col.str.lower().isin({"nan", "none", ""}), np.nan)
            df[c] = col.astype(object)
    return df


def to_binary(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0) > 0).astype(int)
    return s.astype(str).str.strip().str.lower().isin(POSITIVE_TOKENS).astype(int)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features X (leakage columns removed) and binary targets Y with columns CAD, LAD, LCX, RCA."""
    lower = {c.lower(): c for c in df.columns}
    missing = [v for v in ("lad", "lcx", "rca") if v not in lower]
    if missing:
        raise ValueError(f"Dataset is missing vessel label column(s): {missing}. Columns found: {list(df.columns)}")
    cad_col = lower.get("cad") or lower.get("cath")
    if cad_col is None:
        raise ValueError("Dataset needs a 'Cath' (or 'CAD') column holding the overall CAD label.")
    Y = pd.DataFrame(
        {
            "CAD": to_binary(df[cad_col]),
            "LAD": to_binary(df[lower["lad"]]),
            "LCX": to_binary(df[lower["lcx"]]),
            "RCA": to_binary(df[lower["rca"]]),
        },
        index=df.index,
    )
    X = df.drop(columns=[c for c in df.columns if c.lower() in LEAKAGE_COLS])
    return X, Y


# --------------------------------------------------------------------------- form schema
def build_schema(X: pd.DataFrame) -> list[dict]:
    """Describe every input feature so the web form is generated automatically (new column -> new field)."""
    fields: list[dict] = []
    for c in X.columns:
        s = X[c].dropna()
        if s.empty:
            continue
        if pd.api.types.is_numeric_dtype(X[c]):
            uniq = sorted(s.unique().tolist())
            if len(uniq) <= 6 and all(float(u).is_integer() for u in uniq):
                fields.append({"name": c, "type": "discrete", "options": [int(u) for u in uniq],
                               "default": int(s.mode().iloc[0])})
            else:
                fields.append({"name": c, "type": "number", "min": float(s.min()), "max": float(s.max()),
                               "default": round(float(s.median()), 2)})
        else:
            s = s.astype(str)
            fields.append({"name": c, "type": "category", "options": sorted(s.unique().tolist()),
                           "default": str(s.mode().iloc[0])})
    return fields


# --------------------------------------------------------------------------- preprocessing
def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat,
            ),
        ],
        verbose_feature_names_out=False,
    )
