# CoronaryMap 3D – Cardiovascular Risk Visualization & Prediction

Multimodal AI Hackathon 2026 · Track A

A web app that predicts **overall coronary artery disease (CAD)** and **stenosis of the LAD, LCX and RCA**
from clinical, ECG, laboratory and echocardiographic features, then paints the predicted risk onto an
interactive 3D heart. A dashboard explains every prediction with SHAP values.

> ⚠ **For decision support / educational purposes only.** Not a medical device and not a substitute for
> formal diagnostic imaging (coronary angiography, CT angiography) or clinical judgement.

## Features

| Requirement | Where |
|---|---|
| Classifiers for CAD, LAD, LCX, RCA | `backend/train.py` (Random Forest vs Gradient Boosting, picked by CV ROC-AUC) |
| Leakage prevention | `LAD, LCX, RCA, Cath` are dropped from the inputs in `backend/common.py → split_xy` |
| Metrics | accuracy, precision, recall, F1, ROC-AUC – hold-out **and** 5-fold out-of-fold (`models/metrics.json`, shown in the UI) |
| Interactive 3D heart (Three.js) | `frontend/app.js` – rotate / zoom / pan, click an artery or the heart |
| Arteries recoloured from predicted probabilities | green → yellow → red, smooth transitions, high-risk arteries pulse |
| SHAP explanation + physiological breakdown | right-hand dashboard (`/api/predict` returns SHAP for each target) |
| Visible disclaimers | top banner, in-viewer badge, footer |
| Works without a dedicated GPU | simple geometry, pixel ratio capped at 1.5 |
| Extensible | the input form is generated from `models/schema.json`; add a column → retrain → new field appears |
| Beginner-friendly UI | plain-language labels with the clinical name beside them, "?" help on every field, Yes/No toggles, sliders, 10 key inputs up front (`backend/field_info.py`) |
| Simple / Clinician views | Simple shows the heart, one headline and a plain-English story; Clinician shows technical names and opens the full details drawer |
| "What if...?" | slide one input and watch the model's predictions and the arteries change live |
| "Patients like this" | the 5 most similar dataset patients and what their angiography showed |
| Honest confidence | states how many of the 55 details were provided |

## Project layout

```
cardio-risk-3d/
├─ backend/
│  ├─ common.py        dataset loading, targets, schema, preprocessing
│  ├─ field_info.py    friendly labels, groups, help text, typical ranges (edit freely, no retraining)
│  ├─ train.py         training + validation + saving models
│  └─ app.py           FastAPI: /api/predict, /api/schema, /api/metrics, /api/samples (+ serves frontend)
├─ frontend/
│  ├─ index.html  style.css  app.js     Three.js viewer + dashboard
│  └─ models/heart.glb                  (optional) your own anatomical mesh
├─ data/               put the dataset here (make_demo_data.py makes a SYNTHETIC test file)
├─ models/             trained pipelines (*.joblib), metrics.json, schema.json, samples.json, reference.npz
└─ requirements.txt
```

## Setup

Prerequisites: **Python 3.10–3.12**, a modern browser, internet access (Three.js is loaded from a CDN).

```bash
# 1. create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell / cmd)
source .venv/bin/activate       # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt
```

### Dataset

Download the **Extension of Z-Alizadeh Sani** dataset (UCI Machine Learning Repository or Kaggle) and place the
`.csv` / `.xlsx` file in `data/`. It must contain the columns `LAD`, `LCX`, `RCA` and `Cath`.

No dataset yet? `python data/make_demo_data.py` creates a **synthetic** file so you can test the app. Never report
results from it.

### Train

```bash
python -m backend.train
```

Writes `models/*.joblib`, `schema.json`, `metrics.json` and `samples.json` and prints the metrics.

### Run

```bash
uvicorn backend.app:app --reload
```

Open <http://127.0.0.1:8000>.

## Using the app

1. Pick **Meet Alex / Meet Priya**, a **Random real patient** (shown with its angiography result for comparison), or
   edit the fields on the left. Tap **?** next to any field for a plain-language explanation. Blank fields are imputed
   with the training median / mode, and the confidence line tells you how many details were provided.
2. Predictions update about 350 ms after each change; the arteries recolour and the heart beats at the entered pulse rate.
3. Rotate with the left mouse button, zoom with the wheel, pan with the right button. Click **LAD / LCX / RCA** (on the heart
   or the chips) or the heart itself (overall CAD) to read why in plain English and see similar patients.
4. Use **What if...?** to change one value and watch the predictions move.
5. Click **See the details** (or switch to **Clinician**) for the SHAP chart, measurement table and model scores.

## Method

* **Preprocessing:** median imputation for numeric features; most-frequent imputation + one-hot encoding for
  categorical ones. All of it lives inside the sklearn `Pipeline`, so training and serving are identical.
* **Targets:** `Cath` (Cad / Normal) → overall CAD; `LAD`, `LCX`, `RCA` → binary vessel stenosis (≥ 50 %).
* **Validation:** stratified 80/20 hold-out; model choice by 5-fold CV ROC-AUC on the training split only; separate
  5-fold out-of-fold metrics over all rows. The deployed model is refit on all data.
* **Explainability:** `shap.TreeExplainer` per target. For each prediction the API returns every feature's SHAP
  value and its share of the total absolute effect.
* **3D mapping:** each coronary artery is a separate mesh named after its target. `applyResultTo3D()` maps probability
  → hue (green 0 % → red 100 %). Hit-test meshes make thin arteries easy to click.

## Results

Run `python -m backend.train` and paste your numbers here (hold-out):

| Target | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| CAD | | | | | | |
| LAD | | | | | | |
| LCX | | | | | | |
| RCA | | | | | | |

## Using a real anatomical mesh (optional)

Download an open-source heart model (e.g. BodyParts3D, NIH 3D, Sketchfab with a permissive licence), convert it to
`.glb`, and save it as `frontend/models/heart.glb`. It replaces the procedural heart. If the arteries don't line up,
adjust the point lists in `ARTERIES` (top of `frontend/app.js`).

## Extending

* **New clinical feature:** add the column to the dataset, run `python -m backend.train` – the form, model and SHAP
  chart pick it up automatically.
* **New model:** add a branch in `build_pipeline()` in `backend/train.py` (tree-based models work with the SHAP
  code as is).
* **New anatomical structure:** add an entry to `ARTERIES` and a target name in `VESSELS` / `TARGETS`.

## Limitations

* Only 303 patients from a single centre; probabilities are not calibrated for other populations.
* Vessel models are independent, so a vessel probability can exceed the overall CAD probability.
* The dataset has no spatial lesion information; colours show vessel-level probabilities, not lesion locations.
* The default heart is stylised. Region-with-RWMA is used only as a numeric input.

## Credits

Dataset: Z-Alizadeh Sani et al., UCI Machine Learning Repository (CC BY 4.0). 3D rendering: Three.js. Explanations: SHAP.
