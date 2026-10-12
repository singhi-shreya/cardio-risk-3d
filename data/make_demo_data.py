"""Create a SYNTHETIC stand-in dataset (same column names as Z-Alizadeh Sani) so you can test the
pipeline before downloading the real data. Never use this for your final results.

Run:  python data/make_demo_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 303
sig = lambda x: 1 / (1 + np.exp(-x))
yn = lambda p: np.where(rng.random(n) < p, "Y", "N")

age = rng.normal(58, 10, n).clip(30, 86).round()
male = rng.random(n) < 0.55
weight = np.where(male, rng.normal(82, 11, n), rng.normal(72, 11, n)).round()
length = np.where(male, rng.normal(172, 7, n), rng.normal(159, 6, n)).round()
bmi = (weight / (length / 100) ** 2).round(1)
dm = (rng.random(n) < sig(-1.2 + 0.03 * (age - 58))).astype(int)
htn = (rng.random(n) < sig(-0.3 + 0.04 * (age - 58))).astype(int)
smoker = (rng.random(n) < np.where(male, 0.3, 0.05)).astype(int)
bp = (rng.normal(128, 16, n) + 8 * htn).clip(90, 200).round()
pr = rng.normal(75, 11, n).clip(50, 110).round()
fbs = (rng.normal(100, 20, n) + 45 * dm).clip(65, 400).round()
ldl = rng.normal(105, 33, n).clip(40, 250).round()
hdl = rng.normal(42, 10, n).clip(20, 90).round()
tg = rng.normal(150, 70, n).clip(40, 600).round()
ef = rng.normal(47, 8, n).clip(15, 65).round()
typical = (rng.random(n) < 0.35).astype(int)
rwma = rng.choice([0, 1, 2, 3, 4], n, p=[0.55, 0.15, 0.15, 0.1, 0.05])

risk = (0.045 * (age - 58) + 0.5 * male + 0.7 * dm + 0.5 * htn + 0.5 * smoker + 1.0 * typical
        + 0.01 * (ldl - 105) - 0.02 * (hdl - 42) - 0.05 * (ef - 47) + 0.35 * rwma)
lad = rng.random(n) < sig(risk - 0.4 + rng.normal(0, 0.8, n))
lcx = rng.random(n) < sig(risk - 1.0 + rng.normal(0, 0.8, n))
rca = rng.random(n) < sig(risk - 0.8 + rng.normal(0, 0.8, n))
cad = lad | lcx | rca

df = pd.DataFrame({
    "Age": age, "Weight": weight, "Length": length, "Sex": np.where(male, "Male", "Fmale"), "BMI": bmi,
    "DM": dm, "HTN": htn, "Current Smoker": smoker, "EX-Smoker": (rng.random(n) < 0.1).astype(int),
    "FH": (rng.random(n) < 0.15).astype(int), "Obesity": np.where(bmi > 30, "Y", "N"), "CRF": yn(0.03),
    "CVA": yn(0.03), "Airway disease": yn(0.08), "Thyroid Disease": yn(0.05), "CHF": yn(0.02), "DLP": yn(0.4),
    "BP": bp, "PR": pr, "Edema": (rng.random(n) < 0.08).astype(int), "Weak Peripheral Pulse": yn(0.03),
    "Lung rales": yn(0.04), "Systolic Murmur": yn(0.2), "Diastolic Murmur": yn(0.03),
    "Typical Chest Pain": typical, "Dyspnea": yn(0.4),
    "Function Class": rng.choice([0, 1, 2, 3], n, p=[.5, .2, .2, .1]),
    "Atypical": yn(0.4), "Nonanginal": yn(0.1), "Exertional CP": yn(0.02), "LowTH Ang": yn(0.05),
    "Q Wave": (rng.random(n) < 0.05 + 0.1 * (rwma > 0)).astype(int),
    "St Elevation": (rng.random(n) < 0.04).astype(int),
    "St Depression": (rng.random(n) < 0.15 + 0.2 * cad).astype(int),
    "Tinversion": (rng.random(n) < 0.2 + 0.25 * cad).astype(int), "LVH": yn(0.1),
    "Poor R Progression": yn(0.07), "BBB": rng.choice(["N", "LBBB", "RBBB"], n, p=[.9, .05, .05]),
    "FBS": fbs, "CR": rng.normal(1.0, 0.2, n).clip(0.5, 2.5).round(1), "TG": tg, "LDL": ldl, "HDL": hdl,
    "BUN": rng.normal(17, 5, n).clip(6, 50).round(), "ESR": rng.normal(20, 14, n).clip(1, 90).round(),
    "HB": rng.normal(13.2, 1.6, n).clip(8, 18).round(1), "K": rng.normal(4.2, 0.4, n).clip(3, 6).round(1),
    "Na": rng.normal(141, 3, n).clip(128, 150).round(),
    "WBC": rng.normal(7500, 2000, n).clip(3000, 15000).round(-2),
    "Lymph": rng.normal(32, 9, n).clip(5, 60).round(), "Neut": rng.normal(60, 10, n).clip(30, 90).round(),
    "PLT": rng.normal(230, 55, n).clip(100, 500).round(), "EF-TTE": ef, "Region RWMA": rwma,
    "VHD": rng.choice(["N", "mild", "Moderate", "Severe"], n, p=[.5, .35, .1, .05]),
    "LAD": lad.astype(int), "LCX": lcx.astype(int), "RCA": rca.astype(int),
    "Cath": np.where(cad, "Cad", "Normal"),
})
out = Path(__file__).with_name("demo_synthetic.csv")
df.to_csv(out, index=False)
print(f"Wrote {out} ({len(df)} rows) - SYNTHETIC, for pipeline testing only.")
