"""Friendly labels, groups, help text and typical ranges for the dataset's clinical columns.

Edit this file to rename a field, move it to another group, change its help text, or choose which fields
appear first. Columns not listed here still work: they show their raw name under "Other details".
No retraining is needed after editing this file; just restart the server.
"""
from __future__ import annotations

import re


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


GROUPS = ["About you", "Habits and history", "Symptoms", "Blood tests",
          "Heart tracing (ECG)", "Heart ultrasound (echo)", "Other details"]

# Fields shown by default in the "Simple" view, in this order (normalised column names).
KEY_ORDER = ["age", "sex", "bp", "pr", "currentsmoker", "dm", "typicalchestpain", "ldl", "fbs", "eftte"]

YES = {"y", "yes", "1", "true"}
NO = {"n", "no", "0", "false"}


def i(label, group, help="", unit=None, ref=None, vital=False, options=None):
    return {"label": label, "group": group, "help": help, "unit": unit, "ref": ref,
            "vital": vital, "options": options or {}}


A, H, S, B, E, U, O = GROUPS
INFO = {
    "age": i("Age", A, "Your age in years.", "years", vital=True),
    "sex": i("Sex", A, "Biological sex.", options={"fmale": "Female", "female": "Female", "male": "Male"}),
    "weight": i("Weight", A, "Body weight in kilograms.", "kg"),
    "length": i("Height", A, "Height in centimetres.", "cm"),
    "bmi": i("BMI (body mass index)", A, "Weight relative to height. 18.5 to 24.9 is typical for adults.", "kg/m²",
             [18.5, 24.9], True),
    "dm": i("Diabetes (DM)", H, "Has diabetes been diagnosed?"),
    "htn": i("High blood pressure (HTN)", H, "Has high blood pressure been diagnosed?"),
    "currentsmoker": i("Currently smokes", H, "Do you smoke now?"),
    "exsmoker": i("Used to smoke (EX-Smoker)", H, "Did you smoke in the past but stopped?"),
    "fh": i("Family history of heart disease (FH)", H, "A close relative has had heart disease."),
    "obesity": i("Obesity", H, "Has obesity been diagnosed?"),
    "crf": i("Chronic kidney failure (CRF)", H, "Long-term kidney failure."),
    "cva": i("Past stroke (CVA)", H, "Has there been a stroke before?"),
    "airwaydisease": i("Airway disease", H, "Long-term lung or airway disease, such as asthma."),
    "thyroiddisease": i("Thyroid disease", H, "Any diagnosed thyroid condition."),
    "chf": i("Heart failure (CHF)", H, "The heart is not pumping as well as it should."),
    "dlp": i("Abnormal blood fats (DLP)", H, "Dyslipidemia: cholesterol or fats in the blood are out of range."),
    "bp": i("Blood pressure (BP)", A, "The top number of a reading (systolic), in mmHg. Under 120 is typical.",
            "mmHg", [90, 120], True),
    "pr": i("Pulse rate (PR)", A, "Heartbeats per minute at rest. 60 to 100 is typical. The 3D heart beats at this rate.",
            "bpm", [60, 100], True),
    "edema": i("Swelling in the legs (Edema)", S, "Fluid build-up, often in the ankles or legs."),
    "weakperipheralpulse": i("Weak pulse in limbs", S, "A hard-to-feel pulse in the arms or legs."),
    "lungrales": i("Crackling lung sounds (rales)", S, "Crackling heard in the lungs with a stethoscope."),
    "systolicmurmur": i("Murmur while beating (systolic)", S, "An extra sound heard as the heart squeezes."),
    "diastolicmurmur": i("Murmur between beats (diastolic)", S, "An extra sound heard as the heart relaxes."),
    "typicalchestpain": i("Typical chest pain", S, "Chest pain with the classic heart-related pattern, such as pressure during effort."),
    "dyspnea": i("Shortness of breath (dyspnea)", S, "Difficulty breathing during normal activity."),
    "functionclass": i("Activity limitation (Function class)", S, "How much symptoms limit daily activity. Higher means more limited."),
    "atypical": i("Atypical chest pain", S, "Chest pain that does not fit the classic heart pattern."),
    "nonanginal": i("Chest pain not from the heart", S, "Non-anginal chest pain."),
    "exertionalcp": i("Chest pain on exertion", S, "Chest pain that comes on with physical effort."),
    "lowthang": i("Angina at low effort (LowTH Ang)", S, "Chest pain that starts with very little effort."),
    "fbs": i("Fasting blood sugar (FBS)", B, "Blood sugar after not eating for 8 hours. 70 to 100 is typical.", "mg/dL",
             [70, 100], True),
    "cr": i("Creatinine (CR)", B, "A kidney marker in the blood.", "mg/dL", [0.6, 1.3], True),
    "tg": i("Triglycerides (TG)", B, "A type of blood fat. Under 150 is typical.", "mg/dL", [0, 150], True),
    "ldl": i("LDL cholesterol (the \"bad\" kind)", B, "Cholesterol that can build up in artery walls. Under 100 is typical.",
             "mg/dL", [0, 100], True),
    "hdl": i("HDL cholesterol (the \"good\" kind)", B, "Cholesterol that helps clear the bad kind. Above 40 is typical.",
             "mg/dL", [40, 200], True),
    "bun": i("Urea (BUN)", B, "A kidney marker in the blood.", "mg/dL"),
    "esr": i("Inflammation marker (ESR)", B, "How fast red blood cells settle; higher can mean inflammation.", "mm/h"),
    "hb": i("Haemoglobin (HB)", B, "The protein in red blood cells that carries oxygen.", "g/dL", [12, 17.5], True),
    "k": i("Potassium (K)", B, "A blood salt.", "mEq/L", [3.5, 5.0]),
    "na": i("Sodium (Na)", B, "A blood salt.", "mEq/L", [135, 145]),
    "wbc": i("White blood cells (WBC)", B, "Immune cells in the blood.", "cells/µL"),
    "lymph": i("Lymphocytes (Lymph)", B, "A type of white blood cell, as a percentage.", "%"),
    "neut": i("Neutrophils (Neut)", B, "A type of white blood cell, as a percentage.", "%"),
    "plt": i("Platelets (PLT)", B, "Blood cells that help clotting.", "thousand/µL"),
    "qwave": i("Q wave on ECG", E, "A pattern on the heart tracing that can point to an earlier heart injury."),
    "stelevation": i("ST elevation on ECG", E, "A raised segment on the heart tracing."),
    "stdepression": i("ST depression on ECG", E, "A lowered segment on the heart tracing."),
    "tinversion": i("T-wave inversion on ECG", E, "A flipped T wave on the heart tracing."),
    "lvh": i("Thick heart wall on ECG (LVH)", E, "Left ventricular hypertrophy: the main pumping wall is thickened."),
    "poorrprogression": i("Poor R-wave progression (ECG)", E, "A pattern across the chest leads of the heart tracing."),
    "bbb": i("Bundle branch block (BBB)", E, "A delay in the heart's electrical signal.",
             options={"n": "None", "lbbb": "Left (LBBB)", "rbbb": "Right (RBBB)"}),
    "eftte": i("Heart pumping strength (EF-TTE)", U,
               "Ejection fraction from the heart ultrasound: how much blood the heart pushes out each beat. 55 to 70 is typical.",
               "%", [55, 70], True),
    "regionrwma": i("Wall-motion problem areas (Region RWMA)", U,
                    "Regions of the heart wall that move abnormally on the ultrasound. 0 means none; higher means more."),
    "vhd": i("Heart valve disease (VHD)", U, "How severe any valve problem is.",
             options={"n": "None", "mild": "Mild", "moderate": "Moderate", "severe": "Severe"}),
}


def enrich(fields: list[dict]) -> list[dict]:
    """Add label / group / help / unit / typical range / control kind to each schema field."""
    out = []
    for f in fields:
        f = dict(f)
        key = norm(f["name"])
        info = INFO.get(key, {})
        f["label"] = info.get("label", f["name"])
        f["group"] = info.get("group", "Other details")
        f["help"] = info.get("help", "")
        f["unit"] = info.get("unit")
        f["ref"] = info.get("ref")
        f["vital"] = bool(info.get("vital"))
        f["option_labels"] = info.get("options", {})
        f["key_rank"] = KEY_ORDER.index(key) if key in KEY_ORDER else None
        opts = f.get("options")
        if opts is None:
            f["kind"] = "number"
        else:
            yes = [o for o in opts if norm(o) in YES]
            no = [o for o in opts if norm(o) in NO]
            if len(opts) == 2 and len(yes) == 1 and len(no) == 1:
                f["kind"], f["yes"], f["no"] = "yesno", yes[0], no[0]
            elif len(opts) <= 4:
                f["kind"] = "choice"
            else:
                f["kind"] = "select"
        out.append(f)
    return out
