"""Predict in-hospital mortality risk for a cancer surgery ICU patient.
Runnable interface for the final Random Forest model (models/death_rf.pkl).

Usage:
  python scripts/predict_risk.py --age 72 --gender M --thoracic 1 --charlson 8 \
      --creatinine 1.2 --albumin 2.9 --bilirubin 0.7 --hgb 9.5 --platelets 180 \
      --wbc 11 --lactate 2.1 --ptt 32 --inr 1.4 --sodium 138 --potassium 4.2 \
      --glucose 130 --n_procedures 5

Output: predicted probability of in-hospital mortality.
"""
import argparse, os, joblib
import pandas as pd, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def build_row(a):
    """Build a single feature row in the model's exact training order."""
    lab = ["creatinine", "albumin", "bili_total", "hgb", "platelets", "wbc",
           "lactate", "ptt", "inr", "sodium", "potassium", "glucose"]
    comorb = ["myocardial_infarct", "congestive_heart_failure", "peripheral_vascular_disease",
              "cerebrovascular_disease", "dementia", "chronic_pulmonary_disease", "rheumatic_disease",
              "peptic_ulcer_disease", "mild_liver_disease", "diabetes_without_cc", "diabetes_with_cc",
              "paraplegia", "renal_disease", "malignant_cancer", "severe_liver_disease",
              "metastatic_solid_tumor", "aids", "charlson_comorbidity_index"]
    surg = ["n_procedures", "liver_transplant", "liver_resection", "open_pancreatic_excision",
            "any_pancreatic_excision", "GI_resection", "thoracic_surgery", "ortho_surgery"]
    demo = ["admission_age", "male", "race_black", "race_hispanic", "race_asian", "race_other", "race_white"]
    vals = {
        "creatinine": a.creatinine, "albumin": a.albumin, "bili_total": a.bilirubin,
        "hgb": a.hgb, "platelets": a.platelets, "wbc": a.wbc, "lactate": a.lactate,
        "ptt": a.ptt, "inr": a.inr, "sodium": a.sodium, "potassium": a.potassium,
        "glucose": a.glucose,
        "albumin_missing": 0, "hgb_missing": 0, "bili_total_missing": 0, "lactate_missing": 0,
        "charlson_comorbidity_index": a.charlson,
        "n_procedures": a.n_procedures, "thoracic_surgery": a.thoracic,
        "admission_age": a.age, "male": 1 if a.gender == "M" else 0,
        "race_black": a.race_black, "race_hispanic": a.race_hispanic,
        "race_asian": a.race_asian, "race_other": a.race_other, "race_white": a.race_white,
    }
    row = {c: vals.get(c, 0) for c in lab + [f"{x}_missing" for x in ["albumin", "hgb", "bili_total", "lactate"]] + comorb + surg + demo}
    return pd.DataFrame([row])

def main():
    ap = argparse.ArgumentParser(description="Predict mortality risk with the final RF model")
    for name, default, help_ in [
        ("--age", 68, "age (years)"), ("--thoracic", 0, "1 if thoracic surgery"),
        ("--charlson", 7, "Charlson index"),
        ("--creatinine", 0.9, "mg/dL"), ("--albumin", 3.2, "g/dL"),
        ("--bilirubin", 0.6, "mg/dL"), ("--hgb", 10.1, "g/dL"),
        ("--platelets", 200, "x10^9/L"), ("--wbc", 10.0, "x10^9/L"),
        ("--lactate", 1.6, "mmol/L"), ("--ptt", 29, "s"), ("--inr", 1.2, "INR"),
        ("--sodium", 138, "mEq/L"), ("--potassium", 4.2, "mEq/L"),
        ("--glucose", 127, "mg/dL"), ("--n_procedures", 3.0, "number of procedures"),
    ]:
        ap.add_argument(name, type=float, default=default, help=help_)
    ap.add_argument("--gender", type=str, default="M", help="M or F")
    for name in ["--race_black", "--race_hispanic", "--race_asian", "--race_other", "--race_white"]:
        ap.add_argument(name, type=float, default=0, help="race indicator (0/1)")
    a = ap.parse_args()

    X = build_row(a)
    rf = joblib.load(os.path.join(ROOT, "models", "death_rf.pkl"))
    prob = rf.predict_proba(X)[0, 1]
    print(f"Predicted in-hospital mortality risk: {prob:.3f} ({100*prob:.1f}%)")
    if prob >= 0.182:
        print("Classified: HIGH risk at Youden threshold 0.182")
    else:
        print("Classified: LOW risk at Youden threshold 0.182")

if __name__ == "__main__":
    main()
