"""Compare final RF model vs SAPS II on the same cohort.
Verifies the manuscript's claim that RF (AUC 0.805) exceeds SAPS II.
Run from repo root: python scripts/compare_saps2.py
"""
import pandas as pd, numpy as np, os, warnings, joblib
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Load cohort + SAPS II (joined on stay_id) ---
df = pd.read_csv(os.path.join(ROOT, "data", "surg_data.csv"))
sp = pd.read_csv(os.path.join(ROOT, "data", "sapsii_cohort.csv"))
m = df.merge(sp[["stay_id", "sapsii", "sapsii_prob"]], on="stay_id", how="left")
y = m["outcome_death"].values
print(f"Cohort: {len(m)} first-ICU stays, death {y.mean():.4f}")

# --- SAPS II as a predictor (full cohort AUC) ---
saps = m["sapsii"].values
saps_auc = roc_auc_score(y, saps)
print(f"SAPS II full-cohort AUC: {saps_auc:.4f}")

# --- RF model predictions on same test split ---
# Rebuild features exactly as final_rf_model.py
feat = df.copy()
feat["male"] = (feat["gender"] == "M").astype(int)
for r in ["black", "hispanic", "asian", "other", "white"]:
    if r == "other":
        feat["race_other"] = feat["race"].isin(["OTHER", "UNKNOWN", "UNKNOWN/NOT SPECIFIED"]).astype(int)
    else:
        feat[f"race_{r}"] = feat["race"].str.contains(r.upper(), na=False).astype(int)
for c in ["albumin", "hgb", "bili_total", "lactate"]:
    feat[f"{c}_missing"] = feat[c].isna().astype(int)
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
fc = lab + [f"{c}_missing" for c in ["albumin", "hgb", "bili_total", "lactate"]] + comorb + surg + demo
X = feat[fc].apply(lambda c: c.fillna(c.median()) if c.dtype.kind in "fiu" else c)

from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
rf = joblib.load(os.path.join(ROOT, "models", "death_rf.pkl"))
p_rf = rf.predict_proba(Xte)[:, 1]

# SAPS II on the same test split
te_idx = Xte.index
saps_te = saps[te_idx]
rf_auc = roc_auc_score(yte, p_rf)
saps_te_auc = roc_auc_score(yte, saps_te)
print(f"\n--- Same test split ({len(yte)} pts) ---")
print(f"Random Forest: AUC {rf_auc:.4f} | PR-AUC {average_precision_score(yte, p_rf):.4f} | Brier {brier_score_loss(yte, p_rf):.4f}")
print(f"SAPS II:       AUC {saps_te_auc:.4f}")
print(f"\nAUC improvement: {rf_auc - saps_te_auc:+.4f}")
