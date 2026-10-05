"""Missing-data sensitivity: median imputation vs MICE (IterativeImputer) vs complete-case.
Addresses reviewer concern that median imputation may bias results.
Run from repo root: python scripts/missing_data_sensitivity.py
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "surg_data.csv"))

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
y = feat["outcome_death"].astype(int).values
cont = [c for c in fc if feat[c].dtype.kind in "fiu"]

def eval_strategy(X, y, label):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    p = rf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p); brier = brier_score_loss(yte, p); pra = average_precision_score(yte, p)
    print(f"{label}: n={len(X)} AUC={auc:.3f} PR-AUC={pra:.3f} Brier={brier:.3f}")

# A: median imputation (main analysis)
Xa = feat[fc].apply(lambda c: c.fillna(c.median()) if c.dtype.kind in "fiu" else c)
eval_strategy(Xa, y, "Median imputation (main)")

# B: MICE (IterativeImputer) on continuous features
imp = IterativeImputer(random_state=42, max_iter=20)
Xb = feat[fc].copy()
Xb[cont] = imp.fit_transform(feat[cont])
eval_strategy(Xb, y, "MICE (IterativeImputer)")

# C: complete-case (drop rows with any missing lab)
mask = feat[lab].notna().all(axis=1).values
Xc = feat.loc[mask, fc].copy(); yc = y[mask]
for c in cont:
    Xc[c] = Xc[c].fillna(Xc[c].median())
eval_strategy(Xc, yc, "Complete-case")
