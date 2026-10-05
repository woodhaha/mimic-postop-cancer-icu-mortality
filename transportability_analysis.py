"""Attribution of external-validation AUC drop: feature reduction vs population/temporal migration.
Both external scripts (eicu/mimiciii) train a 35-feature RF on ALL MIMIC-IV rows, then apply
the frozen model to the external cohort (no retraining). This script quantifies:
  49f internal test AUC  ->  35f internal test AUC  (feature-reduction cost, same 70/30 split)
  -> 35f internal        ->  eICU 0.750 / MIMIC-III 0.676  (migration cost)
Run from repo root: python scripts/transportability_analysis.py
"""
import pandas as pd, numpy as np, os, warnings, joblib
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "surg_data.csv"))

# ---- 49-feature set (matches final_rf_model.py) ----
def feats49(d):
    f = d.copy()
    f["male"] = (f["gender"] == "M").astype(int)
    for r in ["black", "hispanic", "asian", "other", "white"]:
        if r == "other":
            f["race_other"] = f["race"].isin(["OTHER", "UNKNOWN", "UNKNOWN/NOT SPECIFIED"]).astype(int)
        else:
            f[f"race_{r}"] = f["race"].str.contains(r.upper(), na=False).astype(int)
    for c in ["albumin", "hgb", "bili_total", "lactate"]:
        f[f"{c}_missing"] = f[c].isna().astype(int)
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
    return f[fc].apply(lambda c: c.fillna(c.median()) if c.dtype.kind in "fiu" else c), fc

# ---- 35-feature set (matches eicu/mimiciii external scripts) ----
COMMON_LABS = ["creatinine", "albumin", "bili_total", "platelets", "wbc",
               "lactate", "ptt", "inr", "sodium", "potassium", "glucose"]
CH_COMORB = ["mi", "chf", "pvd", "cvd", "dementia", "copd", "rheum", "pud", "mild_liver",
             "dm_nocc", "dm_cc", "paraplegia", "renal", "malignancy", "severe_liver", "mets", "aids"]

def feats35(d):
    f = d.copy()
    f["male"] = (f["gender"] == "M").astype(int)
    f["race_white"] = f["race"].str.contains("WHITE", na=False).astype(int)
    f["age"] = f["admission_age"]
    for c in ["albumin", "bili_total", "lactate"]:
        f[f"{c}_missing"] = f[c].isna().astype(int)
    mcol = {"myocardial_infarct": "mi", "congestive_heart_failure": "chf", "peripheral_vascular_disease": "pvd",
            "cerebrovascular_disease": "cvd", "dementia": "dementia", "chronic_pulmonary_disease": "copd",
            "rheumatic_disease": "rheum", "peptic_ulcer_disease": "pud", "mild_liver_disease": "mild_liver",
            "diabetes_without_cc": "dm_nocc", "diabetes_with_cc": "dm_cc", "paraplegia": "paraplegia",
            "renal_disease": "renal", "malignant_cancer": "malignancy", "severe_liver_disease": "severe_liver",
            "metastatic_solid_tumor": "mets", "aids": "aids"}
    for full, short in mcol.items():
        f[short] = f[full].fillna(0).astype(int)
    f["charlson_comorbidity_index"] = pd.to_numeric(f["charlson_comorbidity_index"], errors="coerce")
    fc = ["age", "male", "race_white"] + COMMON_LABS + \
         [f"{c}_missing" for c in ["albumin", "bili_total", "lactate"]] + CH_COMORB + ["charlson_comorbidity_index"]
    return f[fc].apply(lambda c: c.fillna(c.median()) if c.dtype.kind in "fiu" else c), fc

def ci(y, p, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y)); a = []
    for _ in range(n):
        i = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[i])) < 2: continue
        a.append(roc_auc_score(y[i], p[i]))
    return np.percentile(a, [2.5, 97.5])

def main():
    y = df["outcome_death"].astype(int).values
    X49, fc49 = feats49(df)
    X35, fc35 = feats35(df)
    print(f"Features: 49f={len(fc49)}, 35f={len(fc35)}; n={len(df)}, death {y.mean():.3f}")

    Xtr49, Xte49, ytr, yte = train_test_split(X49, y, test_size=0.3, stratify=y, random_state=42)
    Xtr35, Xte35, _, _ = train_test_split(X35, y, test_size=0.3, stratify=y, random_state=42)

    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1)

    # 49f internal test
    rf.fit(Xtr49, ytr); p49 = rf.predict_proba(Xte49)[:, 1]
    a49 = roc_auc_score(yte, p49); c49 = ci(yte, p49)

    # 35f internal test (same split)
    rf.fit(Xtr35, ytr); p35 = rf.predict_proba(Xte35)[:, 1]
    a35 = roc_auc_score(yte, p35); c35 = ci(yte, p35)

    # 35f full-data CV (matches external scripts' training setup)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(rf, X35, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    # external frozen-transfer results (saved by eicu/mimiciii scripts)
    ye = np.load(os.path.join(ROOT, "data", "eicu_y.npy")); pe = np.load(os.path.join(ROOT, "data", "eicu_prob.npy"))
    ym3 = np.load(os.path.join(ROOT, "data", "mimiciii_y.npy")); pm3 = np.load(os.path.join(ROOT, "data", "mimiciii_prob.npy"))
    ae = roc_auc_score(ye, pe); am3 = roc_auc_score(ym3, pm3)

    print("\n=== Transportability attribution ===")
    print(f"49f internal test AUC:  {a49:.3f} (95% CI {c49[0]:.3f}-{c49[1]:.3f})   <-- published")
    print(f"35f internal test AUC:  {a35:.3f} (95% CI {c35[0]:.3f}-{c35[1]:.3f})   [feature-reduction cost on same split]")
    print(f"35f full-data CV AUC:   {cv_auc.mean():.3f} ± {cv_auc.std():.3f}      <-- external scripts' training setup")
    print(f"eICU frozen AUC:        {ae:.3f}    [population migration cost]")
    print(f"MIMIC-III frozen AUC:   {am3:.3f}   [temporal + coding migration cost]")

    print("\n=== Decomposition (AUC delta) ===")
    red = a49 - a35
    mig_e = a35 - ae
    mig_3 = a35 - am3
    print(f"Feature reduction (49f->35f internal): {red:+.3f}")
    print(f"Population migration (35f int -> eICU): {mig_e:+.3f}")
    print(f"Temporal migration (35f int -> MIMIC-III): {mig_3:+.3f}")
    print(f"\nCheck: 49f - eICU = {a49 - ae:+.3f} = reduction + migration = {red + mig_e:+.3f}")
    print(f"       49f - MIMIC-III = {a49 - am3:+.3f} = {red + mig_3:+.3f}")

    pd.DataFrame([
        {"stage": "49f internal test", "AUC": round(a49, 3), "CI_low": round(c49[0], 3), "CI_high": round(c49[1], 3)},
        {"stage": "35f internal test", "AUC": round(a35, 3), "CI_low": round(c35[0], 3), "CI_high": round(c35[1], 3)},
        {"stage": "35f full-data CV", "AUC": round(cv_auc.mean(), 3), "CI_low": "", "CI_high": ""},
        {"stage": "eICU (frozen transfer)", "AUC": round(ae, 3), "CI_low": "", "CI_high": ""},
        {"stage": "MIMIC-III (frozen transfer)", "AUC": round(am3, 3), "CI_low": "", "CI_high": ""},
    ]).to_csv(os.path.join(ROOT, "reports", "transportability.csv"), index=False)
    print("\nSaved: reports/transportability.csv")

if __name__ == "__main__":
    main()
