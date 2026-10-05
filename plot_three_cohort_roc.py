"""Regenerate three-cohort external validation ROC figure with updated
pre-operative-window AUCs. Internal 35-feature curve is computed on the same
70/30 split used by transportability_analysis.py.
Run from repo root: python scripts/plot_three_cohort_roc.py
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- internal 35-feature test curve (same setup as transportability) ----
COMMON_LABS = ["creatinine", "albumin", "bili_total", "platelets", "wbc",
               "lactate", "ptt", "inr", "sodium", "potassium", "glucose"]
CH_COMORB = ["mi", "chf", "pvd", "cvd", "dementia", "copd", "rheum", "pud", "mild_liver",
             "dm_nocc", "dm_cc", "paraplegia", "renal", "malignancy", "severe_liver", "mets", "aids"]
df = pd.read_csv(os.path.join(ROOT, "data", "surg_data.csv"))
f = df.copy()
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
X = f[fc].apply(lambda c: c.fillna(c.median()) if c.dtype.kind in "fiu" else c)
y = f["outcome_death"].astype(int).values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(Xtr, ytr)
pi = rf.predict_proba(Xte)[:, 1]
ai = roc_auc_score(yte, pi)
fpr_i, tpr_i, _ = roc_curve(yte, pi)

# ---- external cohorts ----
ye = np.load(os.path.join(ROOT, "data", "eicu_y.npy"))
pe = np.load(os.path.join(ROOT, "data", "eicu_prob.npy"))
ym = np.load(os.path.join(ROOT, "data", "mimiciii_y.npy"))
pm = np.load(os.path.join(ROOT, "data", "mimiciii_prob.npy"))
ae = roc_auc_score(ye, pe)
am = roc_auc_score(ym, pm)
fpr_e, tpr_e, _ = roc_curve(ye, pe)
fpr_m, tpr_m, _ = roc_curve(ym, pm)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_i, tpr_i, lw=2, color="tab:blue", label=f"MIMIC-IV internal (AUC={ai:.3f})")
ax.plot(fpr_e, tpr_e, lw=2, color="tab:orange", label=f"eICU external (AUC={ae:.3f})")
ax.plot(fpr_m, tpr_m, lw=2, color="tab:green", label=f"MIMIC-III external (AUC={am:.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Internal and external validation (35-feature protocol)")
ax.legend(loc="lower right")
plt.tight_layout()
for out in [os.path.join(ROOT, "figures", "hires", "Figure3_external.pdf"),
            os.path.join(ROOT, "submission", "hires", "Figure3_external.pdf"),
            os.path.join(ROOT, "submission", "bmc_midm", "latex", "figures", "Figure3_external.pdf")]:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved Figure3_external.pdf: internal {ai:.3f}, eICU {ae:.3f}, MIMIC-III {am:.3f}")
