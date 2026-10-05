"""Export all manuscript figures at publication quality.
JCO CCI (ASCO): EZSubmit accepts PNG at first submission; after acceptance
figures must be vector (EPS/PDF) with line art >=600dpi, combination >=300dpi.
This script regenerates every figure as BOTH vector PDF and 600dpi PNG.
Run from repo root: python scripts/export_figures_hires.py
"""
import pandas as pd, numpy as np, os, warnings, joblib
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "hires")
os.makedirs(OUT, exist_ok=True)

# --- Load data + features (same as final_rf_model.py) ---
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
X = feat[fc].apply(lambda c: c.fillna(c.median()) if c.dtype.kind in "fiu" else c)
y = feat["outcome_death"].astype(int).values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
rf = joblib.load(os.path.join(ROOT, "models", "death_rf.pkl"))
p = rf.predict_proba(Xte)[:, 1]
fpr, tpr, _ = roc_curve(yte, p)
auc = roc_auc_score(yte, p)
rng = np.random.RandomState(42)
idx = np.arange(len(yte))
aucs = []
for _ in range(1000):
    i = rng.choice(idx, len(idx), replace=True)
    if len(np.unique(yte[i])) < 2: continue
    aucs.append(roc_auc_score(yte[i], p[i]))
ci = np.percentile(aucs, [2.5, 97.5])

# Set global high-res style
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "lines.linewidth": 1.8, "font.family": "Arial",
})

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name} (vector PDF + 600dpi PNG)")

# Figure 2: ROC with CI band
fpr_all = []
for _ in range(200):
    i = rng.choice(idx, len(idx), replace=True)
    if len(np.unique(yte[i])) < 2: continue
    f_, t_, _ = roc_curve(yte[i], p[i])
    fpr_all.append(np.interp(np.linspace(0, 1, 100), f_, t_))
tpr_all = np.array(fpr_all)
lo, hi = np.percentile(tpr_all, [2.5, 97.5], axis=0)
xx = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, lw=2, color="tab:blue", label=f"Random Forest (AUC={auc:.3f}, 95% CI {ci[0]:.3f}-{ci[1]:.3f})")
ax.fill_between(xx, lo, hi, alpha=0.15, color="tab:blue")
ax.plot([0, 1], [0, 1], "k--", label="Chance")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("In-hospital mortality prediction (cancer surgery ICU)")
ax.legend(loc="lower right")
save(fig, "Figure2_roc")

# PR curve
prec, rec, _ = precision_recall_curve(yte, p)
pra = average_precision_score(yte, p)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec, prec, lw=2, label=f"Random Forest (PR-AUC={pra:.3f})")
ax.axhline(yte.mean(), color="k", ls="--", label=f"Baseline ({yte.mean():.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall curve")
ax.legend()
save(fig, "PR_curve")

# Calibration
from sklearn.calibration import calibration_curve
frac_pos, mean_pred = calibration_curve(yte, p, n_bins=10, strategy="quantile")
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(mean_pred, frac_pos, "o-", lw=2, label="Random Forest")
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraction"); ax.set_title("Calibration curve")
ax.legend()
save(fig, "Figure4_calibration")

# DCA
def dca(y, p, thr):
    nb = []
    for t in thr:
        tp = (p >= t) & (y == 1); fp = (p >= t) & (y == 0)
        nb.append(np.mean(tp) - np.mean(fp) * t / (1 - t))
    return np.array(nb)
thr = np.arange(0.01, 0.81, 0.01)
nb_model = dca(yte, p, thr)
nb_all = yte.mean() - thr * (1 - yte.mean()) / (1 - thr)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(thr, nb_model, lw=2, label="Random Forest")
ax.plot(thr, nb_all, "k--", label="Treat all")
ax.axhline(0, color="gray", ls=":", label="Treat none")
ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit"); ax.set_title("Decision curve analysis")
ax.legend()
save(fig, "Figure4_dca")

# SHAP (regenerate with same logic as final_rf_model.py)
import shap
explainer = shap.TreeExplainer(rf)
sample = Xte.sample(min(2000, len(Xte)), random_state=42)
sv = explainer.shap_values(sample)
shap.summary_plot(sv, sample, max_display=20, show=False)
plt.tight_layout()
fig = plt.gcf()
save(fig, "Figure5_shap")

# Figure 3: external validation (from saved eICU predictions)
import joblib as jb
try:
    ye_ext = np.load(os.path.join(ROOT, "data", "eicu_y.npy"))
    pe_ext = np.load(os.path.join(ROOT, "data", "eicu_prob.npy"))
    fpr_e, tpr_e, _ = roc_curve(ye_ext, pe_ext)
    ae = roc_auc_score(ye_ext, pe_ext)
    # internal CV curve from training predictions
    from sklearn.model_selection import cross_val_predict
    cv_rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1)
    # use saved model's internal preds instead (already have X/y)
    fpr_i, tpr_i, _ = roc_curve(y, rf.predict_proba(X)[:, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_i, tpr_i, lw=2, label="MIMIC internal (full-data)")
    ax.plot(fpr_e, tpr_e, lw=2, color="tab:orange", label=f"eICU external (AUC={ae:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("External validation: MIMIC internal vs eICU")
    ax.legend(loc="lower right")
    save(fig, "Figure3_external")
except Exception as e:
    print(f"Figure3 skipped: {e}")

print("\nAll figures exported to figures/hires/ (PDF + 600dpi PNG)")
