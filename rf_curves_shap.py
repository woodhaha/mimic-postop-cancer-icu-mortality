"""Calibration curve, DCA, and SHAP for final Random Forest model.
Run from repo root: python scripts/rf_curves_shap.py
"""
import pandas as pd, numpy as np, os, warnings, joblib
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "surg_data.csv"))

# --- build features (same as final RF) ---
feat = df.copy()
feat["male"] = (feat["gender"] == "M").astype(int)
for r in ["black","hispanic","asian","other","white"]:
    if r=="other":
        feat["race_other"] = feat["race"].isin(["OTHER","UNKNOWN","UNKNOWN/NOT SPECIFIED"]).astype(int)
    else:
        feat[f"race_{r}"] = feat["race"].str.contains(r.upper(), na=False).astype(int)
for c in ["albumin","hgb","bili_total","lactate"]:
    feat[f"{c}_missing"] = feat[c].isna().astype(int)
lab=["creatinine","albumin","bili_total","hgb","platelets","wbc","lactate","ptt","inr","sodium","potassium","glucose"]
comorb=["myocardial_infarct","congestive_heart_failure","peripheral_vascular_disease","cerebrovascular_disease","dementia","chronic_pulmonary_disease","rheumatic_disease","peptic_ulcer_disease","mild_liver_disease","diabetes_without_cc","diabetes_with_cc","paraplegia","renal_disease","malignant_cancer","severe_liver_disease","metastatic_solid_tumor","aids","charlson_comorbidity_index"]
surg=["n_procedures","liver_transplant","liver_resection","open_pancreatic_excision","any_pancreatic_excision","GI_resection","thoracic_surgery","ortho_surgery"]
demo=["admission_age","male","race_black","race_hispanic","race_asian","race_other","race_white"]
fc = lab + [f"{c}_missing" for c in ["albumin","hgb","bili_total","lactate"]] + comorb + surg + demo
X = feat[fc].apply(lambda c: c.fillna(c.median()) if c.dtype.kind in "fiu" else c)
y = feat["outcome_death"].astype(int).values

Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
rf = joblib.load(os.path.join(ROOT, "models", "death_rf.pkl"))
p = rf.predict_proba(Xte)[:,1]

# --- Calibration curve ---
fig, ax = plt.subplots(figsize=(6,5))
frac_pos, mean_pred = calibration_curve(yte, p, n_bins=10, strategy="quantile")
ax.plot(mean_pred, frac_pos, "o-", lw=2, label="Random Forest")
ax.plot([0,1],[0,1],"k--", label="Perfect calibration")
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraction")
ax.set_title("Calibration curve")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(ROOT,"figures","calibration_rf.png"), dpi=150); plt.close()

# --- DCA (net benefit) ---
def dca(y, p, thresholds):
    nb = []
    for t in thresholds:
        tp = (p>=t) & (y==1); fp = (p>=t) & (y==0)
        nb.append(np.mean(tp) - np.mean(fp)*t/(1-t))
    return np.array(nb)

thr = np.arange(0.01, 0.81, 0.01)
nb_model = dca(yte, p, thr)
nb_treat_all = yte.mean() - np.arange(0.01,0.81,0.01)*(1-yte.mean())/(1-np.arange(0.01,0.81,0.01))
fig, ax = plt.subplots(figsize=(6,5))
ax.plot(thr, nb_model, lw=2, label="Random Forest")
ax.plot(thr, nb_treat_all, "k--", label="Treat all")
ax.axhline(0, color="gray", ls=":", label="Treat none")
ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
ax.set_title("Decision curve analysis")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(ROOT,"figures","dca_rf.png"), dpi=150); plt.close()

# --- SHAP for RF ---
import shap
explainer = shap.TreeExplainer(rf)
sample = Xte.sample(2000, random_state=42)
sv = explainer.shap_values(sample)
shap.summary_plot(sv, sample, max_display=20, show=False)
plt.tight_layout(); plt.savefig(os.path.join(ROOT,"figures","shap_rf.png"), dpi=150, bbox_inches="tight"); plt.close()

print(f"Saved: figures/calibration_rf.png, figures/dca_rf.png, figures/shap_rf.png")
print(f"Test AUC: {roc_auc_score(yte,p):.4f}")
