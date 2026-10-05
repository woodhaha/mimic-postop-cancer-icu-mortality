"""Plot ROC/PR curves and SHAP summary for death prediction model.
Run from repo root: python scripts/plot_results.py
"""
import pandas as pd, numpy as np, matplotlib, warnings, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
y_test = np.load(os.path.join(ROOT, "data", "y_test.npy"))
y_prob = np.load(os.path.join(ROOT, "data", "y_prob.npy"))
model = xgb.XGBClassifier()
model.load_model(os.path.join(ROOT, "models", "death_xgb.json"))

# ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, lw=2, label=f'XGBoost (AUC={auc(fpr,tpr):.3f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("In-hospital mortality prediction (malignancy-surgery ICU)")
plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig(os.path.join(ROOT, "figures", "roc_curve.png"), dpi=150); plt.close()

# PR
prec, rec, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(rec, prec, lw=2, label=f'XGBoost (PR-AUC={auc(rec,prec):.3f})')
plt.axhline(y_test.mean(), color="k", ls="--", label=f"Baseline ({y_test.mean():.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall curve")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(ROOT, "figures", "pr_curve.png"), dpi=150); plt.close()

# SHAP summary (sample 2000 for speed)
import shap
df = pd.read_csv(os.path.join(ROOT, "data", "surg_data.csv"))
# rebuild features same as training
feat = df.copy()
feat["male"] = (feat["gender"]=="M").astype(int)
for r in ["black","hispanic","asian","other","white"]:
    if r=="other":
        feat["race_other"] = feat["race"].isin(["OTHER","UNKNOWN","UNKNOWN/NOT SPECIFIED"]).astype(int)
    else:
        feat[f"race_{r}"] = feat["race"].str.contains(r.upper(), na=False).astype(int)
lab = ["creatinine","albumin","bili_total","hgb","platelets","wbc","lactate","ptt","inr","sodium","potassium","glucose"]
for c in ["albumin","hgb","bili_total","lactate"]:
    feat[f"{c}_missing"] = feat[c].isna().astype(int)
comorb = ["myocardial_infarct","congestive_heart_failure","peripheral_vascular_disease","cerebrovascular_disease",
          "dementia","chronic_pulmonary_disease","rheumatic_disease","peptic_ulcer_disease","mild_liver_disease",
          "diabetes_without_cc","diabetes_with_cc","paraplegia","renal_disease","malignant_cancer",
          "severe_liver_disease","metastatic_solid_tumor","aids","charlson_comorbidity_index"]
surg = ["n_procedures","liver_transplant","liver_resection","open_pancreatic_excision","any_pancreatic_excision",
        "GI_resection","thoracic_surgery","ortho_surgery"]
demo = ["admission_age","male","race_black","race_hispanic","race_asian","race_other","race_white"]
feat_cols = lab + [f"{c}_missing" for c in ["albumin","hgb","bili_total","lactate"]] + comorb + surg + demo
X = feat[feat_cols].apply(lambda c: c.fillna(c.median()) if c.dtype.kind in "fiu" else c)

explainer = shap.TreeExplainer(model)
sample = X.sample(2000, random_state=42)
sv = explainer.shap_values(sample)
shap.summary_plot(sv, sample, max_display=20, show=False)
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures", "shap_summary.png"), dpi=150, bbox_inches="tight"); plt.close()
print("saved figures/roc_curve.png, figures/pr_curve.png, figures/shap_summary.png")
