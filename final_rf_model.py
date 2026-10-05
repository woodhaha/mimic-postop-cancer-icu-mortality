"""Final Random Forest model: train, metrics, and all figures.
Consolidated replacement for the ad-hoc RF pipeline (death_rf.pkl / roc_ci.png
were previously generated without a committed script).
Run from repo root: python scripts/final_rf_model.py
"""
import pandas as pd, numpy as np, os, warnings, joblib
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "surg_data.csv"))
print(f"Loaded {len(df)} rows, death={df.outcome_death.sum()} ({100*df.outcome_death.mean():.1f}%)")

# --- Feature engineering (identical to prior RF) ---
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
print(f"Features: {len(fc)}, shape {X.shape}")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
print(f"Train {len(Xtr)}, Test {len(Xte)}, test death {yte.sum()} ({100*yte.mean():.1f}%)")

# --- Final RF (400 trees, balanced) ---
rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
cv_auc = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
rf.fit(Xtr, ytr)
p = rf.predict_proba(Xte)[:, 1]
auc, pra, brier = roc_auc_score(yte, p), average_precision_score(yte, p), brier_score_loss(yte, p)
print(f"CV AUC (15-fold): {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"Test AUC: {auc:.4f} | PR-AUC: {pra:.4f} | Brier: {brier:.4f}")

# Youden optimal threshold
fpr, tpr, thr = roc_curve(yte, p)
j = tpr - fpr
t_opt = thr[np.argmax(j)]
pred = (p >= t_opt).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
sens, spec = tp/(tp+fn), tn/(tn+fp)
ppv, npv = tp/(tp+fp), tn/(tn+fn)
print(f"Youden thr={t_opt:.3f}: Sens {sens:.3f} Spec {spec:.3f} PPV {ppv:.3f} NPV {npv:.3f}")

# bootstrap 95% CI for AUC
rng = np.random.RandomState(42)
idx = np.arange(len(yte)); aucs = []
for _ in range(1000):
    i = rng.choice(idx, len(idx), replace=True)
    if len(np.unique(yte[i])) < 2: continue
    aucs.append(roc_auc_score(yte[i], p[i]))
ci = np.percentile(aucs, [2.5, 97.5])
print(f"AUC 95% CI: {ci[0]:.3f}-{ci[1]:.3f}")

joblib.dump(rf, os.path.join(ROOT, "models", "death_rf.pkl"))
np.save(os.path.join(ROOT, "data", "y_test.npy"), yte)
np.save(os.path.join(ROOT, "data", "y_prob.npy"), p)
print("Saved: models/death_rf.pkl, data/y_test.npy, data/y_prob.npy")

# --- Figures ---
# ROC with bootstrap CI band
fpr_all, tpr_all = [], []
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
ax.legend(loc="lower right"); plt.tight_layout()
plt.savefig(os.path.join(ROOT, "figures", "roc_ci.png"), dpi=150); plt.close()

# PR curve
prec, rec, _ = precision_recall_curve(yte, p)
plt.figure(figsize=(6, 5))
plt.plot(rec, prec, lw=2, label=f"Random Forest (PR-AUC={pra:.3f})")
plt.axhline(yte.mean(), color="k", ls="--", label=f"Baseline ({yte.mean():.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curve")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(ROOT, "figures", "pr_curve.png"), dpi=150); plt.close()

# Calibration
from sklearn.calibration import calibration_curve
frac_pos, mean_pred = calibration_curve(yte, p, n_bins=10, strategy="quantile")
plt.figure(figsize=(6, 5))
plt.plot(mean_pred, frac_pos, "o-", lw=2, label="Random Forest")
plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
plt.xlabel("Predicted probability"); plt.ylabel("Observed fraction"); plt.title("Calibration curve")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(ROOT, "figures", "calibration_rf.png"), dpi=150); plt.close()

# Calibration slope (logistic regression of outcome on logit of predicted probability)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
_lp = np.log(np.clip(p, 1e-6, 1-1e-6) / (1 - np.clip(p, 1e-6, 1-1e-6))).reshape(-1, 1)
_cal = LogisticRegression(penalty=None, max_iter=1000).fit(_lp, yte)
slope = _cal.coef_[0][0]; intercept = _cal.intercept_[0]
print(f"Calibration slope: {slope:.3f}, intercept: {intercept:.3f}")
iso = IsotonicRegression(out_of_bounds="clip").fit(p, yte)
p_iso = iso.predict(p)
auc_iso = roc_auc_score(yte, p_iso)
_lp_iso = np.log(np.clip(p_iso, 1e-6, 1-1e-6) / (1 - np.clip(p_iso, 1e-6, 1-1e-6))).reshape(-1, 1)
_cal_iso = LogisticRegression(penalty=None, max_iter=1000).fit(_lp_iso, yte)
slope_iso = _cal_iso.coef_[0][0]
print(f"Isotonic recalibrated: AUC {auc_iso:.3f}, slope {slope_iso:.3f}")

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
plt.figure(figsize=(6, 5))
plt.plot(thr, nb_model, lw=2, label="Random Forest")
plt.plot(thr, nb_all, "k--", label="Treat all")
plt.axhline(0, color="gray", ls=":", label="Treat none")
plt.xlabel("Threshold probability"); plt.ylabel("Net benefit"); plt.title("Decision curve analysis")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(ROOT, "figures", "dca_rf.png"), dpi=150); plt.close()

# SHAP
import shap
explainer = shap.TreeExplainer(rf)
sample = Xte.sample(min(2000, len(Xte)), random_state=42)
sv = explainer.shap_values(sample)
shap.summary_plot(sv, sample, max_display=20, show=False)
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures", "shap_rf.png"), dpi=150, bbox_inches="tight"); plt.close()

print("Saved figures: roc_ci.png, pr_curve.png, calibration_rf.png, dca_rf.png, shap_rf.png")
