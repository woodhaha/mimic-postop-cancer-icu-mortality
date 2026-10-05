"""LASSO feature selection + multi-algorithm benchmark for death prediction.
Run from repo root: python scripts/feature_select_multialgo.py
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.feature_selection import SelectFromModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "surg_data.csv"))

# --- feature engineering (mirror train_death_model.py) ---
feat = df.copy()
feat["male"] = (feat["gender"] == "M").astype(int)
for r in ["black","hispanic","asian","other","white"]:
    if r=="other":
        feat["race_other"] = feat["race"].isin(["OTHER","UNKNOWN","UNKNOWN/NOT SPECIFIED"]).astype(int)
    else:
        feat[f"race_{r}"] = feat["race"].str.contains(r.upper(), na=False).astype(int)
feat["charlson_comorbidity_index"] = pd.to_numeric(feat["charlson_comorbidity_index"], errors="coerce")
for col in ["albumin","hgb","bili_total","lactate"]:
    feat[f"{col}_missing"] = feat[col].isna().astype(int)

lab = ["creatinine","albumin","bili_total","hgb","platelets","wbc","lactate","ptt","inr","sodium","potassium","glucose"]
comorb = ["myocardial_infarct","congestive_heart_failure","peripheral_vascular_disease","cerebrovascular_disease",
          "dementia","chronic_pulmonary_disease","rheumatic_disease","peptic_ulcer_disease","mild_liver_disease",
          "diabetes_without_cc","diabetes_with_cc","paraplegia","renal_disease","malignant_cancer",
          "severe_liver_disease","metastatic_solid_tumor","aids","charlson_comorbidity_index"]
surg = ["n_procedures","liver_transplant","liver_resection","open_pancreatic_excision","any_pancreatic_excision",
        "GI_resection","thoracic_surgery","ortho_surgery"]
demo = ["admission_age","male","race_black","race_hispanic","race_asian","race_other","race_white"]
feat_cols = lab + [f"{c}_missing" for c in ["albumin","hgb","bili_total","lactate"]] + comorb + surg + demo

X = feat[feat_cols].apply(lambda c: c.fillna(c.median()) if c.dtype.kind in "fiu" else c)
y = feat["outcome_death"].astype(int).values
print(f"Features: {len(feat_cols)}, shape {X.shape}")

# --- LASSO feature selection ---
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
lasso = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=5000, random_state=42)
lasso.fit(Xs, y)
sel_coef = np.abs(lasso.coef_[0])
sel_mask = sel_coef > 0
selected = [f for f, m in zip(feat_cols, sel_mask) if m]
print(f"\nLASSO selected {len(selected)}/{len(feat_cols)} features (C=0.1):")
print(selected)

# --- Split ---
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# --- Multi-algorithm benchmark (on selected features) ---
Xtr_s = Xtr[selected]; Xte_s = Xte[selected]
Xtr_ss = scaler.fit_transform(Xtr_s); Xte_ss = scaler.transform(Xte_s)

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                  scale_pos_weight=(ytr.shape[0]-ytr.sum())/ytr.sum(), random_state=42, n_jobs=-1),
    "LightGBM": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                    scale_pos_weight=(ytr.shape[0]-ytr.sum())/ytr.sum(), random_state=42, n_jobs=-1),
    "CatBoost": CatBoostClassifier(iterations=300, learning_rate=0.05, depth=4,
                                    auto_class_weights="Balanced",
                                    verbose=0, random_state=42),
}
# For linear/SVM use scaled data; for trees use raw
results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, m in models.items():
    use_ss = isinstance(m, (LogisticRegression, SVC))
    Xcv = Xtr_ss if use_ss else Xtr_s
    aucs = cross_val_score(m, Xcv, ytr, cv=cv, scoring="roc_auc", n_jobs=1)
    m.fit(Xcv, ytr)
    Xtest = Xte_ss if use_ss else Xte_s
    yprob = m.predict_proba(Xtest)[:, 1]
    auc = roc_auc_score(yte, yprob)
    prauc = average_precision_score(yte, yprob)
    brier = brier_score_loss(yte, yprob)
    results.append({"model": name, "cv_auc": f"{aucs.mean():.4f}±{aucs.std():.4f}",
                    "test_auc": round(auc,4), "pr_auc": round(prauc,4), "brier": round(brier,4)})
    print(f"{name:22s} CV AUC {aucs.mean():.4f}±{aucs.std():.4f} | Test AUC {auc:.4f} | PR {prauc:.4f} | Brier {brier:.4f}")

res = pd.DataFrame(results)
res.to_csv(os.path.join(ROOT, "reports", "multialgo_benchmark.csv"), index=False)
pd.DataFrame({"feature": selected}).to_csv(os.path.join(ROOT, "data", "lasso_selected.csv"), index=False)
print("\nSaved: reports/multialgo_benchmark.csv, data/lasso_selected.csv")
print("\n=== Feature count summary ===")
print(f"Full: {len(feat_cols)}, LASSO-selected: {len(selected)}")
