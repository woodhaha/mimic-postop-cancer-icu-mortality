"""Train XGBoost baseline for in-hospital mortality in malignancy-surgery ICU cohort.
Run from repo root: python scripts/train_death_model.py
"""
import pandas as pd, numpy as np, warnings, os
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "surg_data.csv"))
print(f"Loaded {len(df)} rows, death={df.outcome_death.sum()} ({100*df.outcome_death.mean():.1f}%)")

# --- Feature engineering ---
feat = df.copy()

# gender -> binary male
feat["male"] = (feat["gender"] == "M").astype(int)

# race -> categories
feat["race_black"] = feat["race"].str.contains("BLACK", na=False).astype(int)
feat["race_hispanic"] = feat["race"].str.contains("HISPANIC", na=False).astype(int)
feat["race_asian"] = feat["race"].str.contains("ASIAN", na=False).astype(int)
feat["race_other"] = feat["race"].isin(["OTHER", "UNKNOWN", "UNKNOWN/NOT SPECIFIED"]).astype(int)
feat["race_white"] = feat["race"].str.contains("WHITE", na=False).astype(int)

# AKI-risk related derived features
feat["mild_liver_or_severe"] = ((feat["mild_liver_disease"] == 1) | (feat["severe_liver_disease"] == 1)).astype(int)

# Charlson -> numeric
feat["charlson_comorbidity_index"] = pd.to_numeric(feat["charlson_comorbidity_index"], errors="coerce")

# Missing indicators for high-missing labs
for col in ["albumin", "hgb", "bili_total", "lactate"]:
    feat[f"{col}_missing"] = feat[col].isna().astype(int)

# --- Select feature columns ---
id_cols = ["subject_id", "hadm_id", "stay_id", "gender", "race", "outcome_death"]
lab_cols = ["creatinine", "albumin", "bili_total", "hgb", "platelets", "wbc",
            "lactate", "ptt", "inr", "sodium", "potassium", "glucose"]
missing_cols = ["albumin_missing", "hgb_missing", "bili_total_missing", "lactate_missing"]
comorb_cols = ["myocardial_infarct", "congestive_heart_failure", "peripheral_vascular_disease",
               "cerebrovascular_disease", "dementia", "chronic_pulmonary_disease", "rheumatic_disease",
               "peptic_ulcer_disease", "mild_liver_disease", "diabetes_without_cc", "diabetes_with_cc",
               "paraplegia", "renal_disease", "malignant_cancer", "severe_liver_disease",
               "metastatic_solid_tumor", "aids", "charlson_comorbidity_index"]
surg_cols = ["n_procedures", "liver_transplant", "liver_resection", "open_pancreatic_excision",
             "any_pancreatic_excision", "GI_resection", "thoracic_surgery", "ortho_surgery"]
demo_cols = ["admission_age", "male", "race_black", "race_hispanic", "race_asian", "race_other", "race_white"]

feature_cols = lab_cols + missing_cols + comorb_cols + surg_cols + demo_cols
X = feat[feature_cols].copy()
y = feat["outcome_death"].astype(int).values

# Impute numeric with median
X = X.apply(lambda c: c.fillna(c.median()) if c.dtype.kind in "fiu" else c)

print(f"Features: {len(feature_cols)}, shape {X.shape}")

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
print(f"Train {len(X_train)}, Test {len(X_test)}, test death {y_test.sum()} ({100*y_test.mean():.1f}%)")

# --- Model ---
model = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=(y_train.shape[0]-y_train.sum())/y_train.sum(),
    eval_metric="auc", random_state=42, n_jobs=-1)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)], verbose=False)

# --- CV ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
print(f"5-fold CV AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# --- Test metrics ---
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
print(f"Test AUC: {auc:.4f}")
print(f"Test PR-AUC: {pr_auc:.4f} (baseline={y_test.mean():.4f})")

# --- Feature importance ---
imp = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
imp = imp.sort_values("importance", ascending=False)
print("\n=== Top 20 features ===")
print(imp.head(20).to_string(index=False))

# --- Save ---
model.save_model(os.path.join(ROOT, "models", "death_xgb.json"))
np.save(os.path.join(ROOT, "data", "y_test.npy"), y_test)
np.save(os.path.join(ROOT, "data", "y_prob.npy"), y_prob)
pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).to_csv(
    os.path.join(ROOT, "data", "feature_importance.csv"), index=False)
print("\nSaved: models/death_xgb.json, data/y_test.npy, data/y_prob.npy, data/feature_importance.csv")
