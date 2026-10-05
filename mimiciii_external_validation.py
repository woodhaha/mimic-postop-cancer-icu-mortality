"""External validation on MIMIC-III (temporal validation: 2001-2012 vs MIMIC-IV 2008-2022).
Builds same 35 overlapping features (demographics, labs, Charlson from ICD-9),
retrains RF on MIMIC-IV, applies to MIMIC-III cohort.
Run from repo root: python scripts/mimiciii_external_validation.py

Laboratory window: last value within 0-48h of hospital admission (admittime),
chosen as a pre-operative proxy (MIMIC-III lacks a surgery timestamp in
procedures_icd). This replaces the previous "last lab of the entire hospital
stay" which was contaminated by near-death values (time leakage for the
in-hospital mortality outcome). The 48h-admission window is disclosed as an
approximation to the pre-operative period.
"""
import pandas as pd, numpy as np, os, warnings, joblib
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from google.cloud import bigquery

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
client = bigquery.Client()

COMMON_LABS = ["creatinine", "albumin", "bili_total", "platelets", "wbc",
               "lactate", "ptt", "inr", "sodium", "potassium", "glucose"]
CH_COMORB = ["mi","chf","pvd","cvd","dementia","copd","rheum","pud","mild_liver",
             "dm_nocc","dm_cc","paraplegia","renal","malignancy","severe_liver","mets","aids"]

def build_mimiciv_features():
    df = pd.read_csv(os.path.join(ROOT, "data", "surg_data.csv"))
    feat = df.copy()
    feat["male"] = (feat["gender"] == "M").astype(int)
    feat["race_white"] = feat["race"].str.contains("WHITE", na=False).astype(int)
    feat["age"] = feat["admission_age"]
    for c in ["albumin", "bili_total", "lactate"]:
        feat[f"{c}_missing"] = feat[c].isna().astype(int)
    mcol = {"myocardial_infarct":"mi","congestive_heart_failure":"chf","peripheral_vascular_disease":"pvd",
            "cerebrovascular_disease":"cvd","dementia":"dementia","chronic_pulmonary_disease":"copd",
            "rheumatic_disease":"rheum","peptic_ulcer_disease":"pud","mild_liver_disease":"mild_liver",
            "diabetes_without_cc":"dm_nocc","diabetes_with_cc":"dm_cc","paraplegia":"paraplegia",
            "renal_disease":"renal","malignant_cancer":"malignancy","severe_liver_disease":"severe_liver",
            "metastatic_solid_tumor":"mets","aids":"aids"}
    for full, short in mcol.items():
        feat[short] = feat[full].fillna(0).astype(int)
    feat["charlson_comorbidity_index"] = pd.to_numeric(feat["charlson_comorbidity_index"], errors="coerce")
    cols = ["age","male","race_white"] + COMMON_LABS + [f"{c}_missing" for c in ["albumin","bili_total","lactate"]] + CH_COMORB + ["charlson_comorbidity_index"]
    med = feat[cols].median()
    X = feat[cols].fillna(med)
    y = feat["outcome_death"].astype(int).values
    return X, y, cols, med

def fetch_mimiciii_labs(hadm):
    sql = """
    SELECT l.hadm_id,
      l.CREATININE AS creatinine, l.ALBUMIN AS albumin,
      l.BILIRUBIN AS bili_total, l.PLATELET AS platelets,
      l.WBC AS wbc, l.LACTATE AS lactate, l.PTT AS ptt,
      l.INR AS inr, l.SODIUM AS sodium, l.POTASSIUM AS potassium,
      l.GLUCOSE AS glucose
    FROM (
      SELECT l.*, ROW_NUMBER() OVER (PARTITION BY l.hadm_id ORDER BY l.charttime DESC) rn
      FROM `physionet-data.mimiciii_derived.pivoted_lab` l
      JOIN `physionet-data.mimiciii_clinical.admissions` a ON l.hadm_id = a.hadm_id
      WHERE l.hadm_id IN UNNEST(@hadm)
        AND l.charttime BETWEEN TIMESTAMP_ADD(a.admittime, INTERVAL 0 HOUR)
                            AND TIMESTAMP_ADD(a.admittime, INTERVAL 48 HOUR)
    ) l
    WHERE rn = 1
    """
    return client.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter('hadm','INT64',hadm)]
    )).result().to_dataframe()

def fetch_mimiciii_age(hadm):
    sql = """
    SELECT a.hadm_id,
      DATE_DIFF(DATE(a.admittime), DATE(p.dob), YEAR) AS age
    FROM `physionet-data.mimiciii_clinical.admissions` a
    JOIN `physionet-data.mimiciii_clinical.patients` p ON a.subject_id=p.subject_id
    WHERE a.hadm_id IN UNNEST(@hadm)
    """
    return client.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter('hadm','INT64',hadm)]
    )).result().to_dataframe()

def icd9_charlson(code):
    c = str(code).strip().replace(".", "")
    if not c.isdigit(): return []
    res = []; c3 = c[:3]
    if c3 in ("410","412") or c.startswith("410") or c.startswith("412"): res.append("mi")
    if c3 in ("428",) or c.startswith("428"): res.append("chf")
    if c3 in ("443","441") or c3.startswith("44"): res.append("pvd")
    if c3 in ("430","431","432","433","434","435","436","437","438"): res.append("cvd")
    if c3 in ("290",) or c.startswith("290"): res.append("dementia")
    if c3 in ("490","491","492","493","494","495","496","500","501","502","503","504","505","506","507"): res.append("copd")
    if c3 in ("710","714","725"): res.append("rheum")
    if c3 in ("531","532","533","534"): res.append("pud")
    if c3 in ("571",) or c.startswith("571"): res.append("mild_liver")
    if c3.startswith("250") and c3[:4] in ("2504","2505","2506","2507"): res.append("dm_cc")
    elif c.startswith("250"): res.append("dm_nocc")
    if c3 in ("344",) or c3.startswith("344"): res.append("paraplegia")
    if c3 in ("582","583","585","586","588") or c.startswith("585"): res.append("renal")
    if c3.startswith("196") or c3.startswith("197") or c3.startswith("198") or c3.startswith("199"): res.append("mets")
    if c3 in ("571","572","573") or c3.startswith("571") or c3.startswith("572") or c3.startswith("573"): res.append("severe_liver")
    if c3.startswith("042") or c3.startswith("043") or c3.startswith("044"): res.append("aids")
    if c3.startswith("140") or c3.startswith("150") or c3.startswith("151") or c3.startswith("153") or c3.startswith("154") or c3.startswith("155") or c3.startswith("157") or c3.startswith("162"): res.append("malignancy")
    return res

def main():
    Xm, ym, cols, med = build_mimiciv_features()
    print(f"MIMIC-IV: {len(Xm)} rows, {len(cols)} features, death {ym.mean():.3f}")

    cohort = pd.read_csv(os.path.join(ROOT, "data", "mimiciii_cohort.csv"))
    hadm = cohort.hadm_id.tolist()
    labs = fetch_mimiciii_labs(hadm)
    print(f"MIMIC-III labs: {len(labs)} rows")

    dx_sql = """
    SELECT d.hadm_id, d.icd9_code
    FROM `physionet-data.mimiciii_clinical.diagnoses_icd` d
    WHERE d.hadm_id IN UNNEST(@hadm) AND d.icd9_code IS NOT NULL
    """
    dx = client.query(dx_sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter('hadm','INT64',hadm)]
    )).result().to_dataframe()
    print(f"MIMIC-III diagnosis rows: {len(dx)}")
    ch_flags = {c: {} for c in CH_COMORB}
    for _, row in dx.iterrows():
        sid = row["hadm_id"]
        for ch in icd9_charlson(row["icd9_code"]):
            ch_flags[ch][sid] = 1

    ages = fetch_mimiciii_age(hadm)
    ef = cohort.merge(labs, on="hadm_id", how="left").merge(ages, on="hadm_id", how="left")
    ef["male"] = (ef["gender"] == "M").astype(int)
    ef["race_white"] = ef["ethnicity"].str.contains("WHITE", na=False).astype(int)
    # MIMIC-III masks age >89 as >300; cap to 90 (consistent with MIMIC-IV). Restrict to adults.
    ef["age"] = np.where(ef["age"] > 89, 90, ef["age"])
    ef = ef[ef["age"] >= 18].copy()
    for c in ["albumin","bili_total","lactate"]:
        ef[f"{c}_missing"] = ef[c].isna().astype(int)
    for ch in CH_COMORB:
        ef[ch] = ef["hadm_id"].map(ch_flags[ch]).fillna(0).astype(int)
    cci = (ef["mi"]+ef["chf"]+ef["pvd"]+ef["cvd"]+ef["dementia"]+ef["copd"]
           +ef["rheum"]+ef["pud"]+ef["mild_liver"]+ef["dm_nocc"]+2*ef["dm_cc"]
           +2*ef["paraplegia"]+2*ef["renal"]+2*ef["malignancy"]+3*ef["severe_liver"]
           +6*ef["mets"]+6*ef["aids"])
    ef["charlson_comorbidity_index"] = cci
    Xe = ef[cols].fillna(med)
    ye = ef["outcome_death"].astype(int).values
    print(f"MIMIC-III: {len(Xe)} rows, death {ye.mean():.3f}")

    Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=0.3, stratify=ym, random_state=42)
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(rf, Xtr, ytr, cv=cv, scoring="roc_auc")
    rf.fit(Xtr, ytr)
    int_auc = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
    print(f"Internal 35-feature test AUC: {int_auc:.4f}")
    pe = rf.predict_proba(Xe)[:, 1]
    ae = roc_auc_score(ye, pe)
    pra = average_precision_score(ye, pe)
    br = brier_score_loss(ye, pe)
    print(f"\nMIMIC-IV CV AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(f"MIMIC-III external: AUC {ae:.4f} | PR-AUC {pra:.4f} | Brier {br:.4f}")
    print(f"  n={len(ye)}, death rate {ye.mean():.3f}")

    joblib.dump(rf, os.path.join(ROOT,"models","death_rf_mimiciii.pkl"))
    np.save(os.path.join(ROOT,"data","mimiciii_y.npy"), ye)
    np.save(os.path.join(ROOT,"data","mimiciii_prob.npy"), pe)
    pd.DataFrame({"cols": cols}).to_csv(os.path.join(ROOT,"data","mimiciii_common_features.csv"), index=False)
    print("Saved models + data")

if __name__ == "__main__":
    main()
