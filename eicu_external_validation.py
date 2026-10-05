"""External validation on eICU (general oncology ICU cohort).
Build overlapping feature set, retrain RF on MIMIC with those features,
predict on eICU. Run from repo root.

Pre-ICU laboratory window: labresultoffset < 0 (before ICU admission).
For postoperative patients the ICU admission follows surgery, so pre-ICU
labs approximate the preoperative period; eICU lacks a surgery timestamp,
so ICU admission is the anchoring event. General oncology cohort (eICU has
no surgery coding): this is a transfer test in a broader oncology
population, not a postoperative-cohort validation.
"""
import pandas as pd, numpy as np, os, warnings, joblib
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from google.cloud import bigquery

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Overlapping features available in both MIMIC and eICU
COMMON_LABS = ["creatinine", "albumin", "bili_total", "platelets", "wbc",
               "lactate", "ptt", "inr", "sodium", "potassium", "glucose"]

def build_mimic_features():
    df = pd.read_csv(os.path.join(ROOT, "data", "surg_data.csv"))
    feat = df.copy()
    feat["male"] = (feat["gender"] == "M").astype(int)
    # race collapsed to white/non-white (eICU has different ethnicity labels)
    feat["race_white"] = feat["race"].str.contains("WHITE", na=False).astype(int)
    feat["age"] = feat["admission_age"]
    for c in ["albumin", "bili_total", "lactate"]:
        feat[f"{c}_missing"] = feat[c].isna().astype(int)
    ch_comorb = ["mi","chf","pvd","cvd","dementia","copd","rheum","pud","mild_liver",
                 "dm_nocc","dm_cc","paraplegia","renal","malignancy","severe_liver","mets","aids"]
    # map MIMIC column names to short names
    mcol = {"myocardial_infarct":"mi","congestive_heart_failure":"chf","peripheral_vascular_disease":"pvd",
            "cerebrovascular_disease":"cvd","dementia":"dementia","chronic_pulmonary_disease":"copd",
            "rheumatic_disease":"rheum","peptic_ulcer_disease":"pud","mild_liver_disease":"mild_liver",
            "diabetes_without_cc":"dm_nocc","diabetes_with_cc":"dm_cc","paraplegia":"paraplegia",
            "renal_disease":"renal","malignant_cancer":"malignancy","severe_liver_disease":"severe_liver",
            "metastatic_solid_tumor":"mets","aids":"aids"}
    for full, short in mcol.items():
        feat[short] = feat[full].fillna(0).astype(int)
    feat["charlson_comorbidity_index"] = pd.to_numeric(feat["charlson_comorbidity_index"], errors="coerce")
    ch_cols = ch_comorb + ["charlson_comorbidity_index"]
    cols = ["age", "male", "race_white"] + COMMON_LABS + [f"{c}_missing" for c in ["albumin","bili_total","lactate"]] + ch_cols
    med = feat[cols].median()
    X = feat[cols].fillna(med)
    y = feat["outcome_death"].astype(int).values
    return X, y, cols, med

def fetch_eicu():
    client = bigquery.Client()
    sql = """
    WITH mal AS (
      SELECT DISTINCT patientunitstayid
      FROM `physionet-data.eicu_crd.diagnosis`
      WHERE REGEXP_CONTAINS(icd9code, r'^(14[0-9]|15[0-9]|16[0-9]|17[0-9]|18[0-9]|19[0-9]|20[0-9])')
    ),
    labs AS (
      SELECT l.patientunitstayid,
        MAX(CASE WHEN l.labname='creatinine' AND l.labresultoffset<0 THEN l.labresult END) AS creatinine,
        MAX(CASE WHEN l.labname='albumin' AND l.labresultoffset<0 THEN l.labresult END) AS albumin,
        MAX(CASE WHEN l.labname='total bilirubin' AND l.labresultoffset<0 THEN l.labresult END) AS bili_total,
        MAX(CASE WHEN l.labname='platelets x 1000' AND l.labresultoffset<0 THEN l.labresult END) AS platelets,
        MAX(CASE WHEN l.labname='WBC x 1000' AND l.labresultoffset<0 THEN l.labresult END) AS wbc,
        MAX(CASE WHEN l.labname='lactate' AND l.labresultoffset<0 THEN l.labresult END) AS lactate,
        MAX(CASE WHEN l.labname='PTT' AND l.labresultoffset<0 THEN l.labresult END) AS ptt,
        MAX(CASE WHEN l.labname='PT - INR' AND l.labresultoffset<0 THEN l.labresult END) AS inr,
        MAX(CASE WHEN l.labname='sodium' AND l.labresultoffset<0 THEN l.labresult END) AS sodium,
        MAX(CASE WHEN l.labname='potassium' AND l.labresultoffset<0 THEN l.labresult END) AS potassium,
        MAX(CASE WHEN l.labname='glucose' AND l.labresultoffset<0 THEN l.labresult END) AS glucose
      FROM `physionet-data.eicu_crd.lab` l JOIN mal m ON l.patientunitstayid=m.patientunitstayid
      WHERE l.labresult IS NOT NULL AND l.labresultoffset<0
      GROUP BY l.patientunitstayid
    )
    SELECT p.patientunitstayid, p.gender, p.ethnicity, p.age,
      l.creatinine, l.albumin, l.bili_total, l.platelets, l.wbc,
      l.lactate, l.ptt, l.inr, l.sodium, l.potassium, l.glucose,
      CASE WHEN p.hospitaldischargestatus='Expired' THEN 1 ELSE 0 END AS outcome_death
    FROM mal m
    JOIN `physionet-data.eicu_crd.patient` p ON m.patientunitstayid=p.patientunitstayid
    LEFT JOIN labs l ON m.patientunitstayid=l.patientunitstayid
    """
    return client.query(sql).to_dataframe()

def fetch_eicu_dx():
    """Fetch all ICD9 diagnosis codes for malignancy cohort (for Charlson)."""
    client = bigquery.Client()
    sql = """
    WITH mal AS (
      SELECT DISTINCT patientunitstayid
      FROM `physionet-data.eicu_crd.diagnosis`
      WHERE REGEXP_CONTAINS(icd9code, r'^(14[0-9]|15[0-9]|16[0-9]|17[0-9]|18[0-9]|19[0-9]|20[0-9])')
    )
    SELECT d.patientunitstayid, d.icd9code
    FROM `physionet-data.eicu_crd.diagnosis` d JOIN mal m ON d.patientunitstayid=m.patientunitstayid
    WHERE d.icd9code IS NOT NULL
    """
    return client.query(sql).to_dataframe()

# Quan 2005 ICD-9 Charlson mapping (simplified)
def icd9_charlson(code):
    """Return list of Charlson comorbidity names for a single ICD-9 code."""
    c = code.strip().replace(".", "")
    if not c.isdigit(): return []
    res = []
    c3 = c[:3]
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
    if c3.startswith("140") or c3.startswith("141") or c3.startswith("142") or c3.startswith("143") or c3.startswith("144") or c3.startswith("145") or c3.startswith("146") or c3.startswith("147") or c3.startswith("148") or c3.startswith("149") or c3.startswith("150") or c3.startswith("151") or c3.startswith("152") or c3.startswith("153") or c3.startswith("154") or c3.startswith("155") or c3.startswith("156") or c3.startswith("157") or c3.startswith("158") or c3.startswith("159") or c3.startswith("160") or c3.startswith("161") or c3.startswith("162") or c3.startswith("163") or c3.startswith("164") or c3.startswith("165") or c3.startswith("166") or c3.startswith("167") or c3.startswith("168") or c3.startswith("169"): res.append("malignancy")
    if c3 in ("185","186","187","188","189","190","191","192","193","194"): res.append("malignancy")
    if c3.startswith("204") or c3.startswith("205") or c3.startswith("206") or c3.startswith("207") or c3.startswith("208"): res.append("malignancy")
    if c3 in ("571","572","573") or c3.startswith("571") or c3.startswith("572") or c3.startswith("573"): res.append("severe_liver")
    if c3.startswith("196") or c3.startswith("197") or c3.startswith("198") or c3.startswith("199"): res.append("mets")
    if c3.startswith("042") or c3.startswith("043") or c3.startswith("044"): res.append("aids")
    return res

def main():
    # --- MIMIC features ---
    Xm, ym, cols, med = build_mimic_features()
    print(f"MIMIC: {len(Xm)} rows, {len(cols)} features, death {ym.mean():.3f}")

    # --- eICU features ---
    eicu = fetch_eicu()
    print(f"eICU: {len(eicu)} rows, death {eicu.outcome_death.mean():.3f}")
    ef = eicu.copy()
    ef["male"] = (ef["gender"] == "Male").astype(int)
    ef["race_white"] = (ef["ethnicity"] == "Caucasian").astype(int)
    # age: parse numeric, handle '>89' -> 90
    def parse_age(a):
        if pd.isna(a): return np.nan
        s = str(a).replace(">", "").strip()
        try: return float(s)
        except: return np.nan
    ef["age"] = ef["age"].apply(parse_age)
    for c in ["albumin", "bili_total", "lactate"]:
        ef[f"{c}_missing"] = ef[c].isna().astype(int)

    # --- Rebuild Charlson from ICD9 diagnoses ---
    dx = fetch_eicu_dx()
    print(f"eICU diagnosis rows: {len(dx)}")
    # split multi-codes (e.g. "428.0, I50.9"), keep ICD9 parts, map to Charlson
    ch_comorb = ["mi","chf","pvd","cvd","dementia","copd","rheum","pud","mild_liver",
                 "dm_nocc","dm_cc","paraplegia","renal","malignancy","severe_liver","mets","aids"]
    ch_flags = {c: {} for c in ch_comorb}  # patientunitstayid -> 1
    for _, row in dx.iterrows():
        sid = row["patientunitstayid"]
        codes = str(row["icd9code"]).split(",")
        for c in codes:
            for ch in icd9_charlson(c):
                ch_flags[ch][sid] = 1
    for ch in ch_comorb:
        ef[ch] = ef["patientunitstayid"].map(ch_flags[ch]).fillna(0).astype(int)
    # CCI (weighted, excluding age for now)
    cci = (ef["mi"] + ef["chf"] + ef["pvd"] + ef["cvd"] + ef["dementia"] + ef["copd"]
           + ef["rheum"] + ef["pud"] + ef["mild_liver"] + ef["dm_nocc"] + 2*ef["dm_cc"]
           + 2*ef["paraplegia"] + 2*ef["renal"] + 2*ef["malignancy"] + 3*ef["severe_liver"]
           + 6*ef["mets"] + 6*ef["aids"])
    ef["charlson_comorbidity_index"] = cci

    # ensure exact same column order as MIMIC (cols returned from build_mimic_features)
    Xe = ef[cols].fillna(med)
    ye = ef["outcome_death"].astype(int).values

    # --- Train RF on MIMIC 70% training split (frozen model, no test-set leak) ---
    Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=0.3, stratify=ym, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1)
    cv_auc = cross_val_score(rf, Xtr, ytr, cv=cv, scoring="roc_auc")
    rf.fit(Xtr, ytr)
    int_auc = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
    print(f"MIMIC CV AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(f"Internal 35-feature test AUC: {int_auc:.4f}")

    # --- Predict on eICU ---
    pe = rf.predict_proba(Xe)[:, 1]
    ae = roc_auc_score(ye, pe)
    pra = average_precision_score(ye, pe)
    br = brier_score_loss(ye, pe)
    print(f"eICU external: AUC {ae:.4f} | PR-AUC {pra:.4f} | Brier {br:.4f}")
    print(f"  eICU n={len(ye)}, death rate {ye.mean():.3f}, baseline PR {ye.mean():.4f}")

    # --- Figure: internal vs external ROC ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    # internal (MIMIC held-out) curve from the trained model
    fpr_int, tpr_int, _ = roc_curve(yte, rf.predict_proba(Xte)[:, 1])
    fpr_ext, tpr_ext, _ = roc_curve(ye, pe)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_int, tpr_int, lw=2, label=f"MIMIC internal 35f (AUC={int_auc:.3f})")
    ax.plot(fpr_ext, tpr_ext, lw=2, color="tab:orange", label=f"eICU external (AUC={ae:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("External validation: MIMIC internal vs eICU")
    ax.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "figures", "eicu_external.png"), dpi=600); plt.savefig(os.path.join(ROOT, "figures", "hires", "Figure3_external.pdf"), bbox_inches="tight"); plt.close()

    # --- Save ---
    joblib.dump(rf, os.path.join(ROOT, "models", "death_rf_common.pkl"))
    np.save(os.path.join(ROOT, "data", "eicu_y.npy"), ye)
    np.save(os.path.join(ROOT, "data", "eicu_prob.npy"), pe)
    pd.DataFrame({"cols": cols}).to_csv(os.path.join(ROOT, "data", "common_features.csv"), index=False)
    print("Saved: models/death_rf_common.pkl, data/eicu_y.npy, data/eicu_prob.npy")
    print("Saved figure: figures/eicu_external.png")

if __name__ == "__main__":
    main()
