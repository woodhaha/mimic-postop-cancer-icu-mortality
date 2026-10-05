"""Extract pre-surgery features + in-hospital mortality for malignancy-surgery ICU cohort.

Usage: python extract_surg_data.py [--out data.csv]
"""
import argparse, os, time
import pandas as pd
from google.cloud import bigquery

SQL = r"""
WITH mal AS (
  SELECT DISTINCT subject_id, hadm_id FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
  WHERE icd_version=10 AND (icd_code LIKE 'C%' OR icd_code LIKE 'D0%')
),
surg AS (
  SELECT subject_id, hadm_id, MIN(chartdate) AS surg_date
  FROM `physionet-data.mimiciv_3_1_hosp.procedures_icd`
  WHERE icd_version=10 AND icd_code LIKE '0%' GROUP BY subject_id, hadm_id
),
icu AS (
  SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime
  FROM `physionet-data.mimiciv_3_1_icu.icustays` i
  JOIN `physionet-data.mimiciv_3_1_derived.icustay_detail` d ON i.stay_id=d.stay_id
  WHERE d.first_icu_stay = TRUE
),
-- one row per FIRST ICU stay for malignancy-surgery admissions
cohort AS (
  SELECT DISTINCT m.subject_id, m.hadm_id, i.stay_id, s.surg_date
  FROM mal m
  JOIN surg s ON m.subject_id=s.subject_id AND m.hadm_id=s.hadm_id
  JOIN icu i ON m.subject_id=i.subject_id AND m.hadm_id=i.hadm_id
  WHERE s.surg_date <= DATE(i.intime)
),
-- demographics / admission
det AS (
  SELECT stay_id, gender, race, admission_age, los_hospital,
         hospital_expire_flag
  FROM `physionet-data.mimiciv_3_1_derived.icustay_detail`
),
-- Charlson comorbidities (hadm level, 100% coverage)
ch AS (
  SELECT hadm_id,
    myocardial_infarct, congestive_heart_failure, peripheral_vascular_disease,
    cerebrovascular_disease, dementia, chronic_pulmonary_disease, rheumatic_disease,
    peptic_ulcer_disease, mild_liver_disease, diabetes_without_cc, diabetes_with_cc,
    paraplegia, renal_disease, malignant_cancer, severe_liver_disease,
    metastatic_solid_tumor, aids, charlson_comorbidity_index
  FROM `physionet-data.mimiciv_3_1_derived.charlson`
),
-- pre-surgery labs from derived tables (standardized units), last value before surgery
chem AS (
  SELECT l.subject_id, l.hadm_id, l.charttime,
    l.albumin, l.creatinine, l.bicarbonate, l.glucose, l.sodium, l.potassium, l.bun
  FROM `physionet-data.mimiciv_3_1_derived.chemistry` l
  JOIN cohort c ON l.subject_id=c.subject_id AND l.hadm_id=c.hadm_id
  WHERE l.charttime < c.surg_date
),
cbc AS (
  SELECT l.subject_id, l.hadm_id, l.charttime,
    l.hemoglobin AS hgb, l.platelet AS platelets, l.wbc
  FROM `physionet-data.mimiciv_3_1_derived.complete_blood_count` l
  JOIN cohort c ON l.subject_id=c.subject_id AND l.hadm_id=c.hadm_id
  WHERE l.charttime < c.surg_date
),
coag AS (
  SELECT l.subject_id, l.hadm_id, l.charttime, l.inr, l.pt, l.ptt
  FROM `physionet-data.mimiciv_3_1_derived.coagulation` l
  JOIN cohort c ON l.subject_id=c.subject_id AND l.hadm_id=c.hadm_id
  WHERE l.charttime < c.surg_date
),
-- last pre-surgery value per admission (ROW_NUMBER)
prelab AS (
  SELECT subject_id, hadm_id, albumin, creatinine, bicarbonate, glucose, sodium, potassium, bun
  FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY subject_id, hadm_id ORDER BY charttime DESC) rn FROM chem)
  WHERE rn=1
),
precbc AS (
  SELECT subject_id, hadm_id, hgb, platelets, wbc
  FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY subject_id, hadm_id ORDER BY charttime DESC) rn FROM cbc)
  WHERE rn=1
),
precoag AS (
  SELECT subject_id, hadm_id, inr, pt, ptt
  FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY subject_id, hadm_id ORDER BY charttime DESC) rn FROM coag)
  WHERE rn=1
),
lactate_pre AS (
  SELECT l.subject_id, l.hadm_id, l.valuenum AS lactate
  FROM (SELECT l.*, ROW_NUMBER() OVER (PARTITION BY l.subject_id, l.hadm_id ORDER BY l.charttime DESC) rn
        FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
        JOIN cohort c ON l.subject_id=c.subject_id AND l.hadm_id=c.hadm_id
        WHERE l.itemid=50813 AND l.valuenum IS NOT NULL
          AND l.charttime < c.surg_date) l
  WHERE rn=1
),
bili_pre AS (
  SELECT l.subject_id, l.hadm_id, l.valuenum AS bili_total
  FROM (SELECT l.*, ROW_NUMBER() OVER (PARTITION BY l.subject_id, l.hadm_id ORDER BY l.charttime DESC) rn
        FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
        JOIN cohort c ON l.subject_id=c.subject_id AND l.hadm_id=c.hadm_id
        WHERE l.itemid=50885 AND l.valuenum IS NOT NULL
          AND l.charttime < c.surg_date) l
  WHERE rn=1
),
-- surgery features: count + type from procedure codes
surgfeat AS (
  SELECT s.subject_id, s.hadm_id,
    COUNT(*) AS n_procedures,
    MAX(CASE WHEN s.icd_code LIKE '0FY00%' THEN 1 ELSE 0 END) AS liver_transplant,
    MAX(CASE WHEN s.icd_code LIKE '0FT0%' THEN 1 ELSE 0 END) AS liver_resection,
    MAX(CASE WHEN s.icd_code LIKE '0FBG0ZZ' THEN 1 ELSE 0 END) AS open_pancreatic_excision,
    MAX(CASE WHEN s.icd_code LIKE '0FBG%' OR s.icd_code LIKE '0FBH%' THEN 1 ELSE 0 END) AS any_pancreatic_excision,
    MAX(CASE WHEN s.icd_code LIKE '0DB0%' OR s.icd_code LIKE '0DB1%' OR s.icd_code LIKE '0DB2%' OR s.icd_code LIKE '0DT0%' THEN 1 ELSE 0 END) AS GI_resection,
    MAX(CASE WHEN s.icd_code LIKE '0B%' THEN 1 ELSE 0 END) AS thoracic_surgery,
    MAX(CASE WHEN s.icd_code LIKE '0H%' OR s.icd_code LIKE '0J%' THEN 1 ELSE 0 END) AS ortho_surgery
  FROM `physionet-data.mimiciv_3_1_hosp.procedures_icd` s
  JOIN cohort c ON s.subject_id=c.subject_id AND s.hadm_id=c.hadm_id
  WHERE s.icd_version=10 AND s.icd_code LIKE '0%' AND s.chartdate <= c.surg_date
  GROUP BY s.subject_id, s.hadm_id
)
SELECT
  c.subject_id, c.hadm_id, c.stay_id,
  d.gender, d.race, d.admission_age, d.los_hospital,
  d.hospital_expire_flag AS outcome_death,
  ch.* EXCEPT(hadm_id),
  p.albumin, p.creatinine, p.bicarbonate, p.glucose, p.sodium, p.potassium, p.bun,
  pc.hgb, pc.platelets, pc.wbc,
  cg.inr, cg.pt, cg.ptt,
  lp.lactate, bp.bili_total,
  f.n_procedures, f.liver_transplant, f.liver_resection,
  f.open_pancreatic_excision, f.any_pancreatic_excision,
  f.GI_resection, f.thoracic_surgery, f.ortho_surgery
FROM cohort c
LEFT JOIN det d ON c.stay_id=d.stay_id
LEFT JOIN ch ON c.hadm_id=ch.hadm_id
LEFT JOIN prelab p ON c.subject_id=p.subject_id AND c.hadm_id=p.hadm_id
LEFT JOIN precbc pc ON c.subject_id=pc.subject_id AND c.hadm_id=pc.hadm_id
LEFT JOIN precoag cg ON c.subject_id=cg.subject_id AND c.hadm_id=cg.hadm_id
LEFT JOIN lactate_pre lp ON c.subject_id=lp.subject_id AND c.hadm_id=lp.hadm_id
LEFT JOIN bili_pre bp ON c.subject_id=bp.subject_id AND c.hadm_id=bp.hadm_id
LEFT JOIN surgfeat f ON c.subject_id=f.subject_id AND c.hadm_id=f.hadm_id
"""

def main():
    ap = argparse.ArgumentParser()
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "surg_data.csv"))
    args = ap.parse_args()

    client = bigquery.Client()
    print("Running query...")
    t0 = time.time()
    df = client.query(SQL).to_dataframe()
    print(f"Fetched {len(df)} rows in {time.time()-t0:.1f}s")

    df.to_csv(args.out, index=False)
    print(f"Saved to {args.out}")
    print(f"Columns: {list(df.columns)}")
    print(f"Outcome death: {df['outcome_death'].sum()} / {len(df)} ({100*df['outcome_death'].mean():.1f}%)")

    # sanity: lab medians must fall in physiological ranges (catches unit/itemid mixups)
    ranges = {"bili_total": (0.2, 1.5), "albumin": (2.0, 5.0), "hgb": (7.0, 18.0),
              "lactate": (0.5, 6.0), "creatinine": (0.3, 4.0), "platelets": (50, 500)}
    for col, (lo, hi) in ranges.items():
        if col in df.columns:
            med = df[col].dropna().median()
            ok = lo <= med <= hi
            print(f"  check {col}: median={med:.2f} expected [{lo}-{hi}] {'OK' if ok else '!! FAIL'}")
            if not ok:
                raise SystemExit(f"Sanity check failed: {col} median {med} outside [{lo}-{hi}]")

if __name__ == "__main__":
    main()
