"""Extract MIMIC-III oncology-surgery ICU cohort for external validation.
MIMIC-III uses ICD-9 codes and UPPERCASE column names. Matches MIMIC-IV
cohort: malignancy + surgery + first ICU admission.
Run from repo root: python scripts/extract_mimiciii_cohort.py
"""
from google.cloud import bigquery
import pandas as pd, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
client = bigquery.Client()

SQL = r"""
WITH mal AS (
  SELECT DISTINCT SUBJECT_ID, HADM_ID
  FROM `physionet-data.mimiciii_clinical.diagnoses_icd`
  WHERE ICD9_CODE IS NOT NULL AND (
    ICD9_CODE LIKE '14%' OR ICD9_CODE LIKE '15%' OR ICD9_CODE LIKE '16%' OR
    ICD9_CODE LIKE '17%' OR ICD9_CODE LIKE '18%' OR ICD9_CODE LIKE '19%' OR
    ICD9_CODE LIKE '20%'
  )
),
surg AS (
  SELECT DISTINCT SUBJECT_ID, HADM_ID
  FROM `physionet-data.mimiciii_clinical.procedures_icd`
  WHERE ICD9_CODE IS NOT NULL
    AND SUBSTR(ICD9_CODE, 1, 2) BETWEEN '01' AND '86'
),
icu_first AS (
  SELECT ic.* FROM `physionet-data.mimiciii_clinical.icustays` ic
  JOIN (
    SELECT SUBJECT_ID, MIN(ICUSTAY_ID) AS first_stay
    FROM `physionet-data.mimiciii_clinical.icustays`
    GROUP BY SUBJECT_ID
  ) f ON ic.SUBJECT_ID=f.SUBJECT_ID AND ic.ICUSTAY_ID=f.first_stay
),
cohort AS (
  SELECT DISTINCT m.SUBJECT_ID, m.HADM_ID, i.ICUSTAY_ID AS STAY_ID
  FROM mal m
  JOIN surg s ON m.SUBJECT_ID=s.SUBJECT_ID AND m.HADM_ID=s.HADM_ID
  JOIN icu_first i ON m.SUBJECT_ID=i.SUBJECT_ID AND m.HADM_ID=i.HADM_ID
)
SELECT c.SUBJECT_ID AS subject_id, c.HADM_ID AS hadm_id, c.STAY_ID AS stay_id,
  a.HOSPITAL_EXPIRE_FLAG AS outcome_death,
  a.ADMISSION_TYPE AS admission_type,
  p.GENDER AS gender,
  a.ETHNICITY AS ethnicity,
  DATE_DIFF(DATE(a.DISCHTIME), DATE(a.ADMITTIME), DAY) AS los_hospital,
  i.INTIME AS icu_intime
FROM cohort c
JOIN `physionet-data.mimiciii_clinical.admissions` a ON c.HADM_ID=a.HADM_ID
JOIN `physionet-data.mimiciii_clinical.patients` p ON c.SUBJECT_ID=p.SUBJECT_ID
JOIN `physionet-data.mimiciii_clinical.icustays` i ON c.STAY_ID=i.ICUSTAY_ID
"""

print("Running MIMIC-III cohort extraction...")
df = client.query(SQL).to_dataframe()
print(f"Extracted {len(df)} stays")
print(f"Death: {df.outcome_death.sum()} ({100*df.outcome_death.mean():.1f}%)")
print(f"Admission types: {df.admission_type.value_counts().to_dict()}")
df.to_csv(os.path.join(ROOT, "data", "mimiciii_cohort.csv"), index=False)
print("Saved: data/mimiciii_cohort.csv")
