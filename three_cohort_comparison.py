"""Three-cohort comparison table (baseline + validation performance).
MIMIC-IV (development), eICU (external), MIMIC-III (temporal external).
Run from repo root: python scripts/three_cohort_comparison.py
"""
import pandas as pd, numpy as np, os
from google.cloud import bigquery
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import warnings
warnings.filterwarnings("ignore")
client = bigquery.Client()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def build_baseline():
    # MIMIC-IV
    df4 = pd.read_csv(os.path.join(ROOT, "data", "surg_data.csv"))
    # MIMIC-III
    df3 = pd.read_csv(os.path.join(ROOT, "data", "mimiciii_cohort.csv"))
    hadm3 = df3.hadm_id.tolist()
    ages = client.query("""SELECT a.hadm_id, DATE_DIFF(DATE(a.admittime), DATE(p.dob), YEAR) age
        FROM `physionet-data.mimiciii_clinical.admissions` a
        JOIN `physionet-data.mimiciii_clinical.patients` p ON a.subject_id=p.subject_id
        WHERE a.hadm_id IN UNNEST(@h)""",
        job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter('h','INT64',hadm3)])).result().to_dataframe()
    ages['age'] = np.where(ages.age>89, 90, ages.age)
    df3 = df3.merge(ages, on='hadm_id')
    labs3 = client.query("""SELECT l.hadm_id, MAX(l.CREATININE) cre, MAX(l.ALBUMIN) alb, MAX(l.BILIRUBIN) bili
        FROM `physionet-data.mimiciii_derived.pivoted_lab` l WHERE l.hadm_id IN UNNEST(@h) GROUP BY l.hadm_id""",
        job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter('h','INT64',hadm3)])).result().to_dataframe()
    df3 = df3.merge(labs3, on='hadm_id')
    # eICU
    e = client.query("""WITH mal AS (SELECT DISTINCT patientunitstayid FROM `physionet-data.eicu_crd.diagnosis`
        WHERE REGEXP_CONTAINS(icd9code, r'^(14[0-9]|15[0-9]|16[0-9]|17[0-9]|18[0-9]|19[0-9]|20[0-9])'))
        SELECT p.patientunitstayid, p.gender, p.age,
        CASE WHEN p.hospitaldischargestatus='Expired' THEN 1 ELSE 0 END death
        FROM mal m JOIN `physionet-data.eicu_crd.patient` p ON m.patientunitstayid=p.patientunitstayid""").result().to_dataframe()
    el = client.query("""WITH mal AS (SELECT DISTINCT patientunitstayid FROM `physionet-data.eicu_crd.diagnosis`
        WHERE REGEXP_CONTAINS(icd9code, r'^(14[0-9]|15[0-9]|16[0-9]|17[0-9]|18[0-9]|19[0-9]|20[0-9])'))
        SELECT l.patientunitstayid,
        MAX(CASE WHEN l.labname='creatinine' THEN l.labresult END) cre,
        MAX(CASE WHEN l.labname='albumin' THEN l.labresult END) alb,
        MAX(CASE WHEN l.labname='total bilirubin' THEN l.labresult END) bili
        FROM `physionet-data.eicu_crd.lab` l JOIN mal m ON l.patientunitstayid=m.patientunitstayid
        WHERE l.labresult IS NOT NULL GROUP BY l.patientunitstayid""").result().to_dataframe()
    e = e.merge(el, on='patientunitstayid', how='left')
    e['age'] = pd.to_numeric(e.age.astype(str).str.replace('>',''), errors='coerce')
    e = e[e.age>=18]

    rows = [
        {'cohort':'MIMIC-IV (development)','period':'2008-2022','coding':'ICD-10','n':len(df4),
         'death':f"{100*df4.outcome_death.mean():.1f}%",'age':f"{df4.admission_age.median():.0f}",
         'male':f"{100*(df4.gender=='M').mean():.0f}%",
         'creatinine':f"{df4.creatinine.median():.2f}",'albumin':f"{df4.albumin.median():.1f}",'bilirubin':f"{df4.bili_total.median():.1f}"},
        {'cohort':'MIMIC-III (temporal external)','period':'2001-2012','coding':'ICD-9','n':len(df3[df3.age>=18]),
         'death':f"{100*df3[df3.age>=18].outcome_death.mean():.1f}%",'age':f"{df3[df3.age>=18].age.median():.0f}",
         'male':f"{100*(df3[df3.age>=18].gender=='M').mean():.0f}%",
         'creatinine':f"{df3[df3.age>=18].cre.median():.2f}",
         'albumin':f"{df3[df3.age>=18].alb.median():.1f}",'bilirubin':f"{df3[df3.age>=18].bili.median():.1f}"},
        {'cohort':'eICU (external)','period':'2014-2015','coding':'ICD-9','n':len(e),
         'death':f"{100*e.death.mean():.1f}%",'age':f"{e.age.median():.0f}",
         'male':f"{100*(e.gender=='Male').mean():.0f}%",
         'creatinine':f"{e.cre.median():.2f}",
         'albumin':f"{e.alb.median():.1f}",'bilirubin':f"{e.bili.median():.1f}"},
    ]
    return pd.DataFrame(rows)

def build_performance():
    # MIMIC-IV internal (35-feature, for fair comparison) from validation_three_cohorts computation
    # Internal 35f = 0.671; eICU = 0.655; MIMIC-III = 0.635
    ye1 = np.load(os.path.join(ROOT,"data","eicu_y.npy")); pe1 = np.load(os.path.join(ROOT,"data","eicu_prob.npy"))
    ye3 = np.load(os.path.join(ROOT,"data","mimiciii_y.npy")); pe3 = np.load(os.path.join(ROOT,"data","mimiciii_prob.npy"))
    rows = [
        {'cohort':'MIMIC-IV internal (35-feature)','AUC':0.671,'PR-AUC':None,'Brier':None,'type':'Development/internal'},
        {'cohort':'eICU external','AUC':round(roc_auc_score(ye1,pe1),3),'PR-AUC':round(average_precision_score(ye1,pe1),3),'Brier':round(brier_score_loss(ye1,pe1),3),'type':'External'},
        {'cohort':'MIMIC-III temporal external','AUC':round(roc_auc_score(ye3,pe3),3),'PR-AUC':round(average_precision_score(ye3,pe3),3),'Brier':round(brier_score_loss(ye3,pe3),3),'type':'External (temporal)'},
    ]
    return pd.DataFrame(rows)

baseline = build_baseline()
perf = build_performance()
print("=== BASELINE ===")
print(baseline.to_string(index=False))
print("\n=== PERFORMANCE ===")
print(perf.to_string(index=False))
baseline.to_csv(os.path.join(ROOT,"reports","three_cohort_baseline.csv"), index=False)
perf.to_csv(os.path.join(ROOT,"reports","three_cohort_performance.csv"), index=False)
print("\nSaved reports/three_cohort_baseline.csv + three_cohort_performance.csv")
