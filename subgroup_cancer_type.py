"""Cancer-type subgroup analysis by true ICD-10 diagnosis codes.
Replaces the surgery-type approximation with real cancer-site grouping.
Run from repo root: python scripts/subgroup_cancer_type.py
"""
import pandas as pd, numpy as np, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from google.cloud import bigquery
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 1. Extract ICD-10 cancer diagnoses for cohort (cache to avoid re-query) ---
dx_cache = os.path.join(ROOT, "data", "dx_codes.csv")
if not os.path.exists(dx_cache):
    client = bigquery.Client()
    df = pd.read_csv(os.path.join(ROOT, "data", "surg_data.csv"))
    hadm = df.hadm_id.tolist()
    sql = '''
    SELECT d.hadm_id, d.icd_code
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
    WHERE d.hadm_id IN UNNEST(@hadm) AND d.icd_version=10
    '''
    dx = client.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter('hadm','INT64',hadm)]
    )).result().to_dataframe()
    dx.to_csv(dx_cache, index=False)
    print(f"Extracted {len(dx)} diagnosis records -> {dx_cache}")
else:
    dx = pd.read_csv(dx_cache)
    print(f"Loaded cached {len(dx)} diagnosis records")

# --- 2. Map ICD-10 codes to cancer site ---
def cancer_site(code):
    c3 = code[:3]
    if c3 in ('C34','C33'): return 'lung'
    if c3 in ('C15','C16'): return 'upper_gi'
    if c3 in ('C18','C19','C20','C21'): return 'colorectal'
    if c3 in ('C22','C23','C24','C25'): return 'hepato_pancreatic'
    if c3 in ('C50',): return 'breast'
    if c3 in ('C61','C62','C63'): return 'prostate'
    if c3 in ('C64','C65','C66','C67','C68'): return 'renal_urothelial'
    if c3 in ('C70','C71','C72'): return 'brain'
    if c3 in ('C81','C82','C83','C84','C85','C86','C90','C91','C92','C93','C94','C95','C96'): return 'hematologic'
    return 'other'

dx['site'] = dx.icd_code.apply(cancer_site)
hadm_site = dx[dx.site!='other'].groupby('hadm_id')['site'].agg(
    lambda x: x.value_counts().idxmax() if len(x)>0 else 'other')

# --- 3. Build features (same as final model) ---
df = pd.read_csv(os.path.join(ROOT, "data", "surg_data.csv"))
merged = df.merge(hadm_site.rename('cancer_group'), on='hadm_id', how='left')
merged['cancer_group'] = merged['cancer_group'].fillna('no_primary_code')

# three-category: lung (thoracic), GI, other (mapped non-lung/GI); unmapped excluded
def three_group(g):
    if g=='lung': return 'lung'
    if g in ('upper_gi','colorectal','hepato_pancreatic'): return 'GI'
    if g in ('breast','prostate','renal_urothelial','brain','hematologic'): return 'other'
    return 'excluded'  # no_primary_code -> dropped from subgroup comparison
merged['grp3'] = merged.cancer_group.apply(three_group)

feat = merged.copy(); feat['male']=(feat['gender']=='M').astype(int)
for r in ['black','hispanic','asian','other','white']:
    if r=='other': feat['race_other']=feat['race'].isin(['OTHER','UNKNOWN','UNKNOWN/NOT SPECIFIED']).astype(int)
    else: feat[f'race_{r}']=feat['race'].str.contains(r.upper(),na=False).astype(int)
for c in ['albumin','hgb','bili_total','lactate']: feat[f'{c}_missing']=feat[c].isna().astype(int)
lab=['creatinine','albumin','bili_total','hgb','platelets','wbc','lactate','ptt','inr','sodium','potassium','glucose']
comorb=['myocardial_infarct','congestive_heart_failure','peripheral_vascular_disease','cerebrovascular_disease','dementia','chronic_pulmonary_disease','rheumatic_disease','peptic_ulcer_disease','mild_liver_disease','diabetes_without_cc','diabetes_with_cc','paraplegia','renal_disease','malignant_cancer','severe_liver_disease','metastatic_solid_tumor','aids','charlson_comorbidity_index']
surg=['n_procedures','liver_transplant','liver_resection','open_pancreatic_excision','any_pancreatic_excision','GI_resection','thoracic_surgery','ortho_surgery']
demo=['admission_age','male','race_black','race_hispanic','race_asian','race_other','race_white']
fc=lab+[f'{c}_missing' for c in ['albumin','hgb','bili_total','lactate']]+comorb+surg+demo
X=feat[fc].apply(lambda c:c.fillna(c.median()) if c.dtype.kind in 'fiu' else c)
y=feat['outcome_death'].astype(int).values
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)
rf=joblib.load(os.path.join(ROOT,"models","death_rf.pkl")); p=rf.predict_proba(Xte)[:,1]
te_idx = Xte.index
yte_g = feat.loc[te_idx,'grp3']

# --- 4. Subgroup AUCs with bootstrap CI ---
print("\n=== Cancer-type subgroup (test set) ===")
rows = []
rng = np.random.RandomState(42)
for g in ['lung','GI','other']:
    m = (yte_g==g).values
    if m.sum() < 30: 
        print(f"{g}: n={m.sum()} (too small)"); continue
    yg = yte[m]; pg = p[m]
    auc = roc_auc_score(yg, pg) if len(np.unique(yg))>1 else float('nan')
    idxg = np.arange(len(yg)); aucs=[]
    for _ in range(1000):
        i = rng.choice(idxg, len(idxg), replace=True)
        if len(np.unique(yg[i]))<2: continue
        aucs.append(roc_auc_score(yg[i], pg[i]))
    ci = np.percentile(aucs,[2.5,97.5]) if len(aucs)>10 else (np.nan,np.nan)
    rows.append({"subgroup":g, "n":int(m.sum()), "mortality":round(100*yg.mean(),1),
                 "auc":round(auc,3), "ci_low":round(ci[0],3), "ci_high":round(ci[1],3)})
    print(f"{g}: n={m.sum()}, death={100*yg.mean():.0f}%, AUC={auc:.3f} ({ci[0]:.2f}-{ci[1]:.2f})")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(ROOT,"reports","subgroup_cancer_auc.csv"), index=False)
print("\nSaved: reports/subgroup_cancer_auc.csv")

# --- 5. Save merged data for reproducibility ---
merged.to_csv(os.path.join(ROOT,"data","surg_data_with_cancer_type.csv"), index=False)
print("Saved: data/surg_data_with_cancer_type.csv")
