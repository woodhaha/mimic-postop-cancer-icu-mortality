# Pre-operative prediction of in-hospital mortality in postoperative cancer ICU patients

Analysis code for: *Pre-operative Prediction of In-Hospital Mortality in Cancer Patients After Surgery Requiring ICU Admission: Development and Reduced-Feature External Transferability Analysis of a Machine Learning Model*.

## Overview
Random Forest model predicting in-hospital mortality in postoperative cancer surgery patients admitted to the ICU, using only pre-operative data.
- **Development**: MIMIC-IV (n=3,801; 18.8% mortality), 49 pre-operative features
- **External transferability**: eICU-CRD (general oncology ICU) and MIMIC-III (temporal, same center)
- Internal test AUC 0.728; external AUC 0.655 (eICU) / 0.635 (MIMIC-III)

## Requirements
- Python 3.10+, `pandas`, `scikit-learn`, `google-cloud-bigquery`, `shap`, `matplotlib`
- Google Cloud credentials with access to the `physionet-data` BigQuery project

## Data access
The datasets are available through PhysioNet with credentialed access (DUA required):
- MIMIC-IV: https://physionet.org/content/mimiciv/
- MIMIC-III: https://physionet.org/content/mimiciii/
- eICU-CRD: https://physionet.org/content/eicu-crd/

**Patient-level data are NOT redistributed in this repository** (per the MIMIC/eICU data use agreements).

## Reproduction
```bash
# 1. Extract the development cohort (BigQuery)
python scripts/extract_surg_data.py --out data/surg_data.csv

# 2. Train the internal model + generate figures
python scripts/final_rf_model.py

# 3. External validation
python scripts/extract_mimiciii_cohort.py
python scripts/mimiciii_external_validation.py
python scripts/eicu_external_validation.py

# 4. Calibration, CIs, sensitivity
python scripts/external_calibration.py
python scripts/bootstrap_metrics.py
python scripts/missing_data_sensitivity.py
```

## Key scripts
- `extract_surg_data.py` — cohort definition (malignancy C/D0, surgical root operation, surgery-before-ICU, strict pre-operative labs)
- `final_rf_model.py` — internal 49-feature Random Forest (AUC 0.728, calibration slope 0.691)
- `eicu_external_validation.py` / `mimiciii_external_validation.py` — frozen 35-feature external transferability
- `external_calibration.py`, `bootstrap_metrics.py`, `missing_data_sensitivity.py` — calibration, bootstrap CIs, imputation sensitivity

## License
Code available for research use. Data subject to PhysioNet DUA.
