"""Verify MIMIC-IV access: PhysioNet credentialing + BigQuery.

Usage: python verify_mimic_access.py
"""
import subprocess, sys

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                          shell=True)

def bq(sql):
    r = run(f'bq query --use_legacy_sql=false --max_rows=1 "{sql}"')
    return r.stdout.strip() + r.stderr.strip()

print("=== 1. gcloud account ===")
print(run("gcloud auth list").stdout.strip())

print("\n=== 2. MIMIC-IV patients count ===")
print(bq("SELECT count(*) AS n FROM `physionet-data.mimiciv_3_1_hosp.patients`"))

print("\n=== 3. MIMIC-IV ICU admissions count ===")
print(bq("SELECT count(*) AS n FROM `physionet-data.mimiciv_3_1_icu.icustays`"))
