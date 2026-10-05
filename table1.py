"""Generate Table 1: baseline characteristics, death vs alive."""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "surg_data.csv"))

y = df["outcome_death"]
grp_alive, grp_dead = df[y == 0], df[y == 1]
n_alive, n_dead = len(grp_alive), len(grp_dead)
print(f"Alive n={n_alive}, Dead n={n_dead}")

rows = []

def add_cont(name, col, unit=""):
    a = grp_alive[col].dropna()
    d = grp_dead[col].dropna()
    p = stats.mannwhitneyu(a, d).pvalue if len(a) and len(d) else np.nan
    rows.append({
        "Variable": name, "Alive (n=%d)" % n_alive: f"{a.median():.2f} ({a.quantile(.25):.2f}-{a.quantile(.75):.2f})",
        "Dead (n=%d)" % n_dead: f"{d.median():.2f} ({d.quantile(.25):.2f}-{d.quantile(.75):.2f})",
        "P": "%.3f" % p if p == p else "-",
    })

def add_cat(name, col):
    # binary or categorical: show n(%)
    if col in df.columns and df[col].dtype.kind in "iuf":
        a, d = grp_alive[col].fillna(0), grp_dead[col].fillna(0)
        # binary
        if a.nunique() <= 3:
            na, nd = int(a.sum()), int(d.sum())
            rows.append({
                "Variable": name,
                "Alive (n=%d)" % n_alive: f"{na} ({100*na/len(a):.1f})",
                "Dead (n=%d)" % n_dead: f"{nd} ({100*nd/len(d):.1f})",
                "P": "%.3f" % stats.chi2_contingency([[na, len(a)-na], [nd, len(d)-nd]])[1],
            })
            return
    # categorical string
    vals = sorted(set(df[col].dropna()))
    if len(vals) > 12: vals = vals[:6] + ["..."]
    for v in vals:
        na = (grp_alive[col] == v).sum(); nd = (grp_dead[col] == v).sum()
        rows.append({
            "Variable": f"{name}: {v}",
            "Alive (n=%d)" % n_alive: f"{na} ({100*na/n_alive:.1f})",
            "Dead (n=%d)" % n_dead: f"{nd} ({100*nd/n_dead:.1f})",
            "P": "%.3f" % stats.chi2_contingency([[na, n_alive-na], [nd, n_dead-nd]])[1] if (n_alive+n_dead) else "-",
        })

# --- Demographics ---
add_cont("Age (years)", "admission_age")
add_cont("Hospital LOS (days)", "los_hospital")
add_cat("Gender", "gender")

# --- Surgery type ---
for col, name in [("thoracic_surgery","Thoracic surgery"),("liver_transplant","Liver transplant"),
                  ("open_pancreatic_excision","Open pancreatic excision"),("any_pancreatic_excision","Any pancreatic excision"),
                  ("GI_resection","GI resection"),("ortho_surgery","Orthopedic surgery"),
                  ("liver_resection","Liver resection")]:
    if col in df.columns:
        add_cat(name, col)
add_cont("N procedures", "n_procedures")

# --- Charlson comorbidities ---
comorb = [("myocardial_infarct","Myocardial infarction"),("congestive_heart_failure","CHF"),
          ("peripheral_vascular_disease","Peripheral vascular disease"),("cerebrovascular_disease","Cerebrovascular disease"),
          ("dementia","Dementia"),("chronic_pulmonary_disease","Chronic pulmonary disease"),
          ("rheumatic_disease","Rheumatic disease"),("peptic_ulcer_disease","Peptic ulcer"),
          ("mild_liver_disease","Mild liver disease"),("diabetes_without_cc","Diabetes w/o cc"),
          ("diabetes_with_cc","Diabetes with cc"),("paraplegia","Paraplegia"),
          ("renal_disease","Renal disease"),("severe_liver_disease","Severe liver disease"),
          ("metastatic_solid_tumor","Metastatic solid tumor"),("aids","AIDS")]
for col, name in comorb:
    if col in df.columns:
        add_cat(name, col)
add_cont("Charlson index", "charlson_comorbidity_index")

# --- Pre-surgery labs ---
for col, name in [("creatinine","Creatinine (mg/dL)"),("albumin","Albumin (g/dL)"),
                  ("bili_total","Total bilirubin (mg/dL)"),("hgb","Hemoglobin (g/dL)"),
                  ("platelets","Platelets (x10^9/L)"),("wbc","WBC (x10^9/L)"),
                  ("lactate","Lactate (mmol/L)"),("ptt","PTT (s)"),("inr","INR"),
                  ("sodium","Sodium (mEq/L)"),("potassium","Potassium (mEq/L)"),
                  ("glucose","Glucose (mg/dL)")]:
    if col in df.columns:
        add_cont(name, col)

tab = pd.DataFrame(rows)
print("\n" + tab.to_string(index=False))

os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
tab.to_csv(os.path.join(ROOT, "reports", "table1.csv"), index=False)
tab.to_markdown(os.path.join(ROOT, "reports", "table1.md"), index=False)
print("\nSaved reports/table1.csv and reports/table1.md")
