"""DeepSurv (Cox deep survival) benchmark for death prediction.
DeepSurv = neural network on Cox negative partial log-likelihood.
Outputs risk scores; evaluated via AUC on death outcome.
Run from repo root: python scripts/deepsurv.py
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- build features (same as train_death_model) ---
df = pd.read_csv(os.path.join(ROOT, "data", "surg_data.csv"))
feat = df.copy()
feat["male"] = (feat["gender"] == "M").astype(int)
for r in ["black","hispanic","asian","other","white"]:
    if r=="other":
        feat["race_other"] = feat["race"].isin(["OTHER","UNKNOWN","UNKNOWN/NOT SPECIFIED"]).astype(int)
    else:
        feat[f"race_{r}"] = feat["race"].str.contains(r.upper(), na=False).astype(int)
feat["charlson_comorbidity_index"] = pd.to_numeric(feat["charlson_comorbidity_index"], errors="coerce")
for col in ["albumin","hgb","bili_total","lactate"]:
    feat[f"{col}_missing"] = feat[col].isna().astype(int)

lab = ["creatinine","albumin","bili_total","hgb","platelets","wbc","lactate","ptt","inr","sodium","potassium","glucose"]
comorb = ["myocardial_infarct","congestive_heart_failure","peripheral_vascular_disease","cerebrovascular_disease",
          "dementia","chronic_pulmonary_disease","rheumatic_disease","peptic_ulcer_disease","mild_liver_disease",
          "diabetes_without_cc","diabetes_with_cc","paraplegia","renal_disease","malignant_cancer",
          "severe_liver_disease","metastatic_solid_tumor","aids","charlson_comorbidity_index"]
surg = ["n_procedures","liver_transplant","liver_resection","open_pancreatic_excision","any_pancreatic_excision",
        "GI_resection","thoracic_surgery","ortho_surgery"]
demo = ["admission_age","male","race_black","race_hispanic","race_asian","race_other","race_white"]
feat_cols = lab + [f"{c}_missing" for c in ["albumin","hgb","bili_total","lactate"]] + comorb + surg + demo

X = feat[feat_cols].apply(lambda c: c.fillna(c.median()) if c.dtype.kind in "fiu" else c).values
y = feat["outcome_death"].astype(int).values

# --- survival time: in-hospital time (days from admission); censored = discharge alive ---
adm = df["admission_age"]  # placeholder to avoid NaN; times built from admissions via join not in csv
# Use hospital LOS as survival proxy: dead -> time=los, event=1; alive -> time=los, event=0
los = df["los_hospital"].fillna(df["los_hospital"].median()).values
T = np.where(y == 1, los, los).astype(np.float32)
E = y.astype(np.float32)

Xtr, Xte, ytr, yte, Ttr, Tte, Etr, Ete = train_test_split(
    X, y, T, E, test_size=0.3, stratify=y, random_state=42)

scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)
Xte_s = scaler.transform(Xte).astype(np.float32)

class DeepSurv(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def cox_loss(risk, T, E, device):
    """Negative Cox partial log-likelihood with batch approximation."""
    # sort descending by time
    order = torch.argsort(T, descending=True)
    r, t, e = risk[order], T[order], E[order]
    # log cumulative sum of exp(risk) for risk set (descending time)
    exp_r = torch.exp(r)
    logcum = torch.logcumsumexp(exp_r, dim=0)
    loss = -torch.sum(e * (r - logcum)) / (e.sum() + 1e-8)
    return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

torch.manual_seed(42)
model = DeepSurv(Xtr_s.shape[1]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

Xtr_t = torch.tensor(Xtr_s, device=device)
Ttr_t = torch.tensor(Ttr, device=device)
Etr_t = torch.tensor(Etr, device=device)

losses = []
for epoch in range(200):
    model.train()
    opt.zero_grad()
    risk = model(Xtr_t)
    loss = cox_loss(risk, Ttr_t, Etr_t, device)
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if (epoch+1) % 50 == 0:
        print(f"  epoch {epoch+1}: loss {loss.item():.4f}")

# --- evaluate risk scores ---
model.eval()
with torch.no_grad():
    risk_te = model(torch.tensor(Xte_s, device=device)).cpu().numpy()
    risk_tr = model(torch.tensor(Xtr_s, device=device)).cpu().numpy()

auc = roc_auc_score(yte, risk_te)
prauc = average_precision_score(yte, risk_te)
auc_tr = roc_auc_score(ytr, risk_tr)
print(f"\nDeepSurv: Train AUC {auc_tr:.4f} | Test AUC {auc:.4f} | Test PR-AUC {prauc:.4f}")

np.save(os.path.join(ROOT, "data", "deepsurv_risk_test.npy"), risk_te)
print("Saved data/deepsurv_risk_test.npy")
