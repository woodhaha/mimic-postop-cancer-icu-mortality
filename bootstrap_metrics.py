"""Bootstrap CIs (external AUC, internal PR-AUC/Brier) and isotonic recalibration of external predictions.
Run from repo root: python scripts/bootstrap_metrics.py
"""
import numpy as np, os
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.RandomState(42)

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def calib_slope(y, p):
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return lr.coef_[0][0], lr.intercept_[0]

def bootstrap_auc_ci(y, p, n=1000):
    aucs = []; idx = np.arange(len(y))
    for _ in range(n):
        i = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[i])) < 2: continue
        aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])

# --- external: AUC CI + isotonic recalibration ---
for name, yf, pf in [("eICU", "data/eicu_y.npy", "data/eicu_prob.npy"),
                     ("MIMIC-III", "data/mimiciii_y.npy", "data/mimiciii_prob.npy")]:
    y = np.load(os.path.join(ROOT, yf)); p = np.load(os.path.join(ROOT, pf))
    ci = bootstrap_auc_ci(y, p)
    slope, _ = calib_slope(y, p)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    p_iso = iso.predict(p)
    slope_iso, _ = calib_slope(y, p_iso)
    auc_iso = roc_auc_score(y, p_iso)
    print(f"{name}: AUC {roc_auc_score(y,p):.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f}) | "
          f"slope {slope:.3f} -> isotonic {slope_iso:.3f} | recalib AUC {auc_iso:.3f}")

# --- internal: PR-AUC + Brier bootstrap CI ---
y = np.load(os.path.join(ROOT, "data/y_test.npy")); p = np.load(os.path.join(ROOT, "data/y_prob.npy"))
idx = np.arange(len(y)); pras = []; briers = []
for _ in range(1000):
    i = rng.choice(idx, len(idx), replace=True)
    if len(np.unique(y[i])) < 2: continue
    pras.append(average_precision_score(y[i], p[i]))
    briers.append(brier_score_loss(y[i], p[i]))
pra_ci = np.percentile(pras, [2.5, 97.5]); brier_ci = np.percentile(briers, [2.5, 97.5])
print(f"Internal: PR-AUC {average_precision_score(y,p):.3f} (95% CI {pra_ci[0]:.3f}-{pra_ci[1]:.3f}) | "
      f"Brier {brier_score_loss(y,p):.3f} (95% CI {brier_ci[0]:.3f}-{brier_ci[1]:.3f})")
