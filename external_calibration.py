"""External calibration for eICU and MIMIC-III frozen-model predictions.
Computes calibration slope (logistic regression of outcome on logit prediction),
intercept, Brier, and calibration-in-the-large, plus calibration curves.
Run from repo root: python scripts/external_calibration.py
"""
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def calib_metrics(y, p):
    lp = _logit(p).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(lp, y)
    return dict(
        auc=roc_auc_score(y, p),
        pra=average_precision_score(y, p),
        brier=brier_score_loss(y, p),
        slope=lr.coef_[0][0],
        intercept=lr.intercept_[0],
        mean_pred=p.mean(),
        obs=y.mean(),
        in_large=p.mean() - y.mean(),
    )

cohorts = [
    ("eICU", "data/eicu_y.npy", "data/eicu_prob.npy"),
    ("MIMIC-III", "data/mimiciii_y.npy", "data/mimiciii_prob.npy"),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (name, yf, pf) in zip(axes, cohorts):
    y = np.load(os.path.join(ROOT, yf)); p = np.load(os.path.join(ROOT, pf))
    m = calib_metrics(y, p)
    print(f"{name}: AUC={m['auc']:.3f} PR-AUC={m['pra']:.3f} Brier={m['brier']:.3f} "
          f"slope={m['slope']:.3f} intercept={m['intercept']:.3f} "
          f"mean_pred={m['mean_pred']:.3f} obs={m['obs']:.3f} in_large={m['in_large']:+.3f}")
    frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac, "o-", lw=2, label=f"{name} (slope {m['slope']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraction")
    ax.set_title(f"{name} calibration")
    ax.legend(loc="upper left")

plt.tight_layout()
out = os.path.join(ROOT, "figures", "external_calibration.png")
plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
print(f"Saved {out}")
