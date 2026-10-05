"""Forest plot of cancer-type subgroup AUCs with 95% CI."""
import pandas as pd, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res = pd.read_csv(os.path.join(ROOT,"reports","subgroup_cancer_auc.csv"))

labels = {'lung':'Lung / thoracic','GI':'GI tumors (upper GI, colorectal, hepato-pancreatic)','other':'Other cancers'}
res['label'] = res.subgroup.map(labels)

fig, ax = plt.subplots(figsize=(7,3.5))
ypos = np.arange(len(res))[::-1]
for y,(_,r) in zip(ypos, res.iterrows()):
    ax.plot([r.auc], [y], 'o', color='tab:blue', ms=8, zorder=3)
    ax.plot([r.ci_low, r.ci_high], [y, y], color='tab:blue', lw=2, zorder=2)
    ax.text(0.92, y, f"{r.auc:.3f} ({r.ci_low:.2f}-{r.ci_high:.2f})", va='center', ha='right', fontsize=9)
ax.axvline(0.5, color='gray', ls='--', lw=1, label='Chance')
ax.set_yticks(ypos)
ax.set_yticklabels([res.iloc[i].label for i in range(len(res))], fontsize=9)
ax.set_xlabel('Test-set AUC (95% CI)', fontsize=9)
ax.set_xlim(0.4, 1.0)
ax.set_title('Subgroup analysis by cancer type', fontsize=10)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(ROOT,"figures","subgroup_cancer_forest.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(ROOT,"figures","hires","subgroup_cancer_forest.pdf"), bbox_inches="tight")
print("Saved: figures/subgroup_cancer_forest.png + PDF")
