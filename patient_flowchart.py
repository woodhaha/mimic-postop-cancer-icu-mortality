"""Generate patient flow diagram (STROBE-style) for the study.
Run from repo root: python scripts/patient_flowchart.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(10, 12))
ax.set_xlim(0, 10); ax.set_ylim(0, 14)
ax.axis("off")

def box(x, y, w, h, text, fc="#e8f0fe", ec="#1a73e8", fontsize=10):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                       fc=fc, edgecolor=ec, lw=1.5)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, wrap=True)

def arrow(x, y1, y2):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#1a73e8"))

# Flow
box(2.5, 12.5, 5, 1.0, "MIMIC-IV v3.1\nAll ICU admissions", fc="#f0f0f0")
arrow(5, 12.5, 12.2)

box(2.5, 10.8, 5, 1.2, "Malignancy diagnosis\n(ICD-10 C / D0)\n39,837 admissions", fc="#e8f0fe")
arrow(5, 10.8, 10.5)

box(2.5, 9.1, 5, 1.2, "Underwent surgery\n(ICD-10-PCS Medical & Surgical)\n20,735 admissions", fc="#e8f0fe")
arrow(5, 9.1, 8.8)

box(2.5, 7.4, 5, 1.2, "ICU admission\n(first ICU stay)\n3,801 stays", fc="#e8f0fe")
arrow(5, 7.4, 7.1)

box(2.5, 5.7, 5, 1.2, "Final cohort\n3,801 ICU stays\nDeath: 716 (18.8%)", fc="#fef7e0", ec="#f9ab00")
arrow(5, 5.7, 5.4)

# Split into train/test
arrow(5, 5.7, 4.5)
box(1.5, 3.2, 3.0, 1.0, "Development\n2,661 stays\n(70%)", fc="#e6f4ea", ec="#188038")
box(5.5, 3.2, 3.0, 1.0, "Internal validation\n1,140 stays\n(30%)", fc="#e6f4ea", ec="#188038")

# External
box(2.5, 1.2, 5, 1.0, "External validation: eICU-CRD\ngeneral oncology 8,705 stays\nDeath: 16.4%", fc="#fce8e6", ec="#c5221f")

ax.set_title("Patient Selection Flow Diagram", fontsize=14, pad=10)
plt.tight_layout()
plt.savefig("figures/patient_flow.png", dpi=600, bbox_inches="tight"); plt.savefig("figures/hires/Figure1_flow.pdf", bbox_inches="tight")
plt.close()
print("Saved figures/patient_flow.png")
