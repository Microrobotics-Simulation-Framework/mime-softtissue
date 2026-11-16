"""The elastic stuck punch: axisymmetric (two radii) and 3-D forces against the grid. ``python scripts/punch_figure.py
RUNS_DIR FIG.png``"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d = Path(sys.argv[1])
fig, ax = plt.subplots(figsize=(6.0, 4.0))
for label, pat, m in (("axisymmetric, R 5 mm (inscribed)", "axi_R5_h*.json", "o-"),
                      ("axisymmetric, R 7.07 mm (circumscribed)", "axi_R7.0710678_h*.json", "s-"),
                      ("3-D quarter, 10 x 10 x 5 mm (CLOT-3D's fem3d)", "3d_h*.json", "^-")):
    rows = sorted((json.loads(p.read_text()) for p in d.glob(pat)), key=lambda r: r["h_mm"])
    ax.plot([r["h_mm"] for r in rows], [r["force_mN"] for r in rows], m, label=label)
ax.axhspan(1.4274, 1.4666, color="#999999", alpha=0.25, label="axisymmetric bracket, extrapolated")
ax.set_xlabel("grid spacing h (mm)")
ax.set_ylabel("force at 0.01 mm depth (mN)")
ax.set_title("rigid flat punch, radius 1 mm, stuck, elastic; E 50 kPa, nu 0.4 [placeholder]", fontsize=8)
ax.set_xlim(0, 0.36)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=130)
