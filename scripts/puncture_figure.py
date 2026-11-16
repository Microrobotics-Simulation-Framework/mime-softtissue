"""Force against depth for the puncture arms, in two panels: the solve methods on the frictionless blunt punch, and
the cases (friction, hollow needle, wider punch, deletion). Lines join converged steps only; x marks a step that did
not converge (its force is not a result). ``python scripts/puncture_figure.py RUNS_DIR FIG.png``"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

d = Path(sys.argv[1])
panels = [("solve methods, blunt punch r 0.5 mm, frictionless", (0, 16),
           [("pf_base", "staggered Newton"), ("pf_floor1e-3", "floor k_res 1e-3"), ("pf_eta1", "viscous eta = Gc/l"),
            ("pf_bfgs", "BFGS, monolithic"), ("pf_reg3", "regularised Newton"),
            ("pf_reg_eta0.1", "regularised + viscous 0.1 Gc/l"), ("pf_reg_floor1e-3", "regularised + floor 1e-3")]),
          ("cases (regularised Newton)", (0, 50),
           [("pf_reg3", "punch r 0.5, frictionless"), ("pf_reg_mu0.3", "punch r 0.5, friction 0.3"),
            ("punch06_reg", "punch r 0.6, frictionless"), ("tube_reg", "needle OD 1.2 / ID 0.8, frictionless"),
            ("tube_reg_mu0.3", "needle, friction 0.3"), ("del_reg", "element deletion, punch r 0.5")])]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (title, ylim, runs) in zip(axes, panels):
    for name, label in runs:
        p = d / f"{name}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        bfgs = rec["args"].get("scheme") == "bfgs"
        rows = rec["rows"]
        x = np.array([r["depth_mm"] for r in rows])
        F = np.array([r["force"] for r in rows]) * 1e3
        ok = np.array([(r["newton_capped"] == 0) if bfgs else r["converged"] for r in rows])
        Fl = np.where(ok, F, np.nan)
        line, = ax.plot(x, Fl, "-", lw=1.4, label=label)
        ax.plot(x[~ok], np.clip(F[~ok], *ylim), "x", color=line.get_color(), ms=5)
    ax.set_ylim(*ylim)
    ax.set_xlabel("depth (mm)")
    ax.set_ylabel("force (mN)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
fig.suptitle("axisymmetric puncture, placeholder gel (E 50 kPa, nu 0.4, Gc 1 J/m^2, l 0.1 mm); "
             "x = step not converged (clipped to the axis)", fontsize=9)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=130)
