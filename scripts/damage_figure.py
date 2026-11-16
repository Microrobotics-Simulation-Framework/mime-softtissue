"""The damage field of a puncture run at its peak and at its last step, in the deformed (R, Z) half-plane near the
indenter. ``python scripts/damage_figure.py RUN.npz FIG.png TITLE``"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

z = np.load(sys.argv[1])
x, y = z["x"] * 1e3, z["y"] * 1e3
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
for ax, key, lab in ((axes[0], "peak", "at the peak"), (axes[1], "last", "last step")):
    d, u = z[f"{key}_d"], z[f"{key}_u"] * 1e3
    X, Y = np.meshgrid(x, y, indexing="ij")
    Xd, Yd = X + u[..., 0], Y + u[..., 1]
    if d.ndim == 1:  # element damage (deletion): shown at element centres
        nx, ny = len(x) - 1, len(y) - 1
        d = d.reshape(nx, ny)
        Xc = 0.25 * (Xd[:-1, :-1] + Xd[1:, :-1] + Xd[1:, 1:] + Xd[:-1, 1:])
        Yc = 0.25 * (Yd[:-1, :-1] + Yd[1:, :-1] + Yd[1:, 1:] + Yd[:-1, 1:])
        m = ax.scatter(Xc.ravel(), Yc.ravel(), c=d.ravel(), s=4, cmap="magma_r", vmin=0, vmax=1)
    else:
        m = ax.pcolormesh(Xd, Yd, d, shading="gouraud", cmap="magma_r", vmin=0, vmax=1)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(3.5, 5.05)
    ax.set_aspect("equal")
    ax.set_xlabel("R (mm)")
    ax.set_title(f"{lab}: depth {float(z[f'{key}_depth']) * 1e3:.2f} mm", fontsize=9)
axes[0].set_ylabel("Z (mm)")
fig.colorbar(m, ax=axes, label="damage d", shrink=0.8)
fig.suptitle(sys.argv[3], fontsize=9)
fig.savefig(sys.argv[2], dpi=130, bbox_inches="tight")
