"""The fit with block 60's wall share removed: each Jm's P(d) at block 60 times Jm 4's P(block 120) / P(block 60) at the
same d/R (one correction for every Jm: an approximation). python fit_wall.py LO HI B60_JM4_RUN B120_JM4_RUN DIR..."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from tab import eq5, events, load  # noqa: E402

lo, hi = float(sys.argv[1]), float(sys.argv[2])
q = np.linspace(lo, hi, int(round((hi - lo) / 0.5)) + 1)
_, _, z6, f6, _ = load(sys.argv[3]); _, _, z12, f12, _ = load(sys.argv[4])
corr = np.interp(q * 0.5, z12, f12) / np.interp(q * 0.5, z6, f6)
fin = []
for folder in sys.argv[5:]:
    for p in sorted(Path(folder).glob("*/run.json")):
        a, rows, z, f, d = load(p)
        if a["jm"] <= 0 or events(f):
            continue
        fin.append((a["jm"], np.interp(q * a["R"], z, f) * corr, a["R"], 3 * d["mu"]))
fin.sort(key=lambda t: t[0])
jms = np.array([t[0] for t in fin]); lp = np.log(np.array([t[1] for t in fin])); R, E = fin[0][2], fin[0][3]
grid = np.exp(np.linspace(np.log(jms.min()), np.log(jms.max()), 2001))
lpg = np.array([np.interp(np.log(grid), np.log(jms), lp[:, k]) for k in range(len(q))]).T
print("wall correction (b120/b60) at d/R", lo, hi, ":", round(corr[0], 4), round(corr[-1], 4))
for kp in (0.25, 0.26, 0.27):
    t = np.log(eq5(q * R * 1e-3, R * 1e-3, E, kp)); err = np.mean((lpg - t[None, :]) ** 2, axis=1); k = int(np.argmin(err))
    r = np.exp(lpg[k] - t)
    print(f"k' {kp}: Jm {grid[k]:.3f} rms log err {np.sqrt(err[k]):.3f} ratio {r.min():.3f}-{r.max():.3f}")
t = np.log(eq5(q * R * 1e-3, R * 1e-3, E))
out = []
for k, qq in enumerate(q):
    s = lpg[:, k] - t[k]; i = np.where(np.diff(np.sign(s)) != 0)[0]
    out.append(f"{qq:g}:{grid[i[0]]:.2f}" if len(i) else f"{qq:g}:-")
print("depth-wise Jm:", " ".join(out[::2]))
