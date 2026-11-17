"""Fit Gent's Jm to eq. 5 from a sweep of runs: python fit.py LO HI DIR [DIR...]
For each run (arm folder with run.json): P/eq5 on a d/R grid over [LO, HI] (only where no drop came before). The fit:
Jm minimising the mean squared log ratio, by interpolation in log Jm (piecewise linear between the sweep's members), at
k' and at k' +- 1 sigma (the fitted range). Also Jm where the ratio is 1 at each d/R (where the band is crossed)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tab import eq5, events, load  # noqa: E402

lo, hi = float(sys.argv[1]), float(sys.argv[2])
q = np.linspace(lo, hi, int(round((hi - lo) / 0.5)) + 1)
curves = {}
for folder in sys.argv[3:]:
    for p in sorted(Path(folder).glob("*/run.json")):
        a, rows, z, f, d = load(p)
        R, E = a["R"], 3 * d["mu"]
        ev = events(f)
        zc = z[ev[0][0]] if ev else z[-1]
        jm = a["jm"] if a["jm"] > 0 else np.inf
        ok = q * R <= zc + 1e-9
        P = np.interp(q * R, z, f)
        curves[(jm, p.parent.name)] = (P, ok, R, E, zc / R)
if not curves:
    sys.exit("no runs")
for (jm, name), (P, ok, R, E, zc) in sorted(curves.items()):
    r = P / eq5(q * R * 1e-3, R * 1e-3, E)
    print(f"{name:18s} Jm {jm:6g} clean to d/R {zc:5.2f}  ratio " + " ".join(f"{qq:g}:{rr:.3f}" for qq, rr in
                                                                         zip(q[::4], r[::4])))
fin = sorted([(jm, v) for (jm, _), v in curves.items() if np.isfinite(jm) and v[1].all()], key=lambda t: t[0])
if len(fin) < 2:
    sys.exit("fewer than two clean finite-Jm runs over the range")
jms = np.array([t[0] for t in fin])
lp = np.log(np.array([t[1][0] for t in fin]))  # (n_jm, n_q)
R, E = fin[0][1][2], fin[0][1][3]
grid = np.exp(np.linspace(np.log(jms.min()), np.log(jms.max()), 2001))
lpg = np.array([np.interp(np.log(grid), np.log(jms), lp[:, k]) for k in range(len(q))]).T  # (n_grid, n_q)
for kp in (0.25, 0.26, 0.27):
    target = np.log(eq5(q * R * 1e-3, R * 1e-3, E, kp))
    err = np.mean((lpg - target[None, :]) ** 2, axis=1)
    k = int(np.argmin(err))
    ratio = np.exp(lpg[k] - target)
    inside = np.abs(ratio - 1) <= 0.04
    print(f"k' {kp}: Jm {grid[k]:.3f} rms log err {np.sqrt(err[k]):.3f}; ratio {ratio.min():.3f}-{ratio.max():.3f} "
          f"(d/R {lo:g}: {ratio[0]:.3f}, {hi:g}: {ratio[-1]:.3f}); within 4 % at d/R "
          f"{', '.join(f'{x:g}' for x in q[inside]) or 'none'}")
target = np.log(eq5(q * R * 1e-3, R * 1e-3, E))
cross = []
for k, qq in enumerate(q):
    s = lpg[:, k] - target[k]
    i = np.where(np.diff(np.sign(s)) != 0)[0]
    cross.append(f"{qq:g}:{grid[i[0]]:.2f}" if len(i) else f"{qq:g}:-")
print("Jm putting P on eq. 5 at each d/R: " + " ".join(cross[::2]))
