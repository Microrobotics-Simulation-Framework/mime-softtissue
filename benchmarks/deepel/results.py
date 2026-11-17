"""Collects every arm into one results file: python results.py OUT.json DIR..."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tab import eq5, events, load  # noqa: E402

out = {}
for folder in sys.argv[2:]:
    for p in sorted(Path(folder).glob("*/run.json")):
        a, rows, z, f, d = load(p)
        R, mu = a["R"], d["mu"]
        ev = events(f)
        k1 = ev[0][0] if ev else len(f)
        pre = rows[:k1]
        qs = [q for q in (1, 2, 4, 6, 8, 10, 12, 13) if q * R <= z[-1] + 1e-9]
        out[p.parent.name] = dict(
            R_mm=R, h_mm=a["h"], contact=a["contact"], kfac=a["kfac"], edge_points=a["edge_points"], jm=a["jm"],
            fillet=a.get("fillet", 1.0), axis_h=a.get("axis_h", 0.0), block_mm=a["block"], grid=d["grid"],
            reached_d_over_R=round(float(z[-1] / R), 3), wall_min=round(rows[-1]["wall_s"] / 60, 1),
            newton_capped=int(sum(r["capped"] for r in rows)),
            drops=[[round(float(z[k] / R), 3), round(float(s), 4)] for k, s in ev],
            before_first_drop=dict(edge_pen_over_R=round(max(r["edge_pen_mm"] for r in pre) / R, 4) if pre else None,
                                   min_J=round(min(r["J_min"] for r in pre), 3) if pre else None,
                                   capped=int(sum(r["capped"] for r in pre))),
            P_over_muR2={str(q): round(float(np.interp(q * R, z, f)) / (mu * (R * 1e-3) ** 2), 3) for q in qs},
            P_over_eq5={str(q): round(float(np.interp(q * R, z, f) / eq5(q * R * 1e-3, R * 1e-3, 3 * mu)), 4)
                        for q in qs},
            clean_to_d_over_R=round(float(z[k1] / R), 3) if ev else None)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(len(out), "arms")
