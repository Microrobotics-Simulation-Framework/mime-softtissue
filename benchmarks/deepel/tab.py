"""Round table: python tab.py DIR [DIR...]; each DIR holds arm folders with run.json (or *.json files)."""
import json
import sys
from pathlib import Path

import numpy as np

KP, KPS, KPP = 0.26, 0.01, 8 / 3


def eq5(d, R, E, kp=KP):
    return kp * E * d * d + KPP * E * R * d


def load(p):
    d = json.loads(Path(p).read_text())
    a = d["args"]
    rows = d["rows"]
    z = np.array([r["depth_mm"] for r in rows])
    f = np.array([r["force_N"] for r in rows])
    return a, rows, z, f, d


def drops(f, thr=0.005):
    out, fm = [], 0.0
    for k, v in enumerate(f):
        if fm > 0 and v < (1 - thr) * fm:
            out.append((k, 1 - v / fm))
        fm = max(fm, v)
    return out


def events(f, thr=0.005):
    """Drop events: the first step of each run of consecutive drop steps, with the largest drop in it."""
    ev, last = [], -2
    for k, s in drops(f, thr):
        if k == last + 1 and ev:
            ev[-1][1] = max(ev[-1][1], s)
        else:
            ev.append([k, s])
        last = k
    return ev


def main():
    paths = []
    for folder in sys.argv[1:]:
        paths += sorted(Path(folder).glob("*/run.json")) + sorted(Path(folder).glob("*.json"))
    for p in paths:
        try:
            a, rows, z, f, d = load(p)
        except Exception as e:  # noqa: BLE001
            print(p, "unreadable", e)
            continue
        R = a["R"]
        mu = d["mu"]
        E = 3 * mu
        name = p.parent.name if p.name == "run.json" else p.stem
        ev = events(f)
        k1 = ev[0][0] if ev else len(f)
        pre = rows[:k1]
        epen = max((r["edge_pen_mm"] for r in pre), default=float("nan"))
        npen = max((r["node_pen_mm"] for r in pre), default=float("nan"))
        jmin = min((r["J_min"] for r in pre), default=float("nan"))
        capped = sum(r["capped"] for r in rows)
        capped_pre = sum(r["capped"] for r in pre)
        sub = next((r["depth_mm"] for r in rows if r["sub_inside"] > 0), None)
        wall = rows[-1]["wall_s"] / 60 if rows else 0
        evs = " ".join(f"{z[k] / R:.2f}({s * 100:.1f}%)" for k, s in ev[:6])
        print(f"{name:22s} R {R} h {a['h']} {a['contact']} k{a['kfac']:g} b{a['block']:g} grid {d['grid']} to "
              f"{z[-1]:.2f} mm (d/R {z[-1] / R:.1f}) wall {wall:.0f} min capped {capped} (pre {capped_pre})")
        print(f"    drops at d/R: {evs or 'none'}{' ...' if len(ev) > 6 else ''} | first sub-surface node inside at "
              f"{'-' if sub is None else f'd/R {sub / R:.2f}'}")
        print(f"    before first drop: max edge pen {epen / R:.4f} R, node pen {npen / R:.4f} R, min J {jmin:.3f}; "
              f"lmax at end {rows[-1]['lmax']:.2f}")
        rat = []
        for q in (1, 2, 4, 6, 8, 10, 12, 13):
            if q * R <= z[-1] + 1e-9:
                dq = q * R * 1e-3
                Fq = float(np.interp(q * R, z, f))
                clean = q * R <= (z[k1] if ev else z[-1] + 1)
                rat.append(f"{q}:{Fq / eq5(dq, R * 1e-3, E):.3f}{'' if clean else '*'}")
        print(f"    P/eq5 (k' {KP}) by d/R: {' '.join(rat)}   (* after the first drop)")
        pmu = [f"{q}:{float(np.interp(q * R, z, f)) / (mu * (R * 1e-3) ** 2):.1f}" for q in (2, 4, 8, 12)
               if q * R <= z[-1] + 1e-9]
        print(f"    P/(mu R^2): {' '.join(pmu)}")


if __name__ == "__main__":
    main()
