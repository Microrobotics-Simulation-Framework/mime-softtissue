"""Summaries of puncture runs (``scripts/puncture.py``): the peak, the fall, convergence through and past it, the
energy balance and its plant, and a figure. ``python scripts/puncture_report.py OUT.json FIG.png RUN.json ...``"""
import json
import sys
from pathlib import Path

import numpy as np


def summary(path):
    rec = json.loads(Path(path).read_text())
    rows = rec["rows"]
    depth = np.array([r["depth_mm"] for r in rows])
    F = np.array([r["force"] for r in rows]) * 1e3
    bfgs = rec["args"].get("scheme") == "bfgs"  # its iteration count is BFGS's (cap 1000), not the staggered loop's
    conv = np.array([(r["newton_capped"] == 0) if bfgs else r["converged"] for r in rows])
    Fc = np.where(conv, F, -np.inf)  # the peak and the low among converged steps only
    k = int(np.argmax(Fc))
    out = dict(run=Path(path).stem, args=rec["args"], status=rec.get("status", "running"), steps=len(rows),
               peak_mN=float(F[k]), peak_depth_mm=float(depth[k]))
    after = np.where(conv[k:], F[k:], np.inf)
    j = k + int(np.argmin(after))
    out.update(low_mN=float(F[j]), low_depth_mm=float(depth[j]), fall_pct=float(100 * (1 - F[j] / F[k])))
    first_bad = int(np.argmin(conv)) if not conv.all() else None
    out["first_unconverged_depth_mm"] = None if first_bad is None else float(depth[first_bad])
    out["converged_steps"] = int(conv.sum())
    out["converged_through_low"] = bool(conv[:j + 1].all())
    past = depth >= depth[j] + 0.2 - 1e-9
    out["converged_to_0.2mm_past_low"] = bool(conv.all() and past.any()) if j > k else False
    # energy balance: W = trapezoid of F d(depth) against U + Gamma + E_c + sum(friction) + sum(viscous)
    dd = np.diff(np.concatenate([[0.0], depth])) * 1e-3
    Fn = F * 1e-3
    W = np.cumsum(0.5 * (Fn + np.concatenate([[0.0], Fn[:-1]])) * dd)
    stored = np.array([r["energy"] + r["crack_energy"] + r["contact_energy"] for r in rows])
    stored = stored + np.cumsum([r["friction_work"] + r["viscous_work"] for r in rows])
    err = (W - stored) / np.where(W > 0, W, np.nan)
    out["balance_max_abs_to_peak_pct"] = float(100 * np.nanmax(np.abs(err[:k + 1])))
    out["balance_min_pct"] = float(100 * np.nanmin(err))
    Fp = np.array([r["force_plant"] for r in rows])
    Wp = np.cumsum(0.5 * (Fp + np.concatenate([[0.0], Fp[:-1]])) * dd)
    errp = (Wp - stored) / np.where(W > 0, W, np.nan)
    out["plant_balance_max_abs_to_peak_pct"] = float(100 * np.nanmax(np.abs(errp[:k + 1])))
    out["max_penetration_over_h_to_peak"] = float(-min(0.0, min(r["penetration_over_h"] for r in rows[:k + 1])))
    out["high_damage_energy_share_last"] = rows[-1]["high_damage_energy_share"]
    out["max_stretch_damaged_last"] = rows[-1]["max_stretch_damaged"]
    out["d_max_at_peak"] = rows[k]["d_max"]
    out["newton_capped_total"] = int(sum(r["newton_capped"] for r in rows))
    out["wall_s"] = rec.get("wall_s")
    out["_curve"] = (depth.tolist(), F.tolist(), conv.tolist())
    return out


def main():
    dst, fig, *runs = sys.argv[1:]
    S = [summary(p) for p in runs]
    Path(dst).write_text(json.dumps([{k: v for k, v in s.items() if k != "_curve"} for s in S], indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_, ax = plt.subplots(figsize=(7.5, 4.5))
    for s in S:
        d, F, c = (np.asarray(v) for v in s["_curve"])
        line, = ax.plot(d, F, "-", lw=1.4, label=s["run"])
        ax.plot(d[~c.astype(bool)], F[~c.astype(bool)], "x", color=line.get_color(), ms=5)
    ax.set_xlabel("depth (mm)")
    ax.set_ylabel("force (mN)")
    ax.set_title("axisymmetric puncture, placeholder gel; x = step not converged", fontsize=9)
    ax.legend(fontsize=7)
    fig_.tight_layout()
    fig_.savefig(fig, dpi=130)
    for s in S:
        print({k: v for k, v in s.items() if k not in ("_curve", "args")})


if __name__ == "__main__":
    main()
