"""HERTZ-CHECK, one arm: python -I probe/hertz_arm.py NAME key=value ... ; writes out/NAME.json.

The engine (mime_softtissue.engine.puncture3d.run_puncture) called directly. Lengths in mm.
Keys: tip = sphere | punch; R (sphere-tipped needle radius, or the punch's radius), fillet (punch); E (Pa), kr (kappa / mu);
gc (J/m^2; 1e9 = elastic), l; h, L (block half-width = height), fxy, fz, gcap (the grid's cap, times h), sym, ax;
cfg = tip3d | all2 (the GelBlock presets), then overrides vol, bound, aa, cap, verify, contact, pen;
steps: fixed = the depths in `at` (fractions of R for a sphere, mm for a punch) as fixed steps, or adaptive = 1
(first 0.02 mm, at most `amax` mm, landing on `at`); depth (punch, mm) and step (punch, mm) for fixed steps;
hist = mean | gauss (the history field's source only, the elastic energy keeps `vol`);
plant = emu (Young's modulus where the shear modulus belongs: mu := E, kappa := kr E) | sym1 (the symmetry
multiplier dropped: every domain's multiplier set to 1)."""
import json, os, sys, time
os.environ.setdefault("JAX_ENABLE_X64", "1"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
from dataclasses import replace
from mime_softtissue.engine import puncture3d as P
from mime_softtissue.engine.material import Material

name = sys.argv[1]; a = dict(x.split("=") for x in sys.argv[2:])
f = lambda k, d: float(a.get(k, d))
os.makedirs("out", exist_ok=True)
E, kr = f("E", 5e4), f("kr", 4.6666666666666667)
nu = (3 * kr - 2) / (2 * (3 * kr + 1))
mu = E / (2 * (1 + nu))
plant = a.get("plant", "")
if plant == "emu":
    mu_used = E
elif plant == "":
    mu_used = mu
else:
    mu_used = mu
if plant == "sym1":
    P.SYMMETRIES = {"quarter": 1.0, "half": 1.0, "full": 1.0}
mat = Material(mu=mu_used, kappa=kr * mu_used, gc=f("gc", 1e9), l=f("l", 0.1) * 1e-3, k_res=1e-6, stretch_limit=10.0)
presets = {"tip3d": P.Settings(),
           "all2": P.Settings(anderson=0, stagger_cap=0, newton_cap=0, volumetric="gauss", damage_bound="clip",
                              verify=False)}
st = presets[a.get("cfg", "tip3d")]
if "hist" in a:  # the history field's source alone: tension_energy evaluated as `hist` (mean | gauss), whatever `vol` is
    _tension = P.S.tension_energy
    P.S.tension_energy = lambda u, grid, m, volumetric="mean", _h=a["hist"]: _tension(u, grid, m, _h)
    print(f"{name}: history source patched to {a['hist']!r}", flush=True)
over = {}
if "vol" in a: over["volumetric"] = a["vol"]
if "bound" in a: over["damage_bound"] = a["bound"]
if "aa" in a: over["anderson"] = int(a["aa"])
if "cap" in a: over["stagger_cap"] = int(a["cap"]); over["newton_cap"] = int(a["cap"])
if "verify" in a: over["verify"] = bool(int(a["verify"]))
if "contact" in a: over["contact_at"] = a["contact"]
if "pen" in a: over["penalty_factor"] = f("pen", 100.0)
R = f("R", 5.0) * 1e-3
tipk = a.get("tip", "sphere")
if tipk == "sphere":
    tip = P.sphere_tipped_needle(R)
    at = [float(x) * R for x in a.get("at", "0.01,0.02,0.03,0.04,0.05").split(",")]
else:
    tip = P.blunt_punch(R, f("fillet", 0.05) * 1e-3)
    at = []
adaptive = bool(int(a.get("adaptive", 0)))
if tipk == "sphere":
    depth = at[-1]
    if adaptive:
        over.update(adaptive=True, first_step=0.02e-3, step=f("amax", 0.04) * 1e-3)
    else:  # fixed steps landing on every depth in `at` (equal increments)
        inc = np.diff([0.0] + at)
        assert np.allclose(inc, inc[0]), "fixed steps take equal increments"
        over.update(adaptive=False, step=float(inc[0]))
else:
    depth = f("depth", 0.44) * 1e-3
    over.update(adaptive=bool(adaptive), step=f("step", 0.04) * 1e-3)
st = replace(st, **over)
blk = P.Block(half_width=f("L", 20) * 1e-3, height=f("L", 20) * 1e-3, h=f("h", 0.2) * 1e-3, symmetry=a.get("sym", "quarter"),
              fine_xy=f("fxy", 1.5) * 1e-3, fine_z=f("fz", 1.0) * 1e-3, cap=f("gcap", 16.0), axis_x=f("ax", 0.0) * 1e-3)
t0 = time.time()
print(f"{name}: nu {nu:.4f} mu {mu:.1f} (used {mu_used:.1f}) settings {st}", flush=True)
run = P.run_puncture(blk, mat, tip, depth, settings=st, frames=1, output_depths=at, wall=f("wall", 18000.0),
                     on_step=lambda r: print(f"  depth {r['depth_m'] * 1e3:.4f} mm F {r['force_N']:.6e} N stag "
                                             f"{r['staggered']} newton {r['newton']} krylov {r['krylov']} shifts "
                                             f"{r['shifts']} pen/h {r['penetration_over_h']:.3e} d_max {r['d_max']:.2e} "
                                             f"{r['step_s']:.0f} s", flush=True))
out = {"name": name, "args": a, "nu": nu, "mu": mu, "mu_used": mu_used, "E": E, "kr": kr, "R_m": R, "tip": tipk,
       "rows": {k: np.asarray(v).tolist() for k, v in run.rows.items()},
       "counts": {k: v for k, v in run.counts.items() if k != "rejected_rows"}, "rejected_rows": run.counts["rejected_rows"],
       "settings": {k: (v if not isinstance(v, float) else float(v)) for k, v in vars(st).items()},
       "block": vars(blk), "unknowns": run.counts["unknowns"], "cells": run.counts["cells"], "wall_s": time.time() - t0}
json.dump(out, open(f"out/{name}.json", "w"), indent=1, default=float)
print(f"{name}: status {run.counts['status']} steps {run.counts['steps']} rejected {run.counts['rejected']} "
      f"unknowns {run.counts['unknowns']} wall {time.time() - t0:.0f} s", flush=True)
