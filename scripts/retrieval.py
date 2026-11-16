"""Axisymmetric puncture and retrieval: a rigid tip pressed into a gel block to a set depth through a signed-distance
penalty, then pulled back out, with regularised Coulomb friction and an optional adhesive shear (placeholder), phase
field or the element-deletion control. Writes its record after every step.

    python scripts/retrieval.py --out FILE --shape punch|cone|headed|tube [--friction 0.3] [--adhesion 0] (Pa)
        [--set-depth 0.6] [--end-depth -0.1] (mm) [--eta 0.1] (Gc/l) [--mode phase_field|local|elastic] [--h 0.05]
        [--step 0.02] [--no-regularise] [--plant-slip] [--wall 3000] (s)

Case [all placeholder]: block radius 5 mm, height 5 mm, bottom fixed, side free; gel E 50 kPa, nu 0.4, Gc 1 J/m^2,
l 0.1 mm. Shapes: punch radius 0.5 mm, fillet 0.05 mm; cone half-angle 20 deg to a 0.5 mm shaft, apex 0.05 mm; headed
tip, head 0.6 mm radius and 0.2 mm long on a 0.4 mm neck, edges 0.05 mm; tube OD 1.2 mm, ID 0.8 mm (round 3's).

The insertion's predictor is round 3's (proportional); the withdrawal's moves only the indenter (prescribed_only),
since a proportional guess shrinks the field towards zero as the depth does."""
import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mime_softtissue.engine import contact as ct
from mime_softtissue.engine.fracture.phase_field import discretisation_toughness
from mime_softtissue.engine.material import Material, shear_and_bulk
from mime_softtissue.engine.solver import Solver
from mime_softtissue.engine.specimen import axisymmetric_block, with_indenter

E, NU, GC, LEN = 50e3, 0.4, 1.0, 0.1e-3
RB, HB = 5e-3, 5e-3

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--shape", default="punch", choices=("punch", "cone", "headed", "tube"))
ap.add_argument("--mode", default="phase_field", choices=("phase_field", "local", "elastic"))
ap.add_argument("--kres", type=float, default=1e-6)
ap.add_argument("--eta", type=float, default=0.1)
ap.add_argument("--friction", type=float, default=0.3)
ap.add_argument("--adhesion", type=float, default=0.0)
ap.add_argument("--h", type=float, default=0.05)
ap.add_argument("--step", type=float, default=0.02)
ap.add_argument("--set-depth", type=float, default=0.6)
ap.add_argument("--end-depth", type=float, default=-0.1)
ap.add_argument("--k-factor", type=float, default=100.0)
ap.add_argument("--stretch-limit", type=float, default=2.0)
ap.add_argument("--wall", type=float, default=3000.0)
ap.add_argument("--max-newton", type=int, default=40)
ap.add_argument("--max-staggered", type=int, default=400)
ap.add_argument("--no-regularise", action="store_true")
ap.add_argument("--no-correction", action="store_true", help="Gc fed as is, without the (1 + h/2l) correction")
ap.add_argument("--plant-slip", action="store_true", help="control's plant: the friction blind to the tip's motion")
ap.add_argument("--lag-relaxation", type=float, default=1.0, help="friction: the lagged pressure's under-relaxation")
ap.add_argument("--stall", type=float, default=0.0, help="Newton's stagnation stop, m (0: off)")
ap.add_argument("--head-radius", type=float, default=0.6)
ap.add_argument("--punch-radius", type=float, default=0.5)
a = ap.parse_args()

if a.plant_slip:
    ct.indenter_step = lambda depth, depth_start: jnp.zeros_like(depth)

h = a.h * 1e-3
mu, kappa = shear_and_bulk(E, NU)
gc = discretisation_toughness(GC, h, LEN) if (a.mode == "phase_field" and not a.no_correction) else GC
mat = Material(mu=jnp.asarray(mu), kappa=jnp.asarray(kappa), gc=jnp.asarray(gc), l=jnp.asarray(LEN),
               k_res=jnp.asarray(a.kres), stretch_limit=jnp.asarray(a.stretch_limit))
SHAPES = {
    "punch": ct.Indenter("punch", radius=a.punch_radius * 1e-3, fillet=0.05e-3, top=HB),
    "cone": ct.Indenter("cone", radius=0.5e-3, fillet=0.05e-3, top=HB, half_angle=math.radians(20.0)),
    "headed": ct.Indenter("headed", radius=0.4e-3, fillet=0.05e-3, top=HB, head_radius=a.head_radius * 1e-3,
                          head_length=0.2e-3),
    "tube": ct.Indenter("tube", radius=0.6e-3, fillet=0.0, top=HB, inner=0.4e-3),
}
ind = SHAPES[a.shape]
k_pen = a.k_factor * E / h


def solver_with(predictor: str) -> Solver:
    block = axisymmetric_block(radius=RB, height=HB, h=h, fine_r=1.5e-3, fine_z=1.5e-3, predictor=predictor)
    sp = with_indenter(block, ind, stiffness=k_pen, friction=a.friction, slip=1e-2 * h)
    if a.adhesion:
        from dataclasses import replace
        sp = replace(sp, contact=replace(sp.contact, adhesion=a.adhesion))
    return Solver(sp, a.mode, tol_d=1e-5, max_newton=a.max_newton, max_staggered=a.max_staggered,
                  viscosity=a.eta * GC / LEN, newton_regularise=not a.no_regularise, lag_relaxation=a.lag_relaxation,
                  newton_stall=a.stall)


inserting, withdrawing = solver_with("proportional"), solver_with("prescribed_only")
g = inserting.grid
n_in = int(round(a.set_depth / a.step))
n_out = int(round((a.set_depth - a.end_depth) / a.step))
depths = [k * a.step * 1e-3 for k in range(1, n_in + 1)] + [(a.set_depth - k * a.step) * 1e-3 for k in range(1, n_out + 1)]

rec = dict(args=vars(a), k_penalty=k_pen, gc_input=float(gc), cells=[g.nx, g.ny], indenter=str(ind), rows=[])
state = inserting.initial()
t0 = time.time()
status = "end depth"
snap: dict = {"x": g.x, "y": g.y}
for n, dl in enumerate(depths):
    ts = time.time()
    phase = "in" if n < n_in else "out"
    solver = inserting if phase == "in" else withdrawing
    state, out = solver.step(state, jnp.asarray(dl), mat)
    row = {k: float(v) for k, v in out._asdict().items()}
    dq = solver.damage_at_gauss(state.d)
    row.update(depth_mm=dl * 1e3, phase=phase, step_s=time.time() - ts, d_max=float(jnp.max(dq)),
               converged=bool(row["newton_capped"] == 0 and row["iterations"] < a.max_staggered))
    rec["rows"].append(row)
    print(f"{n:3d} {phase} depth {dl*1e3:6.3f} mm F {row['force']*1e3:9.4f} mN (normal {row['force_normal']*1e3:9.4f}) "
          f"it {int(row['iterations']):4d} newton {int(row['newton']):5d} capped {int(row['newton_capped']):3d} "
          f"dmax {row['d_max']:.3f} {row['step_s']:.1f}s", flush=True)
    if n == n_in - 1:
        snap["set_d"], snap["set_u"] = np.asarray(state.d), np.asarray(state.u)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    if not np.isfinite(row["force"]):
        status = "non-finite"
        break
    if time.time() - t0 > a.wall:
        status = "wall time"
        break
rows = rec["rows"]
F = np.asarray([r["force"] for r in rows])
D = np.asarray([r["depth_mm"] * 1e-3 for r in rows])
phase = np.asarray([r["phase"] for r in rows])
W = float(np.sum(0.5 * (F[1:] + F[:-1]) * np.diff(D)) + 0.5 * F[0] * D[0])  # from depth 0, force 0
W_in = float(np.sum(0.5 * (F[1:n_in] + F[:n_in - 1]) * np.diff(D[:n_in])) + 0.5 * F[0] * D[0])
last = rows[-1]
dissipated = sum(r["friction_work"] + r["viscous_work"] for r in rows)
balance = W - (last["energy"] + last["contact_energy"] + last["crack_energy"] + dissipated)
inn, outt = phase == "in", phase == "out"
k_peak = int(np.argmax(np.where(inn, F, -np.inf)))
k_ret = int(np.argmax(np.where(outt, -F, -np.inf))) if outt.any() else None
summary = dict(status=status, wall_s=time.time() - t0, puncture_force=float(F[k_peak]),
               puncture_depth_mm=float(D[k_peak] * 1e3), force_at_set_depth=float(F[n_in - 1]) if len(F) >= n_in else None,
               falls=bool(k_peak < n_in - 1 and F[n_in - 1] < 0.9 * F[k_peak]) if len(F) >= n_in else None,
               retrieval_force=float(-F[k_ret]) if k_ret is not None else None,
               retrieval_depth_mm=float(D[k_ret] * 1e3) if k_ret is not None else None,
               retrieval_force_normal=float(-rows[k_ret]["force_normal"]) if k_ret is not None else None,
               ratio=float(-F[k_ret] / F[k_peak]) if k_ret is not None else None,
               work=W, work_in=W_in, balance=balance, balance_over_work_in=balance / W_in,
               converged_steps=int(sum(r["converged"] for r in rows)), steps=len(rows),
               unconverged_out=int(sum((not r["converged"]) for r in rows if r["phase"] == "out")),
               unconverged_in=int(sum((not r["converged"]) for r in rows if r["phase"] == "in")))
rec["summary"] = summary
snap["last_d"], snap["last_u"] = np.asarray(state.d), np.asarray(state.u)
np.savez_compressed(str(Path(a.out).with_suffix(".npz")), **snap)
Path(a.out).write_text(json.dumps(rec, indent=1))
print(json.dumps(summary, indent=1))
