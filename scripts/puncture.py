"""Axisymmetric puncture with contact: a rigid blunt punch or a hollow needle pressed into a gel block through a
signed-distance penalty, frictionless or with regularised Coulomb friction, phase field (with a stiffness floor or a
viscous damage term as options) or the element-deletion control. Writes its record after every step.

    python scripts/puncture.py --out FILE [--indenter punch|tube] [--mode phase_field|local|elastic] [--kres 1e-6]
        [--eta 0] (in units of Gc/l) [--friction 0] [--h 0.05] [--step 0.02] [--max-depth 1.5] [--k-factor 100]
        [--stretch-limit 2.0] [--past-low 0.2] (mm) [--wall 3000] (s)

Case [all placeholder]: block radius 5 mm, height 5 mm, bottom fixed, side free; gel E 50 kPa, nu 0.4, Gc 1 J/m^2,
l 0.1 mm; punch radius 0.5 mm with a 0.05 mm edge fillet; tube OD 1.2 mm, ID 0.8 mm, fully rounded."""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mime_softtissue.engine import contact as ct
from mime_softtissue.engine.fracture.phase_field import damage_at_gauss, discretisation_toughness
from mime_softtissue.engine.kinematics import axisymmetric as axi
from mime_softtissue.engine.material import Material, shear_and_bulk
from mime_softtissue.engine.solver import Solver
from mime_softtissue.engine.specimen import axisymmetric_block, with_indenter

E, NU, GC, LEN = 50e3, 0.4, 1.0, 0.1e-3
RB, HB = 5e-3, 5e-3

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--indenter", default="punch", choices=("punch", "tube"))
ap.add_argument("--mode", default="phase_field", choices=("phase_field", "local", "elastic"))
ap.add_argument("--kres", type=float, default=1e-6)
ap.add_argument("--eta", type=float, default=0.0)
ap.add_argument("--friction", type=float, default=0.0)
ap.add_argument("--h", type=float, default=0.05)
ap.add_argument("--step", type=float, default=0.02)
ap.add_argument("--max-depth", type=float, default=1.5)
ap.add_argument("--k-factor", type=float, default=100.0)
ap.add_argument("--stretch-limit", type=float, default=2.0)
ap.add_argument("--past-low", type=float, default=0.2)
ap.add_argument("--wall", type=float, default=3000.0)
ap.add_argument("--max-newton", type=int, default=40)
ap.add_argument("--max-staggered", type=int, default=400)
ap.add_argument("--scheme", default="staggered", choices=("staggered", "bfgs"))
ap.add_argument("--punch-radius", type=float, default=0.5, help="mm (the blunt punch)")
ap.add_argument("--regularise", action="store_true", help="Newton shifts an indefinite tangent until it descends")
a = ap.parse_args()

h = a.h * 1e-3
mu, kappa = shear_and_bulk(E, NU)
gc = discretisation_toughness(GC, h, LEN) if a.mode == "phase_field" else GC
mat = Material(mu=jnp.asarray(mu), kappa=jnp.asarray(kappa), gc=jnp.asarray(gc), l=jnp.asarray(LEN),
               k_res=jnp.asarray(a.kres), stretch_limit=jnp.asarray(a.stretch_limit))
if a.indenter == "punch":
    ind = ct.Indenter("punch", radius=a.punch_radius * 1e-3, fillet=0.05e-3, top=HB)
else:
    ind = ct.Indenter("tube", radius=0.6e-3, fillet=0.0, top=HB, inner=0.4e-3)
block = axisymmetric_block(radius=RB, height=HB, h=h, fine_r=1.5e-3, fine_z=1.5e-3, predictor="proportional")
k_pen = a.k_factor * E / h
sp = with_indenter(block, ind, stiffness=k_pen, friction=a.friction, slip=1e-2 * h)
solver = Solver(sp, a.mode, tol_d=1e-5, max_newton=a.max_newton, max_staggered=a.max_staggered,
                viscosity=a.eta * GC / LEN, scheme=a.scheme, newton_regularise=a.regularise)
g = sp.grid
I, J = solver.I, solver.J

# the planted readout: the indenter force with each node's ring area replaced by its radial width (no 2 pi r)
width = np.broadcast_to((np.concatenate([[0.0], np.diff(g.x) / 2]) + np.concatenate([np.diff(g.x) / 2, [0.0]]))
                        [:, None], (g.nx + 1, g.ny + 1)).copy()
plant_contact = dataclasses.replace(sp.contact, weights=width)


@jax.jit
def diagnostics(state, depth):
    u = state.u
    phi = ct.signed_distance(ind, solver.X[..., 0] + u[..., 0], solver.X[..., 1] + u[..., 1], depth)
    pen = jnp.min(jnp.where(jnp.asarray(sp.contact.active) > 0, phi, jnp.inf))
    if a.mode == "phase_field":
        dq = damage_at_gauss(state.d, I, J)
    elif a.mode == "local":
        dq = jnp.broadcast_to(state.d[:, None], (g.n_elements, 4))
    else:
        dq = jnp.zeros((g.n_elements, 4))
    stored = jax.vmap(lambda ue, d, x0, dx, dy: axi.element_stored(ue, d, x0, dx, dy, mat))(
        u[I, J], dq, solver.x0, solver.dx, solver.dy)
    high = dq > 0.95
    share = jnp.sum(jnp.where(high, stored, 0.0)) / jnp.sum(stored)
    stretch = solver.centre_stretch(u)
    el_high = jnp.mean(dq, axis=1) > 0.95
    smax_high = jnp.max(jnp.where(el_high, stretch, 0.0))
    f_plant = jax.grad(lambda z: ct.penalty_energy(plant_contact, solver.X, u, z))(depth)
    return pen, share, smax_high, jnp.max(stretch), jnp.max(dq), f_plant


rec = dict(args=vars(a), k_penalty=k_pen, gc_input=float(gc), cells=[g.nx, g.ny], rows=[])
state = solver.initial()
depths = np.arange(1, int(round(a.max_depth / a.step)) + 1) * a.step * 1e-3
t0 = time.time()
peak, peak_at, low_after = 0.0, None, None
snap: dict = {"x": g.x, "y": g.y}
status = "max depth"
for n, dl in enumerate(depths):
    ts = time.time()
    state, out = solver.step(state, jnp.asarray(dl), mat)
    pen, share, smax_high, smax, dmax, f_plant = (float(v) for v in diagnostics(state, jnp.asarray(dl)))
    row = {k: float(v) for k, v in out._asdict().items()}
    row.update(depth_mm=dl * 1e3, penetration_over_h=pen / h, high_damage_energy_share=share,
               max_stretch_damaged=smax_high, max_stretch=smax, d_max=dmax, force_plant=f_plant,
               step_s=time.time() - ts,
               converged=bool(row["newton_capped"] == 0 and row["iterations"] < a.max_staggered))
    rec["rows"].append(row)
    f = row["force"]
    print(f"{n:3d} depth {dl*1e3:6.3f} mm F {f*1e3:9.4f} mN it {int(row['iterations']):4d} newton "
          f"{int(row['newton']):5d} capped {int(row['newton_capped']):3d} dmax {dmax:.3f} pen/h {pen/h:+.3f} "
          f"share {share:.3f} {row['step_s']:.1f}s", flush=True)
    if f > peak:
        peak, peak_at, low_after = f, n, None
        snap["peak_d"], snap["peak_u"], snap["peak_depth"] = np.asarray(state.d), np.asarray(state.u), dl
    elif peak_at is not None and f < 0.9 * peak:
        if low_after is None or f < rec["rows"][low_after]["force"]:
            low_after = n
    Path(a.out).write_text(json.dumps(rec, indent=1))
    if low_after is not None and (dl - depths[low_after]) * 1e3 >= a.past_low - 1e-9:
        if all(r["force"] >= rec["rows"][low_after]["force"] * 0.999 for r in rec["rows"][low_after:]):
            status = "past the low point"
            break
    if not np.isfinite(f):
        status = "non-finite"
        break
    if time.time() - t0 > a.wall:
        status = "wall time"
        break
rec["status"] = status
rec["wall_s"] = time.time() - t0
snap["last_d"], snap["last_u"], snap["last_depth"] = np.asarray(state.d), np.asarray(state.u), dl
np.savez_compressed(str(Path(a.out).with_suffix(".npz")), **snap)
Path(a.out).write_text(json.dumps(rec, indent=1))
print("status", status, f"{rec['wall_s']:.0f} s")
