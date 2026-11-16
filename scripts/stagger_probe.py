"""Why the staggered loop reaches its cap (400) during withdrawal with friction: the retrieval protocol run with the
engine's own step to ``--to`` (mm, on the way out), then the next withdrawal step's staggered loop replayed in Python
(each Newton solve the engine's own), recording each iteration's damage change and the lagged pressure's change, and
which nodes' pressure changes most; then the same step again with the lag under-relaxed by ``--omega``.

    python scripts/stagger_probe.py --out FILE [--shape punch] [--friction 0.3] [--to 0.58] [--iters 60]"""
import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mime_softtissue.engine import contact as ct
from mime_softtissue.engine.fracture import phase_field
from mime_softtissue.engine.fracture.phase_field import discretisation_toughness
from mime_softtissue.engine.material import Material, shear_and_bulk
from mime_softtissue.engine.solver import Solver
from mime_softtissue.engine.specimen import apply_load, axisymmetric_block, with_indenter

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--shape", default="punch", choices=("punch", "cone"))
ap.add_argument("--friction", type=float, default=0.3)
ap.add_argument("--to", type=float, default=0.58, help="the last withdrawal depth stepped by the engine (mm)")
ap.add_argument("--set-depth", type=float, default=0.6)
ap.add_argument("--iters", type=int, default=60)
ap.add_argument("--omega", type=float, nargs="+", default=[0.5])
a = ap.parse_args()

E, NU, GC, LEN, h = 50e3, 0.4, 1.0, 0.1e-3, 0.05e-3
mu, kappa = shear_and_bulk(E, NU)
mat = Material(mu=jnp.asarray(mu), kappa=jnp.asarray(kappa), gc=jnp.asarray(discretisation_toughness(GC, h, LEN)),
               l=jnp.asarray(LEN), k_res=jnp.asarray(1e-6), stretch_limit=jnp.asarray(2.0))
ind = (ct.Indenter("punch", radius=0.5e-3, fillet=0.05e-3, top=5e-3) if a.shape == "punch" else
       ct.Indenter("cone", radius=0.5e-3, fillet=0.05e-3, top=5e-3, half_angle=math.radians(20.0)))


def solver(predictor):
    block = axisymmetric_block(radius=5e-3, height=5e-3, h=h, fine_r=1.5e-3, fine_z=1.5e-3, predictor=predictor)
    return Solver(with_indenter(block, ind, stiffness=100 * E / h, friction=a.friction, slip=1e-2 * h), "phase_field",
                  tol_d=1e-5, viscosity=0.1 * GC / LEN, newton_regularise=True)


ins, wd = solver("proportional"), solver("prescribed_only")
c = ins.contact
state = ins.initial()
t0 = time.time()
n_in = int(round(a.set_depth / 0.02))
outs = int(round((a.set_depth - a.to) / 0.02))
for k in range(1, n_in + 1):
    state, out = ins.step(state, jnp.asarray(k * 0.02e-3), mat)
for k in range(1, outs + 1):
    state, out = wd.step(state, jnp.asarray((a.set_depth - k * 0.02) * 1e-3), mat)
print(f"at {a.to} mm out: F {float(out.force) * 1e3:.4f} mN, staggered {int(out.iterations)} ({time.time() - t0:.0f} s)",
      flush=True)
depth = jnp.asarray((a.to - 0.02) * 1e-3)


@jax.jit
def solve_u(u, d, lag):
    u1, n, ok = wd.newton(u, wd.damage_at_gauss(d), mat, None, depth, lag)
    return u1, n, ok


@jax.jit
def after(u, d, lag):
    hist = phase_field.updated_history(state.history, wd.plus(u, mat))
    d_new = wd.solve_damage(hist, mat, None, state.d)
    p, n = ct.pressure_and_normal(c, wd.X, u, depth)
    return d_new, p, n


def replay(omega):
    u = apply_load(wd.specimen.load, state.u, depth, state.delta, wd.y_frac)
    p0, n0 = ct.pressure_and_normal(c, wd.X, state.u, state.delta)
    lag = ct.Lag(p0, n0, state.u, jnp.asarray(state.delta))
    d = state.d
    rows = []
    for it in range(a.iters):
        u, n, ok = solve_u(u, d, lag)
        d_new, p, nrm = after(u, d, lag)
        dp_field = p - lag.pressure
        dp = float(jnp.max(jnp.abs(dp_field)) / (jnp.max(p) + 1e-300))
        err = float(jnp.max(jnp.abs(d_new - d)))
        i, j = np.unravel_index(int(jnp.argmax(jnp.abs(dp_field))), dp_field.shape)
        force = float(wd.indenter_force(u, depth, lag))
        rows.append(dict(it=it, newton=int(n), ok=bool(ok), damage_change=err, pressure_change=dp,
                         worst_node=[int(i), int(j)], worst_node_R_mm=float(wd.grid.x[i] * 1e3),
                         worst_node_Z_mm=float(wd.grid.y[j] * 1e3), p_lag=float(lag.pressure[i, j]),
                         p_new=float(p[i, j]), force=force))
        if it < 12 or it % 10 == 0:
            print(f"omega {omega} it {it:3d} newton {int(n):3d} ok {bool(ok)} dd {err:.2e} dp {dp:.2e} node ({i},{j}) "
                  f"R {wd.grid.x[i]*1e3:.3f} Z {wd.grid.y[j]*1e3:.3f} p {float(lag.pressure[i, j]):.1f}->{float(p[i, j]):.1f} "
                  f"F {force*1e3:.4f} mN", flush=True)
        d = d_new
        relaxed = lag.pressure + omega * (p - lag.pressure)
        lag = ct.Lag(relaxed, nrm, lag.u_start, lag.depth_start)
        if err <= 1e-5 and dp <= 1e-3:
            print(f"omega {omega}: converged at iteration {it}", flush=True)
            break
    return rows


rec = dict(args=vars(a), arms={})
for omega in [1.0, *a.omega]:
    rec["arms"][str(omega)] = replay(omega)
    Path(a.out).write_text(json.dumps(rec, indent=1))
print("done", time.time() - t0)
