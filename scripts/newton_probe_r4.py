"""Why the regularised Newton caps at h = 0.025 mm (round 3's frictional punch): the run to ``--to`` mm with the
engine's own step, then the next step's staggered loop replayed in Python with the engine's Newton taken one iteration
at a time (``Solver.newton`` with a cap of 1: the same arithmetic), recording each iteration's accepted step, the free
residual, and the energy change, up to ``--newton`` iterations a solve (the engine's cap is 40).

    python scripts/newton_probe_r4.py --out FILE [--to 0.38] [--h 0.025] [--friction 0.3] [--newton 200]"""
import argparse
import json
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
ap.add_argument("--to", type=float, default=0.38)
ap.add_argument("--h", type=float, default=0.025)
ap.add_argument("--friction", type=float, default=0.3)
ap.add_argument("--newton", type=int, default=200)
ap.add_argument("--steps", type=int, default=2, help="probed steps after --to")
a = ap.parse_args()

E, NU, GC, LEN = 50e3, 0.4, 1.0, 0.1e-3
h = a.h * 1e-3
mu, kappa = shear_and_bulk(E, NU)
mat = Material(mu=jnp.asarray(mu), kappa=jnp.asarray(kappa), gc=jnp.asarray(discretisation_toughness(GC, h, LEN)),
               l=jnp.asarray(LEN), k_res=jnp.asarray(1e-6), stretch_limit=jnp.asarray(2.0))
ind = ct.Indenter("punch", radius=0.5e-3, fillet=0.05e-3, top=5e-3)
block = axisymmetric_block(radius=5e-3, height=5e-3, h=h, fine_r=1.5e-3, fine_z=1.5e-3, predictor="proportional")
sp = with_indenter(block, ind, stiffness=100 * E / h, friction=a.friction, slip=1e-2 * h)
s = Solver(sp, "phase_field", tol_d=1e-5, newton_regularise=True)
one = Solver(sp, "phase_field", tol_d=1e-5, newton_regularise=True, max_newton=1)
step_tol = 1e-11 * float(s.grid.y[-1])

state = s.initial()
n_to = int(round(a.to / 0.02))
t0 = time.time()
for n in range(n_to):
    state, out = s.step(state, jnp.asarray((n + 1) * 0.02e-3), mat)
    print(f"depth {(n + 1) * 0.02:.2f} F {float(out.force) * 1e3:.4f} mN capped {int(out.newton_capped)} "
          f"{time.time() - t0:.0f}s", flush=True)


@jax.jit
def iterate(u, d, depth, lag):
    dq = s.damage_at_gauss(d)
    u1, _, ok = one.newton(u, dq, mat, None, depth, lag)
    tot = lambda v: s.energy(v, dq, mat) + s.contact_energy(v, depth, lag)
    r = jax.grad(tot)(u) * s.free
    return u1, ok, tot(u), tot(u1), jnp.sqrt(jnp.sum(r * r)), jnp.max(jnp.abs(r)), jnp.max(jnp.abs(u1 - u))


@jax.jit
def damage_and_lag(u, d, depth, lag, history, d_start):
    hist = phase_field.updated_history(history, s.plus(u, mat))
    d_new = s.solve_damage(hist, mat, None, d_start)
    new_lag = s._lag(u, depth, lag)
    dp = jnp.max(jnp.abs(new_lag.pressure - lag.pressure)) / (jnp.max(new_lag.pressure) + 1e-300)
    return d_new, hist, new_lag, jnp.max(jnp.abs(d_new - d)), dp


rows = []
for m in range(a.steps):
    depth = jnp.asarray((n_to + m + 1) * 0.02e-3)
    u = apply_load(sp.load, state.u, depth, state.delta, s.y_frac)
    p0, n0 = ct.pressure_and_normal(sp.contact, s.X, state.u, state.delta)
    lag = ct.Lag(p0, n0, state.u, jnp.asarray(state.delta))
    d = state.d
    for k in range(400):
        for it in range(a.newton):
            u1, ok, e0, e1, rn, rmax, size = iterate(u, d, depth, lag)
            rows.append(dict(step=m, staggered=k, newton=it, energy=float(e0), energy_change=float(e1 - e0),
                             residual=float(rn), residual_max=float(rmax), step_size=float(size), converged=bool(ok)))
            if it < 3 or it % 10 == 0 or bool(ok):
                print(f"step {m} stag {k} it {it:3d} E {float(e0):.15e} dE {float(e1 - e0):+.2e} |r| {float(rn):.3e} "
                      f"step {float(size):.3e} ok {bool(ok)}", flush=True)
            u = u1
            if bool(ok):
                break
        d_new, hist, lag, err, dp = damage_and_lag(u, d, depth, lag, state.history, state.d)
        print(f"step {m} stag {k}: newton its {it + 1} damage change {float(err):.2e} pressure change {float(dp):.2e}",
              flush=True)
        d = d_new
        if float(err) <= 1e-5 and float(dp) <= 1e-3:
            break
    state = state._replace(u=u, d=d, history=hist, delta=depth)
    Path(a.out).write_text(json.dumps(dict(args=vars(a), step_tol=step_tol, rows=rows), indent=1))
Path(a.out).write_text(json.dumps(dict(args=vars(a), step_tol=step_tol, rows=rows), indent=1))
print("done", time.time() - t0)
