"""Why Newton stops at its cap after the puncture's peak: the baseline run to ``--to`` mm, then the next step's
staggered iterations with Newton unrolled in Python, recording per iteration the free residual's norm, the step's
largest component, the line search's factor and the energy (the operating point, read before naming a cause).
``python scripts/newton_probe.py --out FILE [--to 0.34] [--kres 1e-6] [--staggered 3] [--newton 60]``"""
import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--to", type=float, default=0.34)
ap.add_argument("--kres", type=float, default=1e-6)
ap.add_argument("--staggered", type=int, default=3)
ap.add_argument("--newton", type=int, default=60)
a = ap.parse_args()

from mime_softtissue.engine import contact as ct  # noqa: E402
from mime_softtissue.engine.assembly import constrain  # noqa: E402
from mime_softtissue.engine.fracture import phase_field  # noqa: E402
from mime_softtissue.engine.kinematics import axisymmetric as axi  # noqa: E402
from mime_softtissue.engine.solvers import blocktri  # noqa: E402

from mime_softtissue.engine.fracture.phase_field import discretisation_toughness  # noqa: E402
from mime_softtissue.engine.material import Material, shear_and_bulk  # noqa: E402
from mime_softtissue.engine.solver import Solver  # noqa: E402
from mime_softtissue.engine.specimen import axisymmetric_block, with_indenter  # noqa: E402

# the baseline case of scripts/puncture.py (its defaults), rebuilt here
E, NU, GC, LEN, h = 50e3, 0.4, 1.0, 0.1e-3, 0.05e-3
mu, kappa = shear_and_bulk(E, NU)
mat = Material(mu=jnp.asarray(mu), kappa=jnp.asarray(kappa), gc=jnp.asarray(discretisation_toughness(GC, h, LEN)),
               l=jnp.asarray(LEN), k_res=jnp.asarray(a.kres), stretch_limit=jnp.asarray(2.0))
ind = ct.Indenter("punch", radius=0.5e-3, fillet=0.05e-3, top=5e-3)
block = axisymmetric_block(radius=5e-3, height=5e-3, h=h, fine_r=1.5e-3, fine_z=1.5e-3, predictor="proportional")
s = Solver(with_indenter(block, ind, stiffness=100 * E / h, slip=1e-2 * h), "phase_field", tol_d=1e-5)
state = s.initial()
steps = int(round(a.to / 0.02))
for n in range(steps):
    state, out = s.step(state, jnp.asarray((n + 1) * 0.02e-3), mat)
print("reached", (steps) * 0.02, "mm, force", float(out.force) * 1e3, "mN", flush=True)
depth = jnp.asarray((steps + 1) * 0.02e-3)


@jax.jit
def pieces(u, d):
    dq = s.damage_at_gauss(d)
    tot = lambda v: s.energy(v, dq, mat) + s.contact_energy(v, depth, None)
    e = tot(u)
    r = jax.grad(tot)(u) * s.free
    ke = jax.vmap(lambda ue, dd, x0, dx, dy: axi.element_hessian(ue, dd, x0, dx, dy, mat))(u[s.I, s.J], dq, s.x0,
                                                                                         s.dx, s.dy)
    diag, upper = s.vec.blocks(ke)
    diag = ct.add_nodal_to_blocks(diag, ct.nodal_hessian(s.contact, s.X, u, depth, None))
    diag, upper = constrain(diag, upper, s.free_blocks)
    du = -blocktri.solve(diag, upper, r.reshape(s.grid.nx + 1, -1)).reshape(u.shape) * s.free
    # the solve's own accuracy: the residual of the linear system, relative
    lin = blocktri.matvec(diag, upper, du.reshape(s.grid.nx + 1, -1)).reshape(u.shape) + r
    return e, r, du, jnp.sqrt(jnp.sum(lin ** 2)) / jnp.sqrt(jnp.sum(r ** 2))


@jax.jit
def energy_at(u, d):
    dq = s.damage_at_gauss(d)
    return s.energy(u, dq, mat) + s.contact_energy(u, depth, None)


u = state.u * (depth / state.delta)
d = state.d
rows = []
for k in range(a.staggered):
    for it in range(a.newton):
        e0, r, du, linrel = pieces(u, d)
        al, e1 = 1.0, float(energy_at(u + du, d))
        n_half = 0
        while (not np.isfinite(e1) or e1 > float(e0) + 1e-10 * abs(float(e0)) + 1e-30) and n_half < 20:
            al /= 2
            e1 = float(energy_at(u + al * du, d))
            n_half += 1
        rows.append(dict(staggered=k, newton=it, energy=float(e0), residual=float(jnp.sqrt(jnp.sum(r ** 2))),
                         residual_max=float(jnp.max(jnp.abs(r))), step_max=float(al * jnp.max(jnp.abs(du))),
                         full_step_max=float(jnp.max(jnp.abs(du))), alpha=al, halvings=n_half,
                         energy_drop=float(e0) - e1, linear_rel=float(linrel)))
        print(f"stag {k} it {it:2d} E {float(e0):.12e} |r| {rows[-1]['residual']:.3e} step {rows[-1]['step_max']:.3e} "
              f"alpha {al:.2e} dE {rows[-1]['energy_drop']:.2e} lin {float(linrel):.1e}", flush=True)
        u = u + al * du
        if rows[-1]["step_max"] < 1e-11 * float(s.grid.y[-1]):
            break
    hist = phase_field.updated_history(state.history, s.plus(u, mat))
    d_new = s.solve_damage(hist, mat, None, state.d)
    print("staggered", k, "damage change", float(jnp.max(jnp.abs(d_new - d))), flush=True)
    d = d_new
Path(a.out).write_text(json.dumps(dict(args=vars(a), rows=rows), indent=1))
