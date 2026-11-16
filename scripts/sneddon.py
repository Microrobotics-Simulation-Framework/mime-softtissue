"""Sneddon's pressurised penny crack in the axisymmetric build, against the closed form (LIT-3D benchmark 1).

A crack of radius a on the plane z = 0 of a cylinder (radius and half-height ``--domain`` times a), the upper half by
symmetry, pressure p on the crack's faces, small load (p/E 1e-5, so the large-strain law is linear to that order).
Two representations:
- ``sharp``: the faces are the plane's nodes at r < a, loaded by p (consistent nodal loads, 2 pi r weighted); the
  plane bonded beyond (u_z = 0 for r >= a).
- ``pf``: the phase field's crack: d prescribed as exp(-dist/l), dist the distance to the crack's disc; the plane all
  symmetric (u_z = 0); the pressure's work p V with V = -2 x the integral over the half of u . grad d (the phase-field
  pressurised-crack form of Bourdin et al. [recalled]); the volume from the same integral.
Closed form: V = 16 (1 - nu^2) p a^3 / (3 E), centre opening w0 = 8 (1 - nu^2) p a / (pi E) [recalled].

    python scripts/sneddon.py --out FILE --rep sharp|pf [--h-over-a 0.05] [--l-over-a 0.1] [--domain 20]
        [--plant none|hoop|no_2_pi_r|whole_plane|longer]"""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mime_softtissue.engine.assembly import constrain
from mime_softtissue.engine.fracture.phase_field import damage_at_gauss
from mime_softtissue.engine.grid import Grid, graded_rows
from mime_softtissue.engine.kinematics import axisymmetric as axi
from mime_softtissue.engine.kinematics.plane_strain import N_GAUSS
from mime_softtissue.engine.material import Material, shear_and_bulk
from mime_softtissue.engine.solver import Solver
from mime_softtissue.engine.solvers import newton
from mime_softtissue.engine.specimen import Load, Reaction, Specimen

E, NU, A = 50e3, 0.4, 1e-3
P = 1e-5 * E

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--rep", default="sharp", choices=("sharp", "pf"))
ap.add_argument("--h-over-a", type=float, default=0.05)
ap.add_argument("--l-over-a", type=float, default=0.1)
ap.add_argument("--domain", type=float, default=20.0)
ap.add_argument("--cap", type=float, default=16.0)
ap.add_argument("--plant", default="none", choices=("none", "hoop", "no_2_pi_r", "whole_plane", "longer"))
ap.add_argument("--no-split", action="store_true", help="diagnostic: the whole energy degraded (no tension split)")
ap.add_argument("--kres", type=float, default=1e-6)
ap.add_argument("--plateau", action="store_true", help="diagnostic: d = 1 within one element of the crack's disc")
a = ap.parse_args()

if a.no_split:
    from mime_softtissue.engine import material as law
    real_parts = law.energy_parts
    law.energy_parts = lambda J, I1, fib, mat: (lambda pm: (pm[0] + pm[1], 0.0 * pm[1]))(real_parts(J, I1, fib, mat))

if a.plant == "hoop":
    axi.hoop_stretch = lambda ue, n, radius: jnp.ones((), dtype=ue.dtype)

h = a.h_over_a * A
L = a.domain * A
x = graded_rows(L, h, 1.5 * A, 1.25, a.cap)
y = graded_rows(L, h, 0.5 * A, 1.25, a.cap)
x[0], x[-1], y[0], y[-1] = 0.0, L, 0.0, L
grid = Grid(x, y)
tip = int(np.argmin(np.abs(x - A)))
if abs(x[tip] - A) > 1e-9 * A:
    raise ValueError("the crack's tip is not a node")
free = np.ones((grid.nx + 1, grid.ny + 1, 2))
free[0, :, 0] = 0.0  # the axis
if a.rep == "sharp":
    bonded_from = tip + 1 if a.plant == "longer" else tip
    free[bonded_from:, 0, 1] = 0.0  # the plane beyond the crack (the tip closed)
else:
    free[:, 0, 1] = 0.0  # the symmetry plane, everywhere
empty = (np.zeros(0, int), np.zeros(0, int))
sp = Specimen(grid=grid, free=free, thickness=1.0, force_factor=1.0, load=Load(empty, -1.0, "prescribed_only"),
              reaction=Reaction(empty, 1.0), kinematics="axisymmetric", energy_factor=1.0)
solver = Solver(sp, "elastic")
mu, kappa = shear_and_bulk(E, NU)
mat = Material(mu=jnp.asarray(mu), kappa=jnp.asarray(kappa), gc=jnp.asarray(1.0), l=jnp.asarray(a.l_over_a * A),
               k_res=jnp.asarray(a.kres), stretch_limit=jnp.asarray(10.0))
I, J = solver.I, solver.J
X, Y = np.meshgrid(x, y, indexing="ij")

if a.rep == "sharp":
    # consistent loads of a uniform pressure on the faces r < a: each segment's 2 pi r-weighted linear shares
    loads = np.zeros(grid.nx + 1)
    faces = range(grid.nx) if a.plant == "whole_plane" else range(tip)
    for i in faces:
        r0, r1 = x[i], x[i + 1]
        dr = r1 - r0
        if a.plant == "no_2_pi_r":
            loads[i] += P * dr / 2
            loads[i + 1] += P * dr / 2
        else:
            loads[i] += 2 * np.pi * P * dr * (2 * r0 + r1) / 6
            loads[i + 1] += 2 * np.pi * P * dr * (r0 + 2 * r1) / 6
    f = np.zeros((grid.nx + 1, grid.ny + 1, 2))
    f[:, 0, 1] = loads
    f = jnp.asarray(f * free)
    dq = jnp.zeros((grid.n_elements, 4))

    def volume(u):  # both faces: twice the half's, from the same consistent weights
        return 2.0 * jnp.sum(f[..., 1] * u[..., 1]) / P

    def work(u):
        return jnp.sum(f * u)
else:
    l = a.l_over_a * A
    dist = np.where(X <= A, np.abs(Y), np.hypot(X - A, Y))
    if a.plateau:
        dist = np.maximum(dist - h, 0.0)
    d = jnp.asarray(np.exp(-dist / l))
    dq = damage_at_gauss(d, I, J)
    w = jax.vmap(axi.gauss_weights)(solver.x0, solver.dx, solver.dy)  # (n_el, 4)
    from mime_softtissue.engine.fracture.phase_field import gauss_gradients
    G = jax.vmap(gauss_gradients)(solver.dx, solver.dy)  # (n_el, 4 gauss, 4 nodes, 2)
    grad_d = jnp.einsum("ea,egai->egi", d[I, J], G)  # (n_el, 4, 2)
    Ng = jnp.asarray(N_GAUSS)

    def volume(u):
        ug = jnp.einsum("ga,eai->egi", Ng, u[I, J])
        return -2.0 * jnp.sum(w * jnp.sum(ug * grad_d, axis=-1))

    def work(u):
        return 0.5 * P * volume(u)  # the half's share of p V


def energy(u):
    return solver.energy(u, dq, mat) - work(u)


def hessian_blocks(u):
    ke = jax.vmap(lambda ue, d_, x0, dx, dy: axi.element_hessian(ue, d_, x0, dx, dy, mat))(
        u[I, J], dq, solver.x0, solver.dx, solver.dy)
    return solver.vec.blocks(ke)


t0 = time.time()
u0 = jnp.zeros((grid.nx + 1, grid.ny + 1, 2))
u, its, ok = jax.jit(lambda u: newton.minimise(energy, hessian_blocks, u, solver.free, solver.free_blocks,
                                               max_iter=30, step_tol=1e-12 * h))(u0)
V = float(volume(u))
V_exact = 16 * (1 - NU ** 2) * P * A ** 3 / (3 * E)
if a.rep == "sharp":
    w0 = 2.0 * float(u[0, 0, 1])
else:  # the phase field's opening at the centre: twice -integral of u_z d_z(d) along the axis (trapezoid on nodes)
    lz = a.l_over_a * A
    integrand = -np.asarray(u[0, :, 1]) * (-np.exp(-y / lz) / lz)
    w0 = 2.0 * float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(y)))
w0_exact = 8 * (1 - NU ** 2) * P * A / (np.pi * E)
rec = dict(args=vars(a), cells=[grid.nx, grid.ny], h=h, iterations=int(its), converged=bool(ok),
           volume=V, volume_exact=V_exact, volume_error=V / V_exact - 1.0, centre_opening=w0,
           centre_opening_exact=w0_exact, centre_error=w0 / w0_exact - 1.0, wall_s=time.time() - t0)
if a.rep == "sharp":  # the opening profile on the faces
    rec["profile_r"] = list(map(float, x[:tip + 1] / A))
    rec["profile_w_over_exact"] = [float(2 * u[i, 0, 1] / (w0_exact * np.sqrt(max(1 - (x[i] / A) ** 2, 1e-30))))
                                   if x[i] < A else None for i in range(tip + 1)]
Path(a.out).write_text(json.dumps(rec, indent=1))
print(json.dumps({k: v for k, v in rec.items() if not k.startswith("profile")}, indent=1))
