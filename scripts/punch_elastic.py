"""A rigid flat punch stuck to a block's top face, elastic, small depth: the axisymmetric build against the 3-D build
(CLOT-3D's ``fem3d``, copied unchanged into ``clot3d_copy/``), the same material law in both.

    python scripts/punch_elastic.py axi --h 0.125 --radius 5 --out FILE [--plant NAME]
    python scripts/punch_elastic.py 3d --h 0.25 --out FILE

Block: height 5 mm, bottom fixed, side free; the axisymmetric block's radius 5 mm (inscribed in the 3-D block's
quarter, 5 x 5 mm) or 7.071 mm (circumscribed); punch radius 1 mm, nodes within it moved down with it (stick), depth
0.01 mm in two steps. E 50 kPa, nu 0.4 [placeholder]. Lengths in the arguments in mm."""
import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

A, H, L, DEPTH = 1e-3, 5e-3, 5e-3, 1e-5
E, NU = 50e3, 0.4

ap = argparse.ArgumentParser()
ap.add_argument("build", choices=("axi", "3d"))
ap.add_argument("--h", type=float, required=True)
ap.add_argument("--radius", type=float, default=5.0)
ap.add_argument("--out", required=True)
ap.add_argument("--plant", default="")
a = ap.parse_args()
h = a.h * 1e-3
mu, kappa = E / (2 * (1 + NU)), E / (3 * (1 - 2 * NU))
res = dict(build=a.build, h_mm=a.h, depth_mm=DEPTH * 1e3, punch_mm=A * 1e3, plant=a.plant)
t0 = time.time()

if a.build == "axi":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    from mime_softtissue.engine.loadpath import run
    from mime_softtissue.engine.material import Material
    from mime_softtissue.engine.solver import Solver
    from mime_softtissue.engine.specimen import axisymmetric_block
    Rb = a.radius * 1e-3
    n1 = int(round(L / h))
    x = np.linspace(0.0, L, n1 + 1)
    if Rb > L * (1 + 1e-9):
        n2 = max(1, int(round((Rb - L) / h)))
        x = np.concatenate([x, L + np.linspace(0.0, Rb - L, n2 + 1)[1:]])
    y = np.linspace(0.0, H, int(round(H / h)) + 1)
    sp = axisymmetric_block(radius=Rb, height=H, h=h, fine_r=Rb, fine_z=H, stuck_punch=A, x=x, y=y)
    one = jnp.asarray(1.0)
    mat = Material(mu=jnp.asarray(mu), kappa=jnp.asarray(kappa), gc=one, l=one, k_res=jnp.asarray(0.0),
                   stretch_limit=one)
    import contextlib
    ctx = contextlib.nullcontext()
    if a.plant:
        import engine.test_axisymmetric as T
        ctx = T.AXI_PLANTS[a.plant]()
    with ctx:
        r = run(Solver(sp, "elastic"), mat, np.array([DEPTH / 2, DEPTH]))
    res.update(radius_mm=a.radius, cells=[len(x) - 1, len(y) - 1], force_mN=float(r["force"][-1]) * 1e3,
               force_half_mN=float(r["force"][0]) * 1e3, newton=r["newton"].tolist(),
               capped=r["newton_capped"].tolist())
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "clot3d_copy"))
    import fem3d as F
    n = int(round(L / h))
    nz = int(round(H / h))
    grid = F.uniform_grid((n, n, nz), (L, L, H))
    shape = grid.nodes
    free = np.ones(shape + (3,))
    free[0, :, :, 0] = 0.0
    free[:, 0, :, 1] = 0.0
    free[:, :, 0, :] = 0.0
    Xn, Yn = np.meshgrid(grid.x, grid.y, indexing="ij")
    punch = (Xn ** 2 + Yn ** 2) <= A ** 2 * (1 + 1e-9)
    free[punch, -1, :] = 0.0
    free = jnp.asarray(free)
    pmask = jnp.asarray(np.zeros(shape, bool)).at[:, :, -1].set(jnp.asarray(punch))
    g = jnp.ones((2, 2, 2) + grid.shape)
    opts = F.GMGOptions()

    def energy(u):
        return F.solid_energy(u, g, grid, mu, kappa)

    def make_energy(gr, c):
        return lambda u: F.solid_energy(u, c["g"], gr, mu, kappa)

    @jax.jit
    def newton_step(u):
        gradE = jax.grad(energy)
        r = gradE(u) * free
        K = jax.linearize(gradE, u)[1]
        levels, transfers = F.build_hierarchy(make_energy, {"g": g}, u, grid, free, opts, ("g",))
        Aop = lambda v: free * K(free * v) + (1.0 - free) * v
        du, k, _, neg = F.pcg(Aop, F.gmg_solver(levels, transfers, opts), r, jnp.zeros_like(r), 1e-11, 400)
        du = -du * free
        e0 = energy(u)

        def worse(s):
            al, e, m = s
            return ((~jnp.isfinite(e)) | (e > e0 + 1e-10 * jnp.abs(e0) + 1e-30)) & (m < 20)

        def halve(s):
            al, _, m = s
            return al / 2, energy(u + (al / 2) * du), m + 1

        al, _, _ = jax.lax.while_loop(worse, halve, (jnp.ones(()), energy(u + du), 0))
        return u + al * du, al * jnp.max(jnp.abs(du)), k, neg

    @jax.jit
    def force_of(u):
        r = jax.grad(energy)(u)
        return -4.0 * jnp.sum(jnp.where(pmask, r[..., 2], 0.0))

    u = jnp.zeros(shape + (3,))
    forces, its = [], []
    prev = 0.0
    for dl in (DEPTH / 2, DEPTH):
        u = u * (dl / prev) if prev > 0 else u
        u = jnp.where(pmask[..., None], jnp.asarray([0.0, 0.0, -dl]), u)
        prev = dl
        for it in range(30):
            u, size, k, neg = newton_step(u)
            its.append(int(k))
            if float(size) < 1e-11 * H:
                break
        forces.append(float(force_of(u)))
    res.update(cells=list(grid.shape), unknowns=int(3 * np.prod(shape)), force_mN=forces[-1] * 1e3,
               force_half_mN=forces[0] * 1e3, pcg_per_newton=its, punch_nodes_quarter=int(np.sum(punch)))
res["wall_s"] = time.time() - t0
Path(a.out).write_text(json.dumps(res, indent=1))
print(json.dumps(res))
