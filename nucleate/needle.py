"""NUCLEATE's needle driver (from GENT-NEEDLE's axi_run.py): ``--sts`` (kPa) and ``--ratio`` (s_hs / s_ts) switch on the
strength-surface phase field (``engine.fracture.strength``; Gc then raw, the cell correction in its delta), else AT2
with the grid-corrected Gc as before; each row also records the crack area (crack energy / Gc).

Engine-direct axisymmetric needle run (the GelBlock's own construction: proportional predictor, nodal penalty 100 E/h,
lagged pressure with zero friction, tol_d 1e-5, viscosity 0.1 Gc/l, regularised Newton, lag relaxation 0.5), with the
stretch reached at every step: the largest isochoric x = J^(-2/3) I1 - 3 and principal stretch over Gauss points whose
damage is under 0.5 (intact), and over all points. Used to measure the stretch range (before the Gent fit) and for the
elastic indentation control. Not the line's runs (those go through MIME's Study).

python axi_run.py OUT.json --R 1.0 --h 0.2 --l 0.5 --depth 2.0 [--jm 1.3] [--mode elastic] [--nu 0.45]"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mime_softtissue.engine import contact as ct  # noqa: E402
from mime_softtissue.engine.fracture import strength as strength_surface  # noqa: E402
from mime_softtissue.engine.fracture.phase_field import discretisation_toughness  # noqa: E402
from mime_softtissue.engine.kinematics import axisymmetric as axi  # noqa: E402
from mime_softtissue.engine.kinematics.axisymmetric import nodal_ring_area  # noqa: E402
from mime_softtissue.engine.material import Material, shear_and_bulk  # noqa: E402
from mime_softtissue.engine.solver import Solver  # noqa: E402
from mime_softtissue.engine.specimen import axisymmetric_block  # noqa: E402


def stretch_stats(solver: Solver, u, d):
    """Per Gauss point: x (isochoric), the largest principal stretch, and the damage there."""
    I, J = solver.I, solver.J

    def per_element(ue, x0, dx, dy):
        F2, Ftt = axi.deformation(ue, x0, dx, dy)

        def at(f, t):
            Jd = (f[0, 0] * f[1, 1] - f[0, 1] * f[1, 0]) * t
            C = f.T @ f
            I1 = C[0, 0] + C[1, 1] + t * t
            x = Jd ** (-2.0 / 3.0) * I1 - 3.0
            tr, det = C[0, 0] + C[1, 1], C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
            top = 0.5 * tr + jnp.sqrt(jnp.maximum(0.25 * tr * tr - det, 0.0))
            low = 0.5 * tr - jnp.sqrt(jnp.maximum(0.25 * tr * tr - det, 0.0))
            lam = jnp.sort(jnp.stack([jnp.sqrt(top), jnp.sqrt(jnp.maximum(low, 1e-300)), t]))
            lmax, lmin = lam[2], lam[0]
            # the mode: ln(l2/l3) / ln(l1/l3), 0 uniaxial tension, 0.5 pure shear, 1 equibiaxial tension
            mode = jnp.log(lam[1] / lam[0]) / jnp.maximum(jnp.log(lam[2] / lam[0]), 1e-12)
            return x, lmax, lmin, Jd, mode
        return jax.vmap(at)(F2, Ftt)

    x, lmax, lmin, Jd, mode = jax.vmap(per_element)(u[I, J], solver.x0, solver.dx, solver.dy)
    dq = solver.damage_at_gauss(d)
    return x, lmax, lmin, Jd, dq, mode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--R", type=float, required=True, help="mm")
    ap.add_argument("--h", type=float, default=0.2, help="mm")
    ap.add_argument("--l", type=float, default=0.5, help="mm")
    ap.add_argument("--gc", type=float, default=25.0)
    ap.add_argument("--mu", type=float, default=7200.0)
    ap.add_argument("--nu", type=float, default=0.45)
    ap.add_argument("--jm", type=float, default=0.0, help="0: neo-Hookean")
    ap.add_argument("--block", type=float, default=25.0, help="mm")
    ap.add_argument("--fine", type=float, default=5.0, help="mm")
    ap.add_argument("--step", type=float, default=0.05, help="mm")
    ap.add_argument("--depth", type=float, required=True, help="mm")
    ap.add_argument("--mode", default="phase_field")
    ap.add_argument("--stop-drop", type=float, default=0.0, help="stop once the force falls this share below its max")
    ap.add_argument("--sts", type=float, default=0.0, help="kPa: the uniaxial tensile strength (0: AT2)")
    ap.add_argument("--ratio", type=float, default=1.5, help="s_hs / s_ts")
    ap.add_argument("--no-fields", action="store_true", help="skip the npz field snapshots")
    a = ap.parse_args()
    mm = 1e-3
    R, h, l = a.R * mm, a.h * mm, a.l * mm
    young = 2 * a.mu * (1 + a.nu)
    mu, kappa = shear_and_bulk(young, a.nu)
    gc = discretisation_toughness(a.gc, h, l) if (a.mode == "phase_field" and not a.sts) else a.gc
    mat = Material(mu=mu, kappa=kappa, gc=gc, l=l, k_res=1e-6, stretch_limit=2.0,
                   gent_jm=(a.jm if a.jm > 0 else None))
    if a.sts:
        mat = mat._replace(strength=strength_surface.coefficients(mat, a.sts * 1e3, a.ratio * a.sts * 1e3, h))
        print(json.dumps(dict(strength=mat.strength._asdict())), flush=True)
    block = axisymmetric_block(radius=a.block * mm, height=a.block * mm, h=h, fine_r=a.fine * mm, fine_z=a.fine * mm,
                               predictor="proportional")
    g = block.grid
    w = np.broadcast_to(nodal_ring_area(g.x)[:, None], (g.nx + 1, g.ny + 1)).copy()
    tip = ct.Indenter("punch", radius=R, fillet=R, top=a.block * mm)
    c = ct.Contact(indenter=tip, stiffness=100.0 * young / h, weights=w, active=np.ones_like(w), friction=0.0,
                   slip=1e-2 * h, always_sliding=True)
    solver = Solver(replace(block, contact=c), a.mode, tol_d=1e-5, viscosity=0.1 * a.gc / l, newton_regularise=True,
                    lag_relaxation=0.5)
    stats = jax.jit(lambda u, d: stretch_stats(solver, u, d))
    state = solver.initial()
    n = int(round(a.depth / a.step))
    rows = []
    t0 = time.perf_counter()
    fmax = 0.0
    centres = np.stack([np.asarray(solver.x0) + 0.5 * np.asarray(solver.dx),
                        np.asarray(g.y[:-1][g.element_columns()[1]]) + 0.5 * np.asarray(solver.dy)], axis=1)

    def save(tag, x, dq, mode, Jd, k):
        if a.no_fields:
            return
        np.savez_compressed(a.out.replace(".json", f"_{tag}.npz"), x=x.astype(np.float32), d=dq.astype(np.float32),
                            mode=mode.astype(np.float32), J=Jd.astype(np.float32), step=k, centres=centres,
                            weights=np.asarray(solver.gauss_weights()))

    def write():
        with open(a.out, "w") as fh:
            json.dump(dict(args=vars(a), grid=[g.nx, g.ny], young=young, gc_input=gc, rows=rows,
                       strength=None if mat.strength is None else mat.strength._asdict()), fh, indent=1)

    for k in range(1, n + 1):
        depth = k * a.step * mm
        state, out = solver.step(state, jnp.asarray(depth), mat)
        x, lmax, lmin, Jd, dq, mode = (np.asarray(v) for v in stats(state.u, state.d))
        intact = dq < 0.5
        f = float(out.force)
        if f >= fmax:  # the fields at the largest force so far (the puncture's, once the force has dropped)
            save("at_max", x, dq, mode, Jd, k)
        fmax = max(fmax, f)
        row = dict(depth_mm=depth / mm, force_N=f, crack_area_mm2=float(out.crack_energy) / float(gc) * 1e6,
                   staggered=int(out.iterations), newton=int(out.newton),
                   capped=int(out.newton_capped), x_max_intact=float(x[intact].max()) if intact.any() else None,
                   lmax_intact=float(lmax[intact].max()) if intact.any() else None,
                   lmin_intact=float(lmin[intact].min()) if intact.any() else None,
                   x_max_all=float(x.max()), lmax_all=float(lmax.max()), d_max=float(dq.max()),
                   d_at=[float(v) for v in centres[int(np.argmax(dq.mean(axis=1)))] / mm],
                   J_max_intact=float(Jd[intact].max()) if intact.any() else None,
                   J_min_intact=float(Jd[intact].min()) if intact.any() else None,
                   x_p999_intact=float(np.percentile(x[intact], 99.9)) if intact.any() else None,
                   n_past_cut=int(np.sum(x > (a.jm * (1 - 1e-3)))) if a.jm > 0 else 0,
                   wall_s=time.perf_counter() - t0)
        rows.append(row)
        write()
        print(json.dumps({k2: (round(v, 5) if isinstance(v, float) else v) for k2, v in row.items()}), flush=True)
        if a.stop_drop and f < (1 - a.stop_drop) * fmax:
            break
    save("last", x, dq, mode, Jd, k)


if __name__ == "__main__":
    main()
