"""Deep elastic indentation by a sphere-tipped needle (engine-direct, axisymmetric, damage off): how deep the numerics
follow an elastic gel without contact or mesh artefacts, and the loading curve P(d).

The needle is ``contact.Indenter("punch", radius=R, fillet=R)`` (a cylinder with a hemispherical end, its exact signed
distance). The block is a cylinder of radius and height ``--block``, its bottom fixed; the grid is uniform ``h`` within
``--fine-r`` of the axis and ``--fine-z`` of the top, then grows by 1.25 a cell up to ``--cap`` mm (an absolute cap, so a
large block stays cheap), then uniform. Frictionless contact with no lagged pressure (one Newton solve a step).

Each step records, besides the force: Newton's iterations and whether it stopped at its cap; the deepest node inside the
tip (-phi at the nodes); the deepest point of the deformed top surface inside the tip (each top edge sampled at
``SAMPLES`` points: the tip passing between nodes); the nodes below the top row inside the tip (the tip through the
surface); the smallest J at the Gauss points and the count under 0.05 (near inversion); the largest principal stretch.

python axi_deep.py OUT.json --R 0.5 --h 0.1 --depth 6.5 [--contact nodal|edge] [--edge-points 4] [--jm 0]"""
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
from mime_softtissue.engine.grid import graded_rows  # noqa: E402
from mime_softtissue.engine.kinematics import axisymmetric as axi  # noqa: E402
from mime_softtissue.engine.kinematics.axisymmetric import nodal_ring_area  # noqa: E402
from mime_softtissue.engine.material import Material, shear_and_bulk  # noqa: E402
from mime_softtissue.engine.solver import Solver  # noqa: E402
from mime_softtissue.engine.specimen import axisymmetric_block  # noqa: E402

SAMPLES = 33  # points along each deformed top edge (ends included) for the between-node penetration readout


def grid_coords(size: float, h: float, fine: float, cap: float) -> np.ndarray:
    return graded_rows(size, h, fine, 1.25, cap / h)


def axis_columns(size: float, h: float, fine: float, cap: float, axis_h: float, axis_zone: float) -> np.ndarray:
    """Columns uniform ``axis_h`` within ``axis_zone`` of the axis, growing by 1.25 to ``h``, uniform ``h`` to ``fine``,
    then growing to ``cap``; the last stretch rescaled to land on ``size``."""
    pts = list(np.arange(0.0, axis_zone + 0.5 * axis_h, axis_h))
    s = axis_h
    while s < h:
        s = min(s * 1.25, h)
        pts.append(pts[-1] + s)
    while pts[-1] < fine - 1e-12:
        pts.append(pts[-1] + h)
    start = len(pts) - 1
    while pts[-1] < size - 1e-12:
        s = min(s * 1.25, cap)
        pts.append(pts[-1] + s)
    a = np.asarray(pts)
    tail = a[start:]
    a[start:] = tail[0] + (tail - tail[0]) * (size - tail[0]) / (tail[-1] - tail[0])
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--R", type=float, required=True, help="needle radius, mm")
    ap.add_argument("--h", type=float, required=True, help="fine-zone spacing, mm")
    ap.add_argument("--depth", type=float, required=True, help="mm")
    ap.add_argument("--step", type=float, default=0.05, help="mm")
    ap.add_argument("--mu", type=float, default=3660.0, help="Pa")
    ap.add_argument("--nu", type=float, default=0.49)
    ap.add_argument("--jm", type=float, default=0.0, help="Gent's Jm (0: neo-Hookean)")
    ap.add_argument("--volumetric", default="mean")
    ap.add_argument("--block", type=float, default=60.0, help="block radius and height, mm")
    ap.add_argument("--fine-r", type=float, default=5.0, help="mm")
    ap.add_argument("--fine-z", type=float, default=8.0, help="mm")
    ap.add_argument("--cap", type=float, default=2.0, help="largest cell, mm")
    ap.add_argument("--contact", default="nodal", choices=["nodal", "edge"])
    ap.add_argument("--edge-points", type=int, default=4)
    ap.add_argument("--kfac", type=float, default=100.0, help="penalty stiffness k = kfac E / h")
    ap.add_argument("--max-newton", type=int, default=60)
    ap.add_argument("--plant-radius", type=float, default=1.0, help="the contact's tip radius times this (a plant)")
    ap.add_argument("--axis-h", type=float, default=0.0, help="mm: the column spacing near the axis (0: h)")
    ap.add_argument("--axis-zone", type=float, default=0.0, help="mm: how far from the axis --axis-h holds")
    ap.add_argument("--fillet", type=float, default=1.0, help="the tip's edge radius over R (1: the sphere-tipped needle; "
                    "small: a flat punch)")
    ap.add_argument("--save-every", type=float, default=0.5, help="mm between saved fields")
    ap.add_argument("--stop-after", type=float, default=0.0, help="minutes: stop (state saved) once past this")
    ap.add_argument("--resume", default="", help="a previous run's OUT.json: continue from its saved state")
    a = ap.parse_args()
    mm = 1e-3
    R, h = a.R * mm, a.h * mm
    young = 2 * a.mu * (1 + a.nu)
    mu, kappa = shear_and_bulk(young, a.nu)
    mat = Material(mu=mu, kappa=kappa, gc=1.0, l=1.0, k_res=1e-6, stretch_limit=2.0,
                   gent_jm=(a.jm if a.jm > 0 else None))
    x = (axis_columns(a.block * mm, h, a.fine_r * mm, a.cap * mm, a.axis_h * mm, a.axis_zone * mm) if a.axis_h > 0
         else grid_coords(a.block * mm, h, a.fine_r * mm, a.cap * mm))
    y = a.block * mm - grid_coords(a.block * mm, h, a.fine_z * mm, a.cap * mm)[::-1]
    x[0], x[-1], y[0], y[-1] = 0.0, a.block * mm, 0.0, a.block * mm
    block = axisymmetric_block(radius=a.block * mm, height=a.block * mm, h=h, fine_r=0.0, fine_z=0.0,
                               predictor="proportional", x=x, y=y)
    g = block.grid
    w = np.broadcast_to(nodal_ring_area(g.x)[:, None], (g.nx + 1, g.ny + 1)).copy()
    rc = R * a.plant_radius
    tip = ct.Indenter("punch", radius=rc, fillet=rc * a.fillet, top=a.block * mm)
    true_tip = ct.Indenter("punch", radius=R, fillet=R * a.fillet, top=a.block * mm)
    extra = {} if a.contact == "nodal" else {"edge_points": a.edge_points}
    c = ct.Contact(indenter=tip, stiffness=a.kfac * young / h, weights=w, active=np.ones_like(w), friction=0.0,
                   **extra)
    vol = {} if a.volumetric == "gauss" else {"volumetric": a.volumetric}
    solver = Solver(replace(block, contact=c), "elastic", newton_regularise=True, max_newton=a.max_newton, **vol)
    X = solver.X
    I, J = solver.I, solver.J
    s = jnp.linspace(0.0, 1.0, SAMPLES)

    @jax.jit
    def diagnostics(u, depth):
        xy = X + u
        phi_n = ct.signed_distance(true_tip, xy[..., 0], xy[..., 1], depth)
        top = xy[:, -1]
        pts = top[:-1, None, :] * (1 - s)[None, :, None] + top[1:, None, :] * s[None, :, None]
        phi_e = ct.signed_distance(true_tip, pts[..., 0], pts[..., 1], depth)
        F2, Ftt = jax.vmap(axi.deformation)(u[I, J], solver.x0, solver.dx, solver.dy)
        Jd = jnp.linalg.det(F2) * Ftt
        C = jnp.einsum("egki,egkj->egij", F2, F2)
        tr, det = C[..., 0, 0] + C[..., 1, 1], C[..., 0, 0] * C[..., 1, 1] - C[..., 0, 1] * C[..., 1, 0]
        top_ev = 0.5 * tr + jnp.sqrt(jnp.maximum(0.25 * tr * tr - det, 0.0))
        lmax = jnp.maximum(jnp.sqrt(top_ev), Ftt)
        return (jnp.min(phi_n), jnp.min(phi_n[:, :-1]), jnp.sum(phi_n[:, :-1] < 0), jnp.min(phi_e),
                jnp.argmin(jnp.min(phi_e, axis=1)), jnp.min(Jd), jnp.sum(Jd < 0.05), jnp.max(lmax),
                jnp.argmax(jnp.max(lmax, axis=1)))

    state = solver.initial()
    n = int(round(a.depth / a.step))
    rows, saved = [], {}
    t0 = time.perf_counter()
    k0 = 0
    if a.resume:
        prev = json.load(open(a.resume))
        rows = prev["rows"]
        st = np.load(a.resume.replace(".json", "_state.npz"))
        k0 = int(st["k"])
        state = state._replace(u=jnp.asarray(st["u"]), delta=jnp.asarray(float(st["delta"])))
        old = np.load(a.resume.replace(".json", "_fields.npz"))
        saved = {key: old[key] for key in old.files if key.startswith("u_")}
        t0 -= rows[-1]["wall_s"] if rows else 0.0
        print(f"resumed at step {k0}, depth {float(st['delta']) / mm:.4f} mm", flush=True)

    def write():
        with open(a.out, "w") as fh:
            json.dump(dict(args=vars(a), grid=[g.nx, g.ny], young=young, mu=mu, kappa=kappa, rows=rows), fh, indent=0)

    next_save = a.save_every * (int(round(k0 * a.step / a.save_every + 1e-9)) + 1)
    t_start = time.perf_counter()
    for k in range(k0 + 1, n + 1):
        depth = k * a.step * mm
        state, out = solver.step(state, jnp.asarray(depth), mat)
        dg = [float(v) for v in diagnostics(state.u, jnp.asarray(depth))]
        f = float(out.force)
        row = dict(depth_mm=depth / mm, force_N=f, newton=int(out.newton), capped=int(out.newton_capped),
                   node_pen_mm=-dg[0] / mm, sub_pen_mm=-dg[1] / mm, sub_inside=int(dg[2]), edge_pen_mm=-dg[3] / mm,
                   edge_pen_at=int(dg[4]), J_min=dg[5], n_J_small=int(dg[6]), lmax=dg[7],
                   lmax_el=int(dg[8]), finite=bool(np.isfinite(f)), wall_s=time.perf_counter() - t0)
        rows.append(row)
        write()
        print(json.dumps({k2: (round(v, 6) if isinstance(v, float) else v) for k2, v in row.items()}), flush=True)
        if depth / mm >= next_save - 1e-9 or k == n:
            saved[f"u_{k}"] = np.asarray(state.u, dtype=np.float32)
            np.savez_compressed(a.out.replace(".json", "_fields.npz"), x=g.x, y=g.y, **saved)
            next_save += a.save_every
        if not np.isfinite(f):
            break
        if a.stop_after and (time.perf_counter() - t_start) / 60.0 > a.stop_after and k < n:
            np.savez(a.out.replace(".json", "_state.npz"), u=np.asarray(state.u), delta=float(state.delta), k=k)
            np.savez_compressed(a.out.replace(".json", "_fields.npz"), x=g.x, y=g.y, **saved)
            print(f"stopped after {a.stop_after} min at step {k}", flush=True)
            break


if __name__ == "__main__":
    main()
