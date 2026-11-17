"""TENSION's needle and punch driver (STRENGTH-CAL's ``strengthcal/run.py`` with ``--criterion tension``: the
tension-only strength surface, ``strength.Tension``; and the largest principal Biot stress recorded each step): the axisymmetric engine called
directly, a sphere-tipped needle (``--tip sphere``: a punch whose fillet is its radius) or a flat punch (``--tip flat``,
its edge rounded by ``--fillet``), pressed frictionless into a cylindrical block whose grid is uniform at ``h`` within
``--fine-r`` of the axis and ``--fine-z`` of the top and graded (ratio 1.25) to cells of at most ``--coarse`` beyond.
``--sts`` (kPa) switches on the strength-surface phase field (``engine.fracture.strength``, Gc raw); ``--mode elastic``
runs the elastic twin (no damage; the strength fields still recorded when ``--sts`` is given).

Each row records the force, the crack area, the iteration counts, the step's wall time and the process's peak memory;
``--fields-at`` saves the strength and stretch fields at chosen depths. ``--max-wall`` stops cleanly after that many
seconds (the rows so far are kept). The run is the GelBlock's own construction otherwise: proportional predictor,
nodal penalty 100 E/h, lagged pressure with zero friction (lag relaxation 0.5), tol_d 1e-5, viscosity 0.1 Gc/l,
regularised Newton; volumetric term at the element's mean dilatation unless ``--volumetric gauss``.

python run.py OUT.json --tip sphere --R 0.5 --h 0.01 --l 0.02 --sts 20 --criterion tension --depth 6"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mime_softtissue.engine import contact as ct  # noqa: E402
from mime_softtissue.engine.fracture import strength as strength_surface  # noqa: E402
from mime_softtissue.engine.kinematics import axisymmetric as axi  # noqa: E402
from mime_softtissue.engine.kinematics.axisymmetric import nodal_ring_area  # noqa: E402
from mime_softtissue.engine.material import Material, shear_and_bulk  # noqa: E402
from mime_softtissue.engine.solver import Solver  # noqa: E402
from mime_softtissue.engine.specimen import axisymmetric_block  # noqa: E402


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tip", choices=("sphere", "flat"), default="sphere")
    ap.add_argument("--R", type=float, required=True, help="mm")
    ap.add_argument("--fillet", type=float, default=0.0, help="mm, the flat punch's edge radius")
    ap.add_argument("--h", type=float, required=True, help="mm")
    ap.add_argument("--l", type=float, required=True, help="mm")
    ap.add_argument("--gc", type=float, default=27.6)
    ap.add_argument("--mu", type=float, default=3660.0, help="Pa")
    ap.add_argument("--nu", type=float, default=0.49)
    ap.add_argument("--jm", type=float, default=0.0, help="Gent's Jm (0: neo-Hookean)")
    ap.add_argument("--block", type=float, default=50.0, help="mm")
    ap.add_argument("--fine-r", type=float, required=True, help="mm")
    ap.add_argument("--fine-z", type=float, required=True, help="mm")
    ap.add_argument("--coarse", type=float, default=1.0, help="mm, the largest cell")
    ap.add_argument("--step", type=float, default=0.05, help="mm")
    ap.add_argument("--depth", type=float, required=True, help="mm")
    ap.add_argument("--mode", default="phase_field")
    ap.add_argument("--volumetric", default="mean")
    ap.add_argument("--stop-drop", type=float, default=0.0, help="stop once the force falls this share below its max")
    ap.add_argument("--sts", type=float, default=0.0, help="kPa: the uniaxial tensile strength (Biot)")
    ap.add_argument("--ratio", type=float, default=1.5, help="s_hs / s_ts (Drucker-Prager only)")
    ap.add_argument("--gate", choices=("i1", "s1"), default="i1", help="tension: where the elastic energy drives damage")
    ap.add_argument("--criterion", choices=("dp", "tension"), default="dp",
                    help="dp: Drucker-Prager (strength.Strength); tension: the largest principal stress (strength.Tension)")
    ap.add_argument("--delta-factor", type=float, default=1.0)
    ap.add_argument("--max-wall", type=float, default=0.0, help="s: stop after this wall time (0: none)")
    ap.add_argument("--fields-at", type=float, nargs="*", default=[], help="mm: depths whose fields are saved")
    ap.add_argument("--max-staggered", type=int, default=400)
    ap.add_argument("--max-newton", type=int, default=40)
    ap.add_argument("--ramp", type=int, default=0, help="start with this many halvings of the step, doubling back")
    ap.add_argument("--cut", type=int, default=0,
                    help="halve a step up to this many times where it does not converge (staggered or Newton cap, or "
                         "a non-finite force), from the last accepted state")
    ap.add_argument("--newton-stall", type=float, default=0.0,
                    help="Newton's stagnation stop, in units of h (solvers.newton's stall_step; 0 off)")
    ap.add_argument("--edge-points", type=int, default=0,
                    help="the contact penalty at this many Gauss points on each top edge (spike-deep-elastic); 0 nodal")
    ap.add_argument("--contact-k", type=float, default=100.0, help="the penalty stiffness, in units of E / h")
    ap.add_argument("--no-lag", action="store_true",
                    help="no lagged pressure (the friction is zero, so the lag only adds staggered iterations)")
    a = ap.parse_args()
    mm = 1e-3
    R, h, l = a.R * mm, a.h * mm, a.l * mm
    young = 2 * a.mu * (1 + a.nu)
    mu, kappa = shear_and_bulk(young, a.nu)
    if a.mode == "phase_field" and not a.sts:
        raise SystemExit("this driver runs the strength surface (--sts) or the elastic twin (--mode elastic)")
    mat = Material(mu=mu, kappa=kappa, gc=a.gc, l=l, k_res=1e-6, stretch_limit=2.0,
                   gent_jm=(a.jm if a.jm > 0 else None))
    if a.sts and a.criterion == "tension":
        mat = mat._replace(strength=strength_surface.tension_coefficients(mat, a.sts * 1e3, h,
                                                                          delta_factor=a.delta_factor, gate=a.gate))
        st = mat.strength
        q = st.delta * a.gc / (8.0 * l)
        w_ts = 0.5 * (st.beta * st.s_ts + 3.0 * q)
        validity = 3.0 * a.gc / (16.0 * w_ts)
        print(json.dumps(dict(strength=mat.strength._asdict(), w_ts=w_ts, l_validity_mm=validity / mm,
                              l_over_validity=l / validity, compressive_factor=3.0 * q / (3.0 * q - 2.0 * w_ts))),
              flush=True)
    elif a.sts:
        mat = mat._replace(strength=strength_surface.coefficients(mat, a.sts * 1e3, a.ratio * a.sts * 1e3, h,
                                                                  delta_factor=a.delta_factor))
        st = mat.strength  # W_hs and W_ts read back from beta1, beta2 (eq. 16), not recomputed
        q = st.delta * a.gc / (8.0 * l)
        w_hs = 1.5 * st.s_hs * (st.beta1 + q / st.s_hs)
        w_ts = st.s_ts / (2.0 * np.sqrt(3.0)) * (st.beta2 + np.sqrt(3.0) * (3 * st.s_hs - st.s_ts) / (st.s_hs * st.s_ts) * q
                                                 + 2.0 * w_hs / (np.sqrt(3.0) * st.s_hs))
        validity = 3.0 * a.gc / (16.0 * w_ts)
        print(json.dumps(dict(strength=mat.strength._asdict(), w_ts=w_ts, l_validity_mm=validity / mm,
                              l_over_validity=l / validity)), flush=True)
    block = axisymmetric_block(radius=a.block * mm, height=a.block * mm, h=h, fine_r=a.fine_r * mm,
                               fine_z=a.fine_z * mm, cap=a.coarse / a.h, predictor="proportional")
    g = block.grid
    w = np.broadcast_to(nodal_ring_area(g.x)[:, None], (g.nx + 1, g.ny + 1)).copy()
    fillet = R if a.tip == "sphere" else a.fillet * mm
    tip = ct.Indenter("punch", radius=R, fillet=fillet, top=a.block * mm)
    c = ct.Contact(indenter=tip, stiffness=a.contact_k * young / h, weights=w, active=np.ones_like(w), friction=0.0,
                   slip=1e-2 * h, always_sliding=not a.no_lag, edge_points=a.edge_points)
    solver = Solver(replace(block, contact=c), a.mode, tol_d=1e-5, viscosity=0.1 * a.gc / l, newton_regularise=True,
                    lag_relaxation=0.5, volumetric=a.volumetric, max_staggered=a.max_staggered,
                    max_newton=a.max_newton, newton_stall=a.newton_stall * h)
    info = dict(nx=g.nx, ny=g.ny, n_el=g.n_elements, dofs=2 * (g.nx + 1) * (g.ny + 1), block_m=2 * (g.ny + 1),
                blocks=g.nx + 1, nproc=os.cpu_count(), rss_after_build_mb=rss_mb())
    print(json.dumps(info), flush=True)

    def fields(u, d, hist):
        """Per Gauss point: the onset function (Pa), the Drucker-Prager F(S) of the degraded Biot stress (Pa), the
        equivalent tensile strength seq (the s_ts at which the point sits on the surface, s_hs / s_ts held), the
        isochoric x, the largest principal stretch, J and the damage."""
        dq = solver.damage_at_gauss(d)

        def element(ue, dqe, x0, dx, dy):
            F2, Ftt = axi.deformation(ue, x0, dx, dy)

            def point(f, t, dg):
                lam = strength_surface.principal_stretches(f, t)
                Jd = lam[0] * lam[1] * lam[2]
                x = Jd ** (-2.0 / 3.0) * jnp.sum(lam * lam) - 3.0
                s = strength_surface.biot_principal(lam, dg, mat)
                s1 = jnp.max(s)
                c1 = jnp.max(lam * s) / Jd  # the largest principal Cauchy stress
                if mat.strength is None or a.criterion == "tension":
                    return x, jnp.max(lam), Jd, x * 0, x * 0, jnp.sum(s), s1, c1
                st = mat.strength
                J2 = ((s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2) / 6.0
                g1 = st.s_ts / (jnp.sqrt(3.0) * (3 * st.s_hs - st.s_ts))
                g0 = -jnp.sqrt(3.0) * st.s_hs * st.s_ts / (3 * st.s_hs - st.s_ts)
                lhs = jnp.sqrt(jnp.maximum(J2, 0.0)) + g1 * jnp.sum(s)
                return x, jnp.max(lam), Jd, lhs + g0, lhs / (-g0 / st.s_ts), jnp.sum(s), s1, c1
            return jax.vmap(point)(F2, Ftt, dqe)
        x, lmax, Jd, dp, seq, I1, s1, c1 = jax.vmap(element)(u[solver.I, solver.J], dq, solver.x0, solver.dx,
                                                              solver.dy)
        if mat.strength is not None:
            st = mat.strength
            ce, tension = solver.strength_driving(u, d, mat)
            onset = 2.0 * jnp.where(tension, hist, 0.0) - ce - 3.0 * st.delta * mat.gc / (8.0 * mat.l)
        else:
            onset = jnp.zeros_like(x)
        return x, lmax, Jd, dp, seq, I1, onset, dq, s1, c1
    fields_j = jax.jit(fields)
    centres = np.stack([np.asarray(solver.x0) + 0.5 * np.asarray(solver.dx),
                        np.asarray(g.y[:-1][g.element_columns()[1]]) + 0.5 * np.asarray(solver.dy)], axis=1)
    state = solver.initial()
    rows: list[dict] = []
    t0 = time.perf_counter()
    fmax = 0.0
    saved = set()

    def write(stop: str = "") -> None:
        with open(a.out, "w") as fh:
            json.dump(dict(args=vars(a), grid=[g.nx, g.ny], info=info, young=young, rows=rows, stop=stop,
                           strength=None if mat.strength is None else mat.strength._asdict()), fh, indent=1)

    stop = "depth"
    done = 0.0  # mm, the last accepted depth
    level = a.ramp  # the step is a.step / 2**level
    k = 0
    while done < a.depth - 1e-9:
        k += 1
        ts = time.perf_counter()
        retries = 0
        while True:
            inc = min(a.step / 2 ** level, a.depth - done)
            trial, out = solver.step(state, jnp.asarray((done + inc) * mm), mat)
            f = float(out.force)
            bad = (not np.isfinite(f)) or (a.cut and (int(out.iterations) >= a.max_staggered or int(out.newton_capped) > 0))
            if bad and a.cut and level < a.ramp + a.cut:
                level += 1
                retries += 1
                continue
            break
        state = trial
        done += inc
        depth = done * mm
        step_s = time.perf_counter() - ts
        x, lmax, Jd, dp, seq, I1, onset, dq, s1, c1 = (np.asarray(v) for v in fields_j(state.u, state.d,
                                                                                       state.history))
        fmax = max(fmax, f)
        ke = lambda arr: [float(v) for v in centres[int(np.argmax(arr.max(axis=1)))] / mm]
        row = dict(depth_mm=depth / mm, force_N=f, crack_area_mm2=float(out.crack_energy) / float(a.gc) * 1e6,
                   staggered=int(out.iterations), newton=int(out.newton), capped=int(out.newton_capped),
                   step_s=step_s, wall_s=time.perf_counter() - t0, rss_mb=rss_mb(), inc_mm=inc, retries=retries,
                   accepted_bad=bool(bad),
                   x_max=float(x.max()), lmax=float(lmax.max()), J_min=float(Jd.min()), d_max=float(dq.max()),
                   d_at=ke(dq))
        row.update(s1_max=float(s1.max()), s1_at=ke(s1), s1_I1pos_max=float(np.where(I1 >= 0, s1, -np.inf).max()),
                   s1_I1pos_at=ke(np.where(I1 >= 0, s1, -np.inf)),
                   s1_I1neg_max=float(np.where(I1 < 0, s1, -np.inf).max()), c1_max=float(c1.max()), c1_at=ke(c1))
        if mat.strength is not None:
            row.update(onset_max=float(onset.max()), onset_at=ke(onset))
            if a.criterion == "dp":
                row.update(seq_max=float(seq.max()), seq_at=ke(seq),
                           seq_I1pos_max=float(np.where(I1 >= 0, seq, -np.inf).max()))
        rows.append(row)
        for fd in a.fields_at:
            if fd not in saved and depth / mm >= fd - 1e-9:
                saved.add(fd)
                np.savez_compressed(a.out.replace(".json", f"_f{fd:g}.npz"), centres=centres, x=x.astype(np.float32),
                                    lmax=lmax.astype(np.float32), J=Jd.astype(np.float32), seq=seq.astype(np.float32),
                                    I1=I1.astype(np.float32), onset=onset.astype(np.float32),
                                    s1=s1.astype(np.float32), c1=c1.astype(np.float32),
                                    d=dq.astype(np.float32), depth=depth / mm)
        write()
        print(json.dumps({k2: (round(v, 6) if isinstance(v, float) else v) for k2, v in row.items()}), flush=True)
        if a.stop_drop and f < (1 - a.stop_drop) * fmax:
            stop = "drop"
            break
        if a.max_wall and time.perf_counter() - t0 > a.max_wall:
            stop = "max_wall"
            break
        if not np.isfinite(f):
            stop = "nan"
            break
        if level > 0 and not bad and retries == 0:
            level -= 1  # grow back towards the full step after a converged one
    write(stop)


if __name__ == "__main__":
    main()
