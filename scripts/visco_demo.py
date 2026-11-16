"""The viscoelastic clot's demo effects, with the engine directly (2-D and axisymmetric):

    python scripts/visco_demo.py tear --out FILE [--h 0.125] [--grip-rate 0.2] (mm/s) [--material elastic|visco|slow]
    python scripts/visco_demo.py puncture --out FILE [--speed 0.1] (mm/s) [--material elastic|visco] [--hold 0] (s)
        [--hold-depth 0] (mm; 0: hold where the run stops) [--h 0.1] [--max-depth 3.0]

``tear``: notch group 4 of the pure-shear study (notch 11.9 mm, l 0.25 mm), its force and the crack's extension against
the grip's displacement. ``puncture``: the tutorial's axisymmetric block (radius and height 5 mm, bottom fixed) pressed by
its sharp cone (shaft radius 0.5 mm, apex radius 0.02 mm, half-angle 15 degrees), frictionless, to past the force's drop or
``--max-depth``; with ``--hold`` the cone then stops at ``--hold-depth`` (or where it is) and the force is recorded while
the clot relaxes (the push-out force).

Materials: ``elastic`` is the replication's (Yeoh C1 363, C2 434 Pa, Gc 0.83 J/m^2, at 0.2 mm/s); ``visco`` takes those
as the response at 0.2 mm/s: long-term s (C1, C2) with the Prony terms (``--beta``, ``--tau``) and s from
``studies.visco_clot.rate_consistent_scale``; ``slow`` is the slow-rate control's (long-term = the elastic constants,
the rate 1e-6 times the relaxation's). The axisymmetric block is compressible (nu 0.45, the tutorial record's
placeholder): kappa = 2 mu (1 + nu) / (3 (1 - 2 nu)) with the elastic record's mu (2 C1), unscaled.
"""
import argparse
import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from mime_softtissue.engine import contact as ct  # noqa: E402
from mime_softtissue.engine.fracture.phase_field import discretisation_toughness  # noqa: E402
from mime_softtissue.engine.loadpath import run  # noqa: E402
from mime_softtissue.engine.material import Material, Prony  # noqa: E402
from mime_softtissue.engine.solver import Solver  # noqa: E402
from mime_softtissue.engine.specimen import axisymmetric_block, morph_columns, with_indenter  # noqa: E402
from mime_softtissue.studies import pure_shear as ps  # noqa: E402
from mime_softtissue.studies import visco_clot as vc  # noqa: E402

C1, C2, GC, L = 363.0, 434.0, 0.83, 0.25e-3
NU = 0.45

ap = argparse.ArgumentParser()
ap.add_argument("case", choices=("tear", "puncture"))
ap.add_argument("--out", required=True)
ap.add_argument("--material", default="visco", choices=("elastic", "visco", "slow"))
ap.add_argument("--beta", type=float, nargs="+", default=None)
ap.add_argument("--tau", type=float, nargs="+", default=list(vc.PUBLISHED_TAU))
ap.add_argument("--h", type=float, default=None, help="mm")
ap.add_argument("--grip-rate", type=float, default=0.2, help="mm/s (tear)")
ap.add_argument("--step", type=float, default=None, help="mm")
ap.add_argument("--speed", type=float, default=0.1, help="mm/s (puncture)")
ap.add_argument("--max-depth", type=float, default=3.0, help="mm")
ap.add_argument("--hold", type=float, default=0.0, help="s")
ap.add_argument("--hold-depth", type=float, default=0.0, help="mm")
ap.add_argument("--hold-steps", type=int, default=60)
ap.add_argument("--wall", type=float, default=5400.0)
ap.add_argument("--scale", type=float, default=None, help="the long-term scale outright (a diagnostic), for any material")
a = ap.parse_args()

prony = None
scale, scale_rms = 1.0, 0.0
if a.material != "elastic":
    if a.beta is None:
        raise SystemExit("--beta is required for a viscoelastic material")
    prony = Prony(jnp.asarray(a.beta), jnp.asarray(a.tau))
    if a.material == "visco":
        scale, scale_rms = vc.rate_consistent_scale(prony)
if a.scale is not None:
    scale, scale_rms = a.scale, float("nan")
rec = {"args": vars(a), "long_term_scale": scale, "scale_rms": scale_rms, "c1": C1, "c2": C2, "gc": GC, "l": L}
t0 = time.time()


def material(h: float, kappa_ratio: float = 0.0) -> Material:
    """The bulk modulus is the elastic record's (kappa_ratio times its mu = 2 C1), whatever the long-term scale: the
    viscous material differs from the elastic one only in its isochoric response."""
    mu = 2 * C1 * scale
    return Material(mu=jnp.asarray(mu), kappa=jnp.asarray(kappa_ratio * 2 * C1),
                    gc=jnp.asarray(discretisation_toughness(GC, h, L)), l=jnp.asarray(L), k_res=jnp.asarray(1e-6),
                    stretch_limit=jnp.asarray(10.0), c2=jnp.asarray(C2 * scale), prony=prony)


def slow_rate(step: float) -> float:
    return 1e-6 * step / max(a.tau)


if a.case == "tear":
    h = (a.h or 0.125) * 1e-3
    step = (a.step or 0.05) * 1e-3
    notch = 11.9e-3
    sp = ps.specimen(notch, h, L)
    rate = None if a.material == "elastic" else (slow_rate(step) if a.material == "slow" else a.grip_rate * 1e-3)
    s = Solver(sp, "phase_field", tol_d=1e-4, max_staggered=300, newton_regularise=True, rate=rate)
    deltas = np.arange(1, int(round(4.5e-3 / step)) + 1) * step
    r = run(s, material(h), deltas, notch=notch, keep_fields=True, stop_below=0.03)
    x_nodes = np.asarray(morph_columns(sp.grid.x, sp.morph, notch))
    ext = np.array([ps.crack_extension(dd, x_nodes, notch) for dd in r["damage"]])
    rec.update({k: np.asarray(r[k]).tolist() for k in ("delta", "force", "energy", "crack_energy", "branch_dissipation",
                                                       "iterations", "newton", "newton_capped", "dt")})
    rec["extension_mm"] = (ext * 1e3).tolist()
    rec["ligament_mm"] = (40e-3 - notch) * 1e3
    rec["cells"] = [sp.grid.nx, sp.grid.ny]
else:
    h = (a.h or 0.1) * 1e-3
    step = (a.step or 0.02) * 1e-3
    RB = HB = 5e-3
    kappa_ratio = 2 * (1 + NU) / (3 * (1 - 2 * NU))  # K / mu
    mat = material(h, kappa_ratio)
    young = 2 * (2 * C1) * (1 + NU)  # the penalty's scale: the elastic record's Young's modulus
    cone = ct.Indenter("cone", radius=0.5e-3, fillet=0.02e-3, top=HB, half_angle=float(np.deg2rad(15.0)))
    block = axisymmetric_block(radius=RB, height=HB, h=h, fine_r=1e-3, fine_z=1e-3, predictor="proportional")
    sp = with_indenter(block, cone, stiffness=100.0 * young / h)
    rate = None if a.material == "elastic" else (slow_rate(step) if a.material == "slow" else a.speed * 1e-3)
    solver = Solver(sp, "phase_field", tol_d=1e-5, max_staggered=400, newton_regularise=True, rate=rate)
    state = solver.initial(mat=mat)
    rows = []
    peak, peak_at = 0.0, 0
    depths = np.arange(1, int(round(a.max_depth * 1e-3 / step)) + 1) * step
    hold_depth = a.hold_depth * 1e-3 if a.hold_depth > 0 else None
    status = "max depth"
    for n, dl in enumerate(depths):
        if hold_depth is not None and dl > hold_depth + 1e-12:
            status = "hold depth"
            break
        state, out = solver.step(state, jnp.asarray(dl), mat)
        row = {k: float(v) for k, v in out._asdict().items()}
        row["depth_mm"] = dl * 1e3
        row["d_max"] = float(jnp.max(state.d))
        rows.append(row)
        f = row["force"]
        print(f"{n:3d} depth {dl*1e3:6.3f} F {f*1e3:9.4f} mN it {int(row['iterations']):4d} dmax {row['d_max']:.3f} "
              f"{time.time()-t0:.0f}s", flush=True)
        if f > peak:
            peak, peak_at = f, n
        if peak_at < n and f < 0.5 * peak and hold_depth is None:
            status = "force dropped below half its peak"
            break
        if not np.isfinite(f) or time.time() - t0 > a.wall:
            status = "non-finite" if not np.isfinite(f) else "wall"
            break
    rec["rows"] = rows
    rec["status"] = status
    rec["cells"] = [sp.grid.nx, sp.grid.ny]
    if a.hold > 0:
        # the hold: log-spaced steps over ``hold`` seconds at the reached depth
        edges = np.concatenate([[0.0], np.geomspace(0.01, a.hold, a.hold_steps)])
        held = []
        depth = jnp.asarray(rows[-1]["depth_mm"] * 1e-3)
        for k, dt in enumerate(np.diff(edges)):
            state, out = solver.step(state, depth, mat, None, None, jnp.asarray(dt))
            held.append({"t_s": float(edges[k + 1]), "force": float(out.force), "energy": float(out.energy),
                         "branch_dissipation": float(out.branch_dissipation), "iterations": int(out.iterations),
                         "d_max": float(jnp.max(state.d))})
        rec["hold"] = held
        print("hold: F(0) %.4f mN, F(end) %.4f mN" % (rows[-1]["force"] * 1e3, held[-1]["force"] * 1e3))
rec["wall_s"] = time.time() - t0
Path(a.out).write_text(json.dumps(rec))
print("wall", rec["wall_s"])
