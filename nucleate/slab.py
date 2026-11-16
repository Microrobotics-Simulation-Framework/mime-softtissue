"""NUCLEATE control: a crack grows by Gc. An axisymmetric annular slab (radius r_i to r_o, height H) clamped on its
top and bottom faces and pulled apart, with a pre-cut crack in its mid-plane from the inner face out to a0 (the
"pure-shear" test of Rivlin and Thomas, as Kamarei, Kumar and Lopez-Pamies 2024 calibrate their delta with, here
axisymmetric). Far from the free faces the slab ahead of the crack is in uniaxial strain (lambda, 1, 1); a crack that
grows releases H W(lambda) per unit area (W of that state), whatever its length, so it runs through at the stretch
lambda_c with H W(lambda_c) = Gc. The measured toughness is H W(lambda at the force peak).

python slab.py OUT.json [--sts KPA --ratio R | --at2] [--l MM] [--h MM] [--H MM] [--delta-plant F]"""
from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mime_softtissue.engine import material as law  # noqa: E402
from mime_softtissue.engine.fracture import strength as S  # noqa: E402
from mime_softtissue.engine.fracture.phase_field import discretisation_toughness  # noqa: E402
from mime_softtissue.engine.grid import Grid  # noqa: E402
from mime_softtissue.engine.material import Material, shear_and_bulk  # noqa: E402
from mime_softtissue.engine.solver import Solver  # noqa: E402
from mime_softtissue.engine.specimen import Load, Reaction, Specimen  # noqa: E402


def graded(H: float, h: float, band: float, ratio: float = 1.3) -> np.ndarray:
    """Rows: spacing h within ``band`` of mid-height, growing by ``ratio`` towards each face."""
    half = [0.0]
    while half[-1] < band - 1e-12:
        half.append(half[-1] + h)
    step = h
    while half[-1] < H / 2 - 1e-12:
        step = min(step * ratio, H / 2 - half[-1]) if H / 2 - half[-1] > step * ratio * 1.5 else H / 2 - half[-1]
        half.append(half[-1] + step)
    half = np.asarray(half)
    return np.concatenate([H / 2 - half[::-1], H / 2 + half[1:]])


def slab(r_i: float, r_o: float, H: float, h: float, band: float) -> tuple[Specimen, int]:
    x = np.linspace(r_i, r_o, int(round((r_o - r_i) / h)) + 1)
    y = graded(H, h, band)
    g = Grid(x, y)
    nx, ny = len(x) - 1, len(y) - 1
    free = np.ones((nx + 1, ny + 1, 2))
    free[:, 0, :] = 0.0
    free[:, -1, :] = 0.0
    top = (np.arange(nx + 1), np.full(nx + 1, ny))
    sp = Specimen(grid=g, free=free, thickness=1.0, force_factor=1.0, load=Load(top, sense=1.0, predictor="spread"),
                  reaction=Reaction(top, sign=1.0), kinematics="axisymmetric", energy_factor=1.0)
    return sp, int(np.argmin(np.abs(y - H / 2)))


def w_uniaxial_strain(mat: Material, lam: float) -> float:
    plus, minus = law.energy_parts(jnp.asarray(lam), jnp.asarray(lam * lam + 2.0), (), mat)
    return float(plus + minus)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sts", type=float, default=0.0, help="kPa (0: AT2)")
    ap.add_argument("--ratio", type=float, default=1.5)
    ap.add_argument("--mu", type=float, default=3660.0)
    ap.add_argument("--nu", type=float, default=0.45)
    ap.add_argument("--gc", type=float, default=27.0)
    ap.add_argument("--l", type=float, default=0.5, help="mm")
    ap.add_argument("--h", type=float, default=0.1, help="mm")
    ap.add_argument("--H", type=float, default=4.0, help="mm")
    ap.add_argument("--width", type=float, default=20.0, help="mm: r_o - r_i")
    ap.add_argument("--ri", type=float, default=20.0, help="mm")
    ap.add_argument("--a0", type=float, default=6.0, help="mm")
    ap.add_argument("--band", type=float, default=1.5, help="mm")
    ap.add_argument("--dlam", type=float, default=0.005)
    ap.add_argument("--lam-max", type=float, default=2.2)
    ap.add_argument("--delta-plant", type=float, default=1.0)
    a = ap.parse_args()
    mm = 1e-3
    H, h, l = a.H * mm, a.h * mm, a.l * mm
    sp, jmid = slab(a.ri * mm, (a.ri + a.width) * mm, H, h, a.band * mm)
    g = sp.grid
    young = 2 * a.mu * (1 + a.nu)
    mu, kappa = shear_and_bulk(young, a.nu)
    gc_in = a.gc if a.sts else discretisation_toughness(a.gc, h, l)
    mat = Material(mu=mu, kappa=kappa, gc=gc_in, l=l, k_res=1e-6, stretch_limit=10.0)
    if a.sts:
        st = S.coefficients(mat, a.sts * 1e3, a.ratio * a.sts * 1e3, h)
        mat = mat._replace(strength=st._replace(delta=st.delta * a.delta_plant))
    solver = Solver(sp, "phase_field", tol_d=1e-5, viscosity=0.1 * a.gc / l, newton_regularise=True, max_staggered=400)
    state = solver.initial()
    d0 = np.zeros((g.nx + 1, g.ny + 1))
    d0[np.asarray(g.x) <= (a.ri + a.a0) * mm + 1e-12, jmid] = 1.0
    state = state._replace(d=jnp.asarray(d0))
    lams = np.arange(1.0 + a.dlam, a.lam_max + 1e-12, a.dlam)
    rows, t0, fmax = [], time.perf_counter(), 0.0
    for lam in lams:
        state, out = solver.step(state, jnp.asarray((lam - 1.0) * H), mat)
        f = float(out.force)
        fmax = max(fmax, f)
        rows.append(dict(lam=float(lam), force_N=f, crack_area_mm2=float(out.crack_energy) / float(gc_in) * 1e6,
                         staggered=int(out.iterations), capped=int(out.newton_capped), d_max=float(jnp.max(state.d)),
                         wall_s=time.perf_counter() - t0))
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
        if f < 0.5 * fmax:
            break
    k = int(np.argmax([r["force_N"] for r in rows]))
    lam_c = rows[k]["lam"]
    gc_meas = H * w_uniaxial_strain(mat._replace(strength=None), lam_c)
    lam_exact = next(x for x in np.arange(1.0, 4.0, 1e-5) if H * w_uniaxial_strain(mat, x) >= a.gc)
    res = dict(args=vars(a), grid=[g.nx, g.ny], strength=None if mat.strength is None else mat.strength._asdict(),
               lam_c=lam_c, gc_measured=gc_meas, gc_over_input=gc_meas / a.gc, lam_griffith=float(lam_exact), rows=rows)
    print(json.dumps({q: res[q] for q in ("lam_c", "gc_measured", "gc_over_input", "lam_griffith")}), flush=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
