"""NUCLEATE control: a uniform axisymmetric bar in uniaxial tension (axis held radially, bottom on rollers, top pulled,
its radial motion free) with the strength-surface phase field (``engine.fracture.strength``) or AT2. The bar is
homogeneous, so its peak nominal stress is the model's uniaxial tensile strength: the declared s_ts for the strength
surface at any l; AT2's (3 sqrt3 / 16) sqrt(E Gc / l) order of magnitude, which moves with l.

python bar.py OUT.json --draws N [--seed S] [--plant 2.0] [--at2]   (each draw: random l, h, s_ts, s_hs/s_ts)
A draw passes when the peak nominal stress is within ``--tol`` of the declared s_ts. ``--plant F``: the coefficients
are computed for F s_ts while s_ts is declared (a planted error the check must catch)."""
from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mime_softtissue.engine.fracture import strength as S  # noqa: E402
from mime_softtissue.engine.grid import Grid  # noqa: E402
from mime_softtissue.engine.material import Material, shear_and_bulk  # noqa: E402
from mime_softtissue.engine.solver import Solver  # noqa: E402
from mime_softtissue.engine.specimen import Load, Reaction, Specimen  # noqa: E402


def bar(radius: float, height: float, nx: int, ny: int) -> Specimen:
    g = Grid(np.linspace(0.0, radius, nx + 1), np.linspace(0.0, height, ny + 1))
    free = np.ones((nx + 1, ny + 1, 2))
    free[0, :, 0] = 0.0  # the axis
    free[:, 0, 1] = 0.0  # the bottom: rollers
    free[:, -1, 1] = 0.0  # the top: pulled axially, free radially
    top = (np.arange(nx + 1), np.full(nx + 1, ny))
    return Specimen(grid=g, free=free, thickness=1.0, force_factor=1.0, load=Load(top, sense=1.0, predictor="proportional"),
                    reaction=Reaction(top, sign=1.0), kinematics="axisymmetric", energy_factor=1.0)


def run(solver: Solver, mat: Material, height: float, stretches: np.ndarray, area: float) -> np.ndarray:
    state = solver.initial()
    out = []
    for lam in stretches:
        state, o = solver.step(state, jnp.asarray((lam - 1.0) * height), mat)
        out.append((float(o.force) / area, float(jnp.max(state.d)), int(o.iterations), int(o.newton_capped)))
    return np.asarray(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plant", type=float, default=1.0)
    ap.add_argument("--at2", action="store_true")
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--mu", type=float, default=3660.0)
    ap.add_argument("--nu", type=float, default=0.45)
    ap.add_argument("--gc", type=float, default=27.0)
    ap.add_argument("--l", type=float, nargs="*", default=None, help="mm; fixed values instead of random draws")
    ap.add_argument("--sts", type=float, default=None, help="kPa; fixed instead of random")
    ap.add_argument("--ratio", type=float, default=None, help="s_hs / s_ts; fixed instead of random")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    radius, height, nx, ny = 1e-3, 2e-3, 4, 8
    h = radius / nx
    young = 2 * a.mu * (1 + a.nu)
    mu, kappa = shear_and_bulk(young, a.nu)
    solver = Solver(bar(radius, height, nx, ny), "phase_field", tol_d=1e-7, max_staggered=200, newton_regularise=True)
    area = np.pi * radius ** 2
    rows = []
    t0 = time.perf_counter()
    ls = a.l if a.l else [None] * a.draws
    for k, lmm in enumerate(ls):
        l = (lmm if lmm is not None else float(rng.uniform(0.1, 1.0))) * 1e-3
        sts = (a.sts if a.sts is not None else float(np.exp(rng.uniform(np.log(4.0), np.log(60.0))))) * 1e3
        ratio = a.ratio if a.ratio is not None else float(rng.uniform(0.6, 3.0))
        base = Material(mu=mu, kappa=kappa, gc=a.gc, l=l, k_res=1e-6, stretch_limit=10.0)
        if a.at2:
            mat = base
            lam_ts = 1.0
        else:
            st = S.coefficients(base, a.plant * sts, a.plant * ratio * sts, h)
            mat = base._replace(strength=st)
            lam_ts = S.uniaxial_state(base, sts)[0]
        top = max(lam_ts, 1.0) * (1.0 + 0.6 * (a.plant - 1.0)) if not a.at2 else 4.0
        coarse = np.linspace(1.0, 1.0 + 0.85 * (top - 1.0), 25)[1:]
        fine = np.linspace(1.0 + 0.85 * (top - 1.0), 1.0 + 1.25 * (top - 1.0) + 0.02, 81)[1:]
        if a.plant > 1.0 and not a.at2:  # reach the planted surface too
            lam_pl = S.uniaxial_state(base, a.plant * sts)[0]
            fine = np.concatenate([fine, np.linspace(fine[-1], 1.15 * lam_pl, 60)[1:]])
        lams = np.concatenate([coarse, fine])
        res = run(solver, mat, height, lams, area)
        kpk = int(np.argmax(res[:, 0]))
        peak = float(res[kpk, 0])
        onset = next((float(lams[i]) for i in range(len(lams)) if res[i, 1] > 1e-3), None)
        row = dict(l_mm=l * 1e3, h_mm=h * 1e3, sts_kPa=sts / 1e3, ratio=ratio, plant=a.plant, at2=a.at2,
                   peak_kPa=peak / 1e3, peak_over_sts=peak / sts, lam_peak=float(lams[kpk]), lam_ts=lam_ts,
                   onset_lam=onset, capped=int(res[:, 3].sum()), passed=bool(abs(peak / sts - 1.0) <= a.tol),
                   delta=None if a.at2 else mat.strength.delta, wall_s=time.perf_counter() - t0)
        rows.append(row)
        print(json.dumps({q: (round(v, 5) if isinstance(v, float) else v) for q, v in row.items()}), flush=True)
    with open(a.out, "w") as fh:
        json.dump(dict(args=vars(a), rows=rows), fh, indent=1)


if __name__ == "__main__":
    main()
