"""TENSION controls on a uniform axisymmetric bar (NUCLEATE's ``nucleate/bar.py`` with the tension-only surface,
Gent, and compression): axis held radially, bottom on rollers, top moved axially and free radially (``--confined``:
the outer face held radially too, uniaxial strain). Homogeneous, so the peak nominal (Biot) stress in tension is the
model's uniaxial tensile strength.

python bar.py OUT.json --criterion tension|dp [--compress] [--confined] [--draws N --seed S] [--plant F] [--l MM ...]
A tension draw passes when the peak nominal stress is within ``--tol`` of the declared s_ts. ``--plant F``: the
coefficients are computed for F s_ts while s_ts is declared. Compression: reports whether damage ever starts (d > 1e-3)
down to ``--lam-min`` and at which nominal stress."""
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


def bar(radius: float, height: float, nx: int, ny: int, confined: bool, sense: float) -> Specimen:
    g = Grid(np.linspace(0.0, radius, nx + 1), np.linspace(0.0, height, ny + 1))
    free = np.ones((nx + 1, ny + 1, 2))
    free[0, :, 0] = 0.0
    if confined:
        free[-1, :, 0] = 0.0
    free[:, 0, 1] = 0.0
    free[:, -1, 1] = 0.0
    top = (np.arange(nx + 1), np.full(nx + 1, ny))
    return Specimen(grid=g, free=free, thickness=1.0, force_factor=1.0, load=Load(top, sense=sense, predictor="proportional"),
                    reaction=Reaction(top, sign=1.0), kinematics="axisymmetric", energy_factor=1.0)


def coefficients(criterion: str, base: Material, sts: float, ratio: float, h: float, gate: str = "i1"):
    if criterion == "tension":
        return S.tension_coefficients(base, sts, h, gate=gate)
    return S.coefficients(base, sts, ratio * sts, h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--criterion", choices=("tension", "dp"), default="tension")
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plant", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--mu", type=float, default=3660.0)
    ap.add_argument("--nu", type=float, default=0.49)
    ap.add_argument("--gc", type=float, default=27.6)
    ap.add_argument("--jm", type=float, default=3.7, help="Gent's Jm (0: neo-Hookean)")
    ap.add_argument("--l", type=float, nargs="*", default=None, help="mm; fixed values instead of random draws")
    ap.add_argument("--sts", type=float, default=None, help="kPa; fixed instead of random")
    ap.add_argument("--ratio", type=float, default=1.5, help="s_hs / s_ts (dp)")
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--gate", choices=("i1", "s1"), default="i1")
    ap.add_argument("--confined", action="store_true")
    ap.add_argument("--lam-min", type=float, default=0.35)
    ap.add_argument("--viscosity", type=float, default=0.0,
                    help="the damage viscosity, in units of Gc / l (the needle driver's is 0.1; with no elastic driving, as in "
                         "compression, zero leaves the damage system without a zero-order term)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    radius, height, nx, ny = 1e-3, 2e-3, 4, 8
    h = radius / nx
    young = 2 * a.mu * (1 + a.nu)
    mu, kappa = shear_and_bulk(young, a.nu)
    sense = -1.0 if a.compress else 1.0
    def make_solver(l):
        return Solver(bar(radius, height, nx, ny, a.confined, sense), "phase_field", tol_d=1e-7, max_staggered=200,
                      newton_regularise=True, volumetric="gauss", viscosity=a.viscosity * a.gc / l)
    solver = make_solver(1e-4) if a.viscosity == 0.0 else None
    area = np.pi * radius ** 2
    rows = []
    t0 = time.perf_counter()
    ls = a.l if a.l else [None] * a.draws
    for lmm in ls:
        sts = (a.sts if a.sts is not None else float(np.exp(rng.uniform(np.log(2.0), np.log(10.0)))) * a.mu / 1e3) * 1e3
        jm = a.jm if a.draws == 1 or a.l else float(rng.uniform(3.0, 6.0))
        base = Material(mu=mu, kappa=kappa, gc=a.gc, l=1e-4, k_res=1e-6, stretch_limit=10.0,
                        gent_jm=(jm if jm > 0 else None))
        lam_ts, _, w_ts = S.uniaxial_state(base._replace(k_res=0.0), sts)
        lmax = 3 * a.gc / (16 * w_ts)
        l = (lmm * 1e-3 if lmm is not None else float(rng.uniform(0.03, 0.15)) * lmax)
        base = base._replace(l=l)
        st = coefficients(a.criterion, base, a.plant * sts, a.ratio, h, a.gate)
        mat = base._replace(strength=st)
        if a.compress:
            lams = np.linspace(1.0, a.lam_min, 131)[1:]
        else:
            top = lam_ts * (1.0 + 0.6 * (a.plant - 1.0))
            coarse = np.linspace(1.0, 1.0 + 0.85 * (top - 1.0), 25)[1:]
            fine = np.linspace(1.0 + 0.85 * (top - 1.0), 1.0 + 1.25 * (top - 1.0) + 0.02, 81)[1:]
            if a.plant > 1.0:
                lam_pl = S.uniaxial_state(base._replace(k_res=0.0), a.plant * sts)[0]
                fine = np.concatenate([fine, np.linspace(fine[-1], 1.1 * lam_pl, 60)[1:]])
            lams = np.concatenate([coarse, fine])
        solver = solver if a.viscosity == 0.0 else make_solver(l)
        state = solver.initial()
        res = []
        for lam in lams:
            state, o = solver.step(state, jnp.asarray(abs(lam - 1.0) * height), mat)
            res.append((float(o.force) / area, float(jnp.max(state.d)), int(o.iterations), int(o.newton_capped)))
            if not np.isfinite(res[-1][0]):
                break
        res = np.asarray(res)
        lams = lams[:len(res)]
        onset_i = next((i for i in range(len(res)) if res[i, 1] > 1e-3), None)
        row = dict(criterion=a.criterion, compress=a.compress, confined=a.confined, jm=jm, l_mm=l * 1e3,
                   l_over_lmax=l / lmax, h_mm=h * 1e3, sts_kPa=sts / 1e3, sts_over_G=sts / a.mu, plant=a.plant,
                   delta=st.delta, lam_ts=lam_ts, capped=int(res[:, 3].sum()),
                   onset_lam=None if onset_i is None else float(lams[onset_i]),
                   onset_stress_kPa=None if onset_i is None else float(res[onset_i, 0]) / 1e3,
                   lam_end=float(lams[-1]), stress_end_kPa=float(res[-1, 0]) / 1e3, d_max_end=float(res[-1, 1]),
                   wall_s=time.perf_counter() - t0)
        if not a.compress:
            kpk = int(np.argmax(res[:, 0]))
            peak = float(res[kpk, 0])
            row.update(peak_kPa=peak / 1e3, peak_over_sts=peak / sts, lam_peak=float(lams[kpk]),
                       passed=bool(abs(peak / sts - 1.0) <= a.tol))
        else:
            row.update(nucleated=onset_i is not None)
        rows.append(row)
        print(json.dumps({q: (round(v, 5) if isinstance(v, float) else v) for q, v in row.items()}), flush=True)
    with open(a.out, "w") as fh:
        json.dump(dict(args=vars(a), rows=rows), fh, indent=1)


if __name__ == "__main__":
    main()
