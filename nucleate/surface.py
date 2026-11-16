"""NUCLEATE: how close the regularised model's uniform-state strength surface is to the Drucker-Prager surface it
encodes, along five load paths (uniaxial tension, equibiaxial tension, plane-strain tension, uniaxial compression,
hydrostatic tension), for given strengths, l and h. Exact at uniaxial and hydrostatic tension by construction; the
others measure the regularisation's distortion. python surface.py --sts KPA --ratio R --l MM --h MM"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mime_softtissue.engine.fracture import strength as S  # noqa: E402
from mime_softtissue.engine.material import Material, shear_and_bulk  # noqa: E402


BIOT = jax.jit(S.biot_principal)


def free_stretch(mat, fixed, idx_free):
    """The free stretches (indices ``idx_free``, set equal) making their Biot stresses zero."""
    zero = jnp.zeros(())

    def f(x):
        lam = np.array(fixed, float)
        lam[idx_free] = x
        return float(BIOT(jnp.asarray(lam), zero, mat)[idx_free[0]])
    return S._bisect(f, 0.02, 3.0, tol=1e-10)


def path(mat, kind, t):
    if kind == "uniaxial":
        return np.array([t] + [free_stretch(mat, [t, 1, 1], [1, 2])] * 2)
    if kind == "equibiaxial":
        return np.array([t, t, free_stretch(mat, [t, t, 1], [2])])
    if kind == "plane":
        return np.array([t, 1.0, free_stretch(mat, [t, 1, 1], [2])])
    if kind == "compression":
        c = 1.0 / t
        return np.array([c] + [free_stretch(mat, [c, 1, 1], [1, 2])] * 2)
    return np.array([t, t, t])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sts", type=float, nargs="+", default=[20.0])
    ap.add_argument("--ratio", type=float, default=1.5)
    ap.add_argument("--l", type=float, nargs="+", default=[0.5])
    ap.add_argument("--h", type=float, default=0.1)
    ap.add_argument("--mu", type=float, default=3660.0)
    ap.add_argument("--nu", type=float, default=0.45)
    ap.add_argument("--gc", type=float, default=27.0)
    a = ap.parse_args()
    mu, kappa = shear_and_bulk(2 * a.mu * (1 + a.nu), a.nu)
    zero = jnp.zeros(())
    for sts in a.sts:
        for lmm in a.l:
            base = Material(mu=mu, kappa=kappa, gc=a.gc, l=lmm * 1e-3, k_res=0.0, stretch_limit=10.0)
            st = S.coefficients(base, sts * 1e3, a.ratio * sts * 1e3, a.h * 1e-3)
            out = []
            for kind in ("uniaxial", "equibiaxial", "plane", "compression", "hydrostatic"):
                ts = np.exp(np.linspace(0.0, np.log(400.0), 1500))[1:]
                model = dp = None
                for t in ts:
                    lam = path(base, kind, t)
                    s = np.asarray(BIOT(jnp.asarray(lam), zero, base))
                    if dp is None and S.surface(s, sts * 1e3, a.ratio * sts * 1e3) >= 0:
                        dp = (t, np.abs(s).max())
                    if model is None and S.homogeneous_onset(base, st, lam) > 0:
                        model = (t, np.abs(s).max())
                    if model is not None and dp is not None:
                        break
                ratio = None if (model is None or dp is None) else model[1] / dp[1]
                out.append(f"{kind} {'none' if model is None else f'{model[1] / 1e3:.3g}'}/"
                           f"{'none' if dp is None else f'{dp[1] / 1e3:.3g}'} kPa"
                           f" ({'-' if ratio is None else f'{ratio:.3f}'})")
            l_mat = 3 * a.gc / (16 * S.uniaxial_state(base, sts * 1e3)[2])
            print(f"s_ts {sts} kPa, l {lmm} mm (l / material length {lmm * 1e-3 / l_mat:.3g}), delta {st.delta:.3f}, "
                  f"beta1 {st.beta1:.3g}, beta2 {st.beta2:.3g}: " + "; ".join(out), flush=True)


if __name__ == "__main__":
    main()
