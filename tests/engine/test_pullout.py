"""The friction law against a closed form: an elastic tube on a rigid rod with an interference fit, the rod slid along
its axis (``specimen.tube_on_rod``). Steady force mu p 2 pi a L, p from Lame's thick cylinder in plane strain with its
outer face free, the interference less the penalty's penetration p/k [derived]. Line 1 % (round 4's prediction C1.3).

The placeholder adhesion law (tau_a on the nodes in contact) is not tested here: against its closed form (tau_a 2 pi a
L, mu 0) it misses by 5 % with the lag relaxed and does not converge with it at 1.0, and with mu 0.3 Newton caps
(round 4, ``runs/pull``): it does not fit, so it is reported open, not asserted.

The planted errors it must catch: the contact's node weights without their 2 pi R; the lagged normal along the axis
instead of the surface's (the slip then has no tangential part along the axis, so no friction acts)."""
from __future__ import annotations

import contextlib

import jax.numpy as jnp
import numpy as np
import pytest

from mime_softtissue.engine import contact as ct
from mime_softtissue.engine.material import Material, shear_and_bulk
from mime_softtissue.engine.solver import Solver
from mime_softtissue.engine.specimen import tube_on_rod

E, NU = 50e3, 0.4
A, B, LEN, H, MU = 0.5e-3, 2.5e-3, 1.0e-3, 0.05e-3, 0.3


def lame_pressure(delta: float, k: float) -> float:
    p = E * delta * (B ** 2 - A ** 2) / (A * (1 + NU) * ((1 - 2 * NU) * A ** 2 + B ** 2))
    return p / (1 + p / (k * delta))


def pulled(*, friction=MU, adhesion=0.0, weight_plant=False, max_newton=40) -> tuple[float, float]:
    """The steady force after four slides of 0.02 mm, and the closed form's."""
    k, delta = 100 * E / H, 1e-3 * A
    sp = tube_on_rod(inner=A, outer=B, length=LEN, h=H, interference=delta, stiffness=k, friction=friction,
                     slip=1e-2 * H, adhesion=adhesion, weight_plant=weight_plant)
    mu, kappa = shear_and_bulk(E, NU)
    mat = Material(mu=jnp.asarray(mu), kappa=jnp.asarray(kappa), gc=jnp.asarray(1.0), l=jnp.asarray(1e-4),
                   k_res=jnp.asarray(1e-6), stretch_limit=jnp.asarray(10.0))
    solver = Solver(sp, "elastic", max_newton=max_newton)
    state = solver.initial()
    for slide in (0.0, 0.02e-3, 0.04e-3, 0.06e-3, 0.08e-3):
        state, out = solver.step(state, jnp.asarray(5e-3 + slide), mat)
    return float(out.force), (friction * lame_pressure(delta, k) + adhesion) * 2 * np.pi * A * LEN


@contextlib.contextmanager
def normal_along_axis():
    real = ct.pressure_and_normal

    def planted(c, X, u, depth):
        p, n = real(c, X, u, depth)
        return p, jnp.zeros_like(n).at[..., 1].set(1.0)

    ct.pressure_and_normal = planted
    try:
        yield
    finally:
        ct.pressure_and_normal = real


def test_the_pull_out_force_is_lames_pressure_times_friction():
    force, exact = pulled()
    assert abs(force / exact - 1) < 1e-2, (force, exact)


@pytest.mark.parametrize("plant", ["no_2_pi_r", "normal_along_axis"])
def test_the_planted_errors_fail_the_line(plant):
    if plant == "no_2_pi_r":
        # Newton diverges under this plant (measured: the force is 5e4 to 3e5 times the closed form, with the cap at 40, 10
        # or 5 iterations), and a diverged step runs to its cap: 40 iterations took 60 s here against 9 s for 10. The honest
        # arm above keeps 40 and converges (ratio 1.002 at 40, 10 and 5).
        force, exact = pulled(weight_plant=True, max_newton=10)
    else:
        with normal_along_axis():
            force, exact = pulled()
    assert abs(force / exact - 1) > 1e-2, (plant, force, exact)
