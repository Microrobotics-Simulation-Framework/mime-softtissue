"""Incompressible plane stress and the Yeoh law, against closed forms; and the pure-shear sheet's energy account.

The lines, stated before the checks: the energy density's derivative equals the closed-form nominal stress of pure shear
(lambda_2 = 1) and of uniaxial tension (the lateral stretch minimised to lambda^-1/2) to 1e-12; the FE sheet with its
width held carries the closed-form pure-shear force to 1e-5 (Newton's step tolerance); the phase-field run's
|W - U - Gamma| stays within 2 % of W up to the peak. Each has a planted error that must fail it."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime_softtissue.engine.kinematics import plane_strain as ps
from mime_softtissue.engine.loadpath import run
from mime_softtissue.engine.material import Material
from mime_softtissue.engine.solver import Solver
from mime_softtissue.engine.specimen import edge_notched_strip

C1, C2, C3 = 400.0, 125.0, 7.0
WIDTH, HALF, T = 40e-3, 5e-3, 3e-3


def material(c2=C2, c3=C3, gc=1.0, l=1e-3) -> Material:
    return Material(mu=jnp.asarray(2 * C1), kappa=jnp.asarray(0.0), gc=jnp.asarray(gc), l=jnp.asarray(l),
                    k_res=jnp.asarray(1e-6), stretch_limit=jnp.asarray(10.0), c2=jnp.asarray(c2), c3=jnp.asarray(c3))


def yeoh_slope(x):
    return C1 + 2 * C2 * x + 3 * C3 * x ** 2


def density(F, mat, stress=True):
    plus, minus = ps.energy_parts(F, mat, stress)
    return plus + minus


def pure_shear_stress(lam, mat, stress=True):
    return float(jax.grad(density)(jnp.diag(jnp.array([1.0, lam])), mat, stress)[1, 1])


def uniaxial_stress(lam, mat):
    def w(a):
        return density(jnp.diag(jnp.stack([a, jnp.asarray(lam)])), mat)
    a = jnp.asarray(1.0)
    for _ in range(40):
        a = a - jax.grad(w)(a) / jax.hessian(w)(a)
    return float(a), float(jax.grad(density)(jnp.diag(jnp.stack([a, jnp.asarray(lam)])), mat)[1, 1])


@pytest.mark.parametrize("lam", [1.05, 1.3, 1.6, 2.2])
def test_pure_shear_stress_is_the_closed_form(lam):
    x = lam ** 2 + lam ** -2 - 2
    assert pure_shear_stress(lam, material()) == pytest.approx(2 * (lam - lam ** -3) * yeoh_slope(x), rel=1e-12)


@pytest.mark.parametrize("lam", [1.05, 1.3, 1.6, 2.2])
def test_uniaxial_stress_is_the_closed_form(lam):
    lateral, P = uniaxial_stress(lam, material())
    x = lam ** 2 + 2 / lam - 3
    assert lateral == pytest.approx(lam ** -0.5, rel=1e-12)
    assert P == pytest.approx(2 * (lam - lam ** -2) * yeoh_slope(x), rel=1e-12)


def test_plane_strain_in_place_of_plane_stress_fails_the_closed_form():
    """The plant: the thickness held (F33 = 1, plane strain with no bulk term) must miss the closed form by far more
    than the line."""
    lam = 1.3
    x = lam ** 2 + lam ** -2 - 2
    planted = pure_shear_stress(lam, material(), stress=False)
    assert abs(planted / (2 * (lam - lam ** -3) * yeoh_slope(x)) - 1) > 1e-2


def test_the_stiffening_is_smooth_at_the_undeformed_state():
    """Every derivative of the Yeoh law exists at the identity: the stress is zero and the tangent is 2 mu's
    small-strain incompressible plane-stress modulus there (E = 6 C1 = 3 mu, P11 = 4 C1 (2 e11 + e22) for a plane
    sheet)."""
    eps = 1e-7 * np.array([[1.0, 0.0], [0.0, -0.4]])
    P = np.asarray(jax.grad(density)(jnp.eye(2) + eps, material()))
    expected = 4 * C1 * np.array([2 * eps[0, 0] + eps[1, 1], eps[0, 0] + 2 * eps[1, 1]])
    assert np.allclose(np.diag(P), expected, rtol=1e-5, atol=0)
    assert np.isfinite(np.asarray(jax.hessian(density)(jnp.eye(2), material()))).all()


def held_width_sheet():
    sp = edge_notched_strip(width=WIDTH, half_height=HALF, thickness=T, notch=0.0, h=1e-3, fine=1e-3,
                            kinematics="plane_stress")
    free = sp.free.copy()
    free[:, :, 0] = 0.0
    return replace(sp, free=free)


def test_the_sheet_with_its_width_held_carries_the_closed_form_force():
    deltas = np.array([1e-3, 3e-3, 6e-3])
    r = run(Solver(held_width_sheet(), "elastic"), material(), deltas)
    lam = 1 + deltas / (2 * HALF)
    x = lam ** 2 + lam ** -2 - 2
    closed = 2 * (lam - lam ** -3) * yeoh_slope(x) * WIDTH * T
    assert np.max(np.abs(r["force"] / closed - 1)) <= 1e-5


def balance(r):
    d = np.concatenate([[0.0], r["delta"]])
    f = np.concatenate([[0.0], r["force"]])
    work = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(d))])[1:]
    return work, work - r["energy"] - r["crack_energy"]


@pytest.fixture(scope="module")
def notched_run():
    """A coarse notched sheet (h 0.5 mm, l 1 mm, notch 12 mm) to just past its peak."""
    sp = edge_notched_strip(width=WIDTH, half_height=HALF, thickness=T, notch=12e-3, h=0.5e-3, fine=3e-3,
                            kinematics="plane_stress")
    s = Solver(sp, "phase_field", tol_d=1e-5, max_staggered=300, newton_regularise=True)
    return run(s, material(c3=0.0, gc=2.0), np.linspace(0, 8e-3, 81)[1:], stop_below=0.6, after_peak=2)


def test_energy_balance_holds_to_the_line_up_to_the_peak(notched_run):
    work, resid = balance(notched_run)
    peak = int(np.argmax(notched_run["force"]))
    assert peak > 10
    assert np.max(np.abs(resid[:peak + 1]) / work[:peak + 1]) <= 0.02


def test_energy_balance_resolves_a_planted_error(notched_run):
    """A 3 % error planted in the stored energy's book must be seen (the control can fail)."""
    r = dict(notched_run)
    r["energy"] = notched_run["energy"] * 1.03
    work, resid = balance(r)
    peak = int(np.argmax(r["force"]))
    assert np.max(np.abs(resid[:peak + 1]) / work[:peak + 1]) > 0.02
