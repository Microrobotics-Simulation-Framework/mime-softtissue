"""The material law and the element against closed forms, alone: no solver, no MIME."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime_softtissue.engine import material as law
from mime_softtissue.engine.kinematics import plane_strain as ps
from mime_softtissue.engine.material import Fibre, Material, shear_and_bulk
from mime_softtissue.engine.solvers import blocktri

E, NU = 5.7e3, 0.3
MU, KAPPA = shear_and_bulk(E, NU)


def material(fibre=None, k_res=0.0) -> Material:
    return Material(mu=jnp.asarray(MU), kappa=jnp.asarray(KAPPA), gc=jnp.asarray(7.6), l=jnp.asarray(3e-4),
                    k_res=jnp.asarray(k_res), stretch_limit=jnp.asarray(2.8), fibre=fibre)


def total(F, mat):
    plus, minus = ps.energy_parts(F, mat)
    return plus + minus


@pytest.mark.parametrize("gamma", [0.01, 0.3, 1.5])
def test_simple_shear_energy_is_mu_gamma_squared_over_two(gamma):
    F = jnp.array([[1.0, gamma], [0.0, 1.0]])
    assert float(total(F, material())) == pytest.approx(MU * gamma ** 2 / 2, rel=1e-13)


@pytest.mark.parametrize("lam", [0.7, 1.1, 2.0])
def test_equibiaxial_stretch_energy(lam):
    F = lam * jnp.eye(2)
    J = lam ** 2
    expected = 0.5 * MU * (J ** (-2 / 3) * (2 * lam ** 2 + 1) - 3) + 0.25 * KAPPA * (J ** 2 - 1 - 2 * np.log(J))
    assert float(total(F, material())) == pytest.approx(expected, rel=1e-13)


def test_small_strain_stress_is_linear_elasticity_in_plane_strain():
    """P = 2 mu eps + lambda tr(eps) I, lambda = kappa - 2 mu / 3 (3-D moduli; the out-of-plane strain is zero)."""
    eps = 1e-6 * np.array([[1.0, 0.4], [0.4, -0.3]])
    P = np.asarray(jax.grad(total)(jnp.eye(2) + eps, material()))
    lam = KAPPA - 2 * MU / 3
    expected = 2 * MU * eps + lam * np.trace(eps) * np.eye(2)
    assert np.max(np.abs(P - expected)) <= 1e-5 * np.max(np.abs(expected))


def test_element_patch_test_constant_gradient_is_reproduced_exactly():
    """Q1 elements carry a constant deformation gradient exactly: u = (F - I) X gives psi(F) times the area."""
    Fg = np.array([[1.3, 0.2], [-0.1, 0.9]])
    dx, dy = 2e-3, 3e-3
    corners = np.array([[0, 0], [dx, 0], [dx, dy], [0, dy]])
    ue = jnp.asarray(corners @ (Fg - np.eye(2)).T)
    mat = material()
    got = ps.element_strain_energy(ue, jnp.zeros(4), dx, dy, mat)
    assert float(got) == pytest.approx(float(total(jnp.asarray(Fg), mat)) * dx * dy, rel=1e-13)


def test_fibre_energy_of_two_families_under_uniaxial_stretch():
    lam = 1.4
    fibre = Fibre(k1=jnp.asarray(2e3), k2=jnp.asarray(1.5), angle=jnp.asarray(0.0))
    F = jnp.diag(jnp.array([lam, 1 / lam]))
    with_fibre = float(total(F, material(fibre)))
    without = float(total(F, material()))
    expected = 2 * 2e3 / (2 * 1.5) * (np.exp(1.5 * (lam ** 2 - 1) ** 2) - 1)
    assert with_fibre - without == pytest.approx(expected, rel=1e-12)


def test_fibres_take_no_compression():
    fibre = Fibre(k1=jnp.asarray(2e3), k2=jnp.asarray(1.5), angle=jnp.asarray(0.0))
    F = jnp.diag(jnp.array([0.8, 1.0]))
    assert float(total(F, material(fibre))) == float(total(F, material()))


def test_energy_is_infinite_when_the_element_is_inverted():
    F = jnp.array([[-1.0, 0.0], [0.0, 1.0]])
    assert np.isinf(float(total(F, material())))


def test_damage_degrades_the_stored_energy_but_not_compression():
    mat = material(k_res=1e-6)
    stretch, squeeze = jnp.diag(jnp.array([1.2, 1.0])), jnp.diag(jnp.array([0.8, 0.8]))
    for F in (stretch, squeeze):
        plus, minus = ps.energy_parts(F, mat)
        for d in (0.0, 0.5, 1.0):
            g = float(law.degradation(d, mat.k_res))
            assert g == pytest.approx((1 - d) ** 2 + 1e-6)
    plus, minus = ps.energy_parts(squeeze, mat)
    assert float(minus) > 0 and float(plus) == pytest.approx(float(0.5 * MU * ((0.64) ** (-2 / 3) * (1.28 + 1) - 3)), rel=1e-12)


def _dense(diag, upper):
    n, m = diag.shape[0], diag.shape[1]
    A = np.zeros((n * m, n * m))
    for i in range(n):
        A[i * m:(i + 1) * m, i * m:(i + 1) * m] = diag[i]
    for i in range(n - 1):
        A[i * m:(i + 1) * m, (i + 1) * m:(i + 2) * m] = upper[i]
        A[(i + 1) * m:(i + 2) * m, i * m:(i + 1) * m] = upper[i].T
    return A


def test_block_tridiagonal_solve_against_a_dense_solve():
    rng = np.random.default_rng(7)
    n, m = 6, 4
    upper = rng.normal(size=(n - 1, m, m)) * 0.3
    diag = np.stack([(lambda M: M @ M.T + 3 * np.eye(m))(rng.normal(size=(m, m))) for _ in range(n)])
    rhs = rng.normal(size=(n, m))
    x = np.asarray(blocktri.solve(jnp.asarray(diag), jnp.asarray(upper), jnp.asarray(rhs)))
    A = _dense(diag, upper)
    assert np.max(np.abs(A @ x.ravel() - rhs.ravel())) <= 1e-12 * np.max(np.abs(rhs))
    assert np.allclose(np.asarray(blocktri.matvec(jnp.asarray(diag), jnp.asarray(upper), jnp.asarray(x))), rhs, atol=1e-11)
