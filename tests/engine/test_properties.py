"""Property tests of the engine (Hypothesis). Each property is a function ``check_*`` taking its draws from strategies
of what a solid can have; ``test_*`` runs it, and ``tests/test_plants.py`` runs it again with a planted error and states
the detection rate. Tolerances have no absolute part: each is the law's rounding on the scale of the inputs it came
from."""
import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings, strategies as st

import strategies as sx
from mime_softtissue.engine import material as law
from mime_softtissue.engine.fracture import phase_field
from mime_softtissue.engine.grid import Grid
from mime_softtissue.engine.assembly import Assembly
from mime_softtissue.engine.kinematics import plane_strain as ps

EPS = float(np.finfo(sx.DTYPE).eps)
settings.register_profile("engine", deadline=None)
settings.load_profile("engine")


def energy_scale(F, mat) -> float:
    """The size of the terms psi sums, so cancellation is judged against what was added."""
    J = float(jnp.linalg.det(F))
    C = F.T @ F
    I1 = float(C[0, 0] + C[1, 1] + 1.0)
    return float(0.5 * mat.mu * J ** (-2 / 3) * I1 + 0.25 * mat.kappa * (J ** 2 + 1.0 + abs(2 * np.log(J))))


def total_energy(F, mat):
    plus, minus = ps.energy_parts(F, mat)
    return plus + minus


def rotation_defect(F, mat, angle) -> tuple[float, float]:
    """(|psi(QF) - psi(F)|, the scale it is judged on): every term of psi is nonnegative, so |psi| is the sum of their
    magnitudes and the scale is that sum plus the neo-Hookean parts'."""
    F = jnp.asarray(F)
    Q = jnp.asarray(sx.rotation(angle))
    a, b = total_energy(F, mat), total_energy(Q @ F, mat)
    return abs(float(a - b)), energy_scale(F, mat) + abs(float(a))


@settings(max_examples=100)
@given(sx.deformation_gradients(), sx.materials(), st.floats(-np.pi, np.pi, allow_nan=False))
def check_rotation_invariance(F, mat, angle):
    """psi(Q F) = psi(F) for a rotation Q (objectivity), over the stretches a gel reaches (0.6 to 2.8)."""
    err, scale = rotation_defect(F, mat, angle)
    assert err <= 256 * EPS * scale, (err, scale)


@settings(max_examples=100)
@given(sx.deformation_gradients(stretch=(0.85, 1.25)), sx.materials(fibre=True), st.floats(-np.pi, np.pi, allow_nan=False))
def check_rotation_invariance_with_fibres(F, mat, angle):
    """The same with fibre families, over the stretches a fibre network carries before it breaks (0.85 to 1.25): the
    exponential of the fibre law amplifies the rounding of I4 by 4 k2 (I4 - 1)^2, so the tolerance is judged on that
    amplification; drawing 2.8 here would draw a fibre energy of 1e38, which no body has."""
    err, scale = rotation_defect(F, mat, angle)
    I4 = max(float(jnp.max(jnp.abs(jnp.asarray(F).T @ jnp.asarray(F)))) - 1.0, 0.0)
    assert err <= 256 * EPS * scale * (1.0 + 4.0 * float(mat.fibre.k2) * I4 ** 2), (err, scale)


@settings(max_examples=100)
@given(sx.materials(fibre=True))
def check_zero_and_stationary_at_identity(mat):
    """psi(I) = 0 and the first Piola stress vanishes there."""
    I = jnp.eye(2, dtype=sx.DTYPE)
    scale = float(mat.mu + mat.kappa)
    plus, minus = ps.energy_parts(I, mat)
    assert abs(float(plus + minus)) <= 64 * EPS * scale
    P = jax.grad(total_energy)(I, mat)
    assert float(jnp.max(jnp.abs(P))) <= 64 * EPS * scale, np.asarray(P)


@settings(max_examples=60)
@given(sx.materials(), st.lists(st.floats(-0.3, 0.3, allow_nan=False), min_size=8, max_size=8),
       st.floats(1e-4, 1e-3), st.floats(1e-4, 1e-3), st.lists(st.floats(0.0, 0.9), min_size=4, max_size=4))
def check_hessian_symmetric(mat, u, dx, dy, d):
    """The element energy's Hessian in the nodal displacements is symmetric, damaged or not."""
    ue = jnp.asarray(np.asarray(u, dtype=sx.DTYPE).reshape(4, 2) * dx)
    dq = jnp.asarray(np.asarray(d, dtype=sx.DTYPE))
    H = np.asarray(ps.element_hessian(ue, dq, dx, dy, mat))
    assert np.all(np.isfinite(H))
    assert np.max(np.abs(H - H.T)) <= 1e3 * EPS * np.max(np.abs(H)), np.max(np.abs(H - H.T)) / np.max(np.abs(H))


_GRID = Grid(np.linspace(0.0, 4.0e-3, 5), np.linspace(0.0, 3.0e-3, 4))
_ASSEMBLY = Assembly(_GRID, 1)
_I, _J = (jnp.asarray(a) for a in _GRID.element_nodes())
_DX, _DY = (jnp.asarray(a) for a in _GRID.element_sizes())


_VEC = Assembly(_GRID, 2)


@settings(max_examples=40)
@given(st.lists(st.floats(-0.2, 0.2, allow_nan=False), min_size=(5 * 4 * 2), max_size=(5 * 4 * 2)),
       st.lists(st.floats(-1.0, 1.0, allow_nan=False), min_size=(5 * 4 * 2), max_size=(5 * 4 * 2)),
       st.lists(st.floats(0.0, 0.9), min_size=_GRID.n_elements * 4, max_size=_GRID.n_elements * 4), sx.materials())
def check_assembled_stiffness_matches_the_gradient(u, v, damage, mat):
    """The assembled block-tridiagonal tangent stiffness times v equals the directional derivative of the gradient of
    the total energy along v (assembly and element Hessian agree with the energy), on a 4 x 3 grid."""
    from mime_softtissue.engine.solvers import blocktri
    nx, ny = _GRID.nx + 1, _GRID.ny + 1
    scale = float(_DX.min())
    U = jnp.asarray(np.asarray(u, dtype=sx.DTYPE).reshape(nx, ny, 2) * scale)
    V = jnp.asarray(np.asarray(v, dtype=sx.DTYPE).reshape(nx, ny, 2))
    dq = jnp.asarray(np.asarray(damage, dtype=sx.DTYPE).reshape(_GRID.n_elements, 4))

    def energy(w):
        e = jax.vmap(lambda ue, d, dx, dy: ps.element_strain_energy(ue, d, dx, dy, mat))(w[_I, _J], dq, _DX, _DY)
        return jnp.sum(e)

    ke = jax.vmap(lambda ue, d, dx, dy: ps.element_hessian(ue, d, dx, dy, mat))(U[_I, _J], dq, _DX, _DY)
    diag, upper = _VEC.blocks(ke)
    got = blocktri.matvec(diag, upper, V.reshape(nx, -1))
    want = jax.jvp(jax.grad(energy), (U,), (V,))[1].reshape(nx, -1)
    assert float(jnp.max(jnp.abs(got - want))) <= 1e-9 * float(jnp.max(jnp.abs(want)) + 1e-300), \
        float(jnp.max(jnp.abs(got - want)) / jnp.max(jnp.abs(want)))


def test_assembled_stiffness_matches_the_gradient():
    check_assembled_stiffness_matches_the_gradient()


@settings(max_examples=100)
@given(st.lists(st.floats(0.0, 1e7, allow_nan=False), min_size=_GRID.n_elements * 4, max_size=_GRID.n_elements * 4),
       st.integers(0, _GRID.n_elements - 1), st.floats(1e-4, 1e-3), st.floats(1.0, 100.0))
def check_damage_in_unit_interval(history, spike, length, gc):
    """The damage for any nonnegative history, a single hot element included, lies in [0, 1]."""
    H = np.asarray(history, dtype=sx.DTYPE).reshape(_GRID.n_elements, 4)
    H[spike] = 1e9  # a hot element beside cold ones: the consistent mass matrix overshoots there
    d = phase_field.solve_damage(jnp.asarray(H), gc, length, _DX, _DY, _GRID, _ASSEMBLY, _I, _J)
    d = np.asarray(d)
    assert np.all(np.isfinite(d)) and d.min() >= 0.0 and d.max() <= 1.0, (d.min(), d.max())


@settings(max_examples=100)
@given(st.lists(st.floats(0.0, 1e6, allow_nan=False), min_size=16, max_size=16),
       st.lists(st.floats(0.0, 1e6, allow_nan=False), min_size=16, max_size=16))
def check_history_function_never_decreases(previous, plus):
    """The history after any state is at least the history before."""
    a, b = jnp.asarray(previous, dtype=sx.DTYPE), jnp.asarray(plus, dtype=sx.DTYPE)
    assert bool(jnp.all(phase_field.updated_history(a, b) >= a))


def test_rotation_invariance():
    check_rotation_invariance()


def test_rotation_invariance_with_fibres():
    check_rotation_invariance_with_fibres()


def test_zero_and_stationary_at_identity():
    check_zero_and_stationary_at_identity()


def test_hessian_symmetric():
    check_hessian_symmetric()


def test_damage_in_unit_interval():
    check_damage_in_unit_interval()


def test_history_function_never_decreases():
    check_history_function_never_decreases()
