"""The axisymmetric kinematics alone, and a cylinder in uniform stretch through the solver, against closed forms.

Each check has the planted errors it must catch (``AXI_PLANTS``): the hoop term dropped, the radial weight taken at the
element's centre instead of the Gauss point, the 2 pi r weight left out, the hoop stretch read from the axial
displacement. ``tests/test_axi_plants.py`` measures the rates; a check a plant cannot reach is listed as such."""
from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mime_softtissue.engine.kinematics import axisymmetric as axi
from mime_softtissue.engine.loadpath import run
from mime_softtissue.engine.material import Material, shear_and_bulk
from mime_softtissue.engine.solver import Solver
from mime_softtissue.engine.specimen import stretched_cylinder
from mime_softtissue.fit import homogeneous as hom

E, NU = 50e3, 0.4


def material(k_res=0.0):
    mu, kappa = shear_and_bulk(E, NU)
    one = jnp.asarray(1.0)
    return Material(mu=jnp.asarray(mu), kappa=jnp.asarray(kappa), gc=one, l=one, k_res=jnp.asarray(k_res),
                    stretch_limit=one)


# --- plants (monkeypatches of the module functions the energy looks up at trace time) ------------------------------
@contextlib.contextmanager
def _patched(name, fn):
    old = getattr(axi, name)
    setattr(axi, name, fn)
    try:
        yield
    finally:
        setattr(axi, name, old)


def hoop_dropped():
    return _patched("hoop_stretch", lambda ue, n, r: jnp.ones((), ue.dtype))


def weight_at_centre():
    return _patched("gauss_weights", lambda x0, dx, dy: jnp.full((4,), 2.0 * jnp.pi * (x0 + 0.5 * dx) * dx * dy / 4.0))


def no_ring_weight():
    return _patched("gauss_weights", lambda x0, dx, dy: jnp.full((4,), dx * dy / 4.0))


def hoop_from_axial():
    return _patched("hoop_stretch", lambda ue, n, r: 1.0 + (n @ ue[:, 1]) / r)


AXI_PLANTS = {"hoop dropped": hoop_dropped, "weight at centre": weight_at_centre, "no 2 pi r weight": no_ring_weight,
              "hoop from u_Z": hoop_from_axial}


# --- the cylinder in uniform stretch: the solver against the homogeneous closed form ------------------------------
def uniform_stretch_error(stretch: float, nr: int = 7, nz: int = 5, radius: float = 2e-3, height: float = 4e-3):
    """Largest relative departure of (force, lateral stretch, every node's u_R) from the closed form."""
    sp = stretched_cylinder(radius=radius, height=height, nr=nr, nz=nz)
    mat = material()
    s = Solver(sp, "elastic")
    deltas = np.linspace(0.0, (stretch - 1.0) * height, 6)[1:]
    r = run(s, mat, deltas)
    l1, _ = hom.lateral_stretches(jnp.asarray(stretch), mat)
    l1 = float(l1)
    P3 = float(jax.grad(lambda c: hom.psi(jnp.diag(jnp.stack([jnp.asarray(l1), jnp.asarray(l1), c])), mat))(
        jnp.asarray(stretch)))
    force = P3 * np.pi * radius ** 2
    u = r["final_u"]
    R = sp.grid.x[:, None]
    ur_err = np.max(np.abs(u[..., 0] - (l1 - 1.0) * R)) / (abs(l1 - 1.0) * radius)
    f_err = abs(r["force"][-1] - force) / abs(force)
    lam_err = abs((u[-1, -1, 0] / radius + 1.0) - l1) / abs(l1 - 1.0)
    return max(f_err, ur_err, lam_err), dict(force=float(r["force"][-1]), closed=force, lateral=l1, f_err=f_err,
                                              ur_err=ur_err)


def confined_compression_error(stretch: float, nr: int = 7, nz: int = 5, radius: float = 2e-3, height: float = 4e-3):
    """The side held radially: u_R = 0 everywhere and the force P_ZZ(diag(1, 1, stretch)) pi R^2. The lateral
    stress is not zero here, so the radial equation tests the quadrature weight (which uniform stretch, with zero
    lateral stress, cannot)."""
    sp = stretched_cylinder(radius=radius, height=height, nr=nr, nz=nz, confined=True)
    mat = material()
    r = run(Solver(sp, "elastic"), mat, np.linspace(0.0, (stretch - 1.0) * height, 6)[1:])
    P3 = float(jax.grad(lambda c: hom.psi(jnp.diag(jnp.stack([jnp.asarray(1.0), jnp.asarray(1.0), c])), mat))(
        jnp.asarray(stretch)))
    force = P3 * np.pi * radius ** 2
    u = r["final_u"]
    ur_err = np.max(np.abs(u[..., 0])) / (abs(stretch - 1.0) * radius)
    f_err = abs(r["force"][-1] - force) / abs(force)
    return max(f_err, ur_err), dict(force=float(r["force"][-1]), closed=force, f_err=f_err, ur_err=ur_err)


@pytest.mark.parametrize("stretch", [1.3, 0.7])
def test_confined_compression_equals_the_closed_form(stretch):
    err, info = confined_compression_error(stretch)
    assert err < 1e-10, info


@pytest.mark.parametrize("stretch", [1.3, 0.7])
def test_uniform_stretch_equals_the_closed_form(stretch):
    err, info = uniform_stretch_error(stretch)
    assert err < 1e-10, info


# --- element properties (Hypothesis, 100 examples, never shrunk) ---------------------------------------------------
sizes = st.floats(min_value=0.2e-3, max_value=2e-3)
radii = st.one_of(st.just(0.0), st.floats(min_value=0.1e-3, max_value=10e-3))
stretches = st.floats(min_value=0.7, max_value=1.4)
shears = st.floats(min_value=-0.3, max_value=0.3)


def _homogeneous_ue(x0, dx, dy, a, b, c):
    """Nodal displacements of the axisymmetric homogeneous field u_R = (a - 1) R, u_Z = c R + (b - 1) Z."""
    R = np.array([x0, x0 + dx, x0 + dx, x0])
    Z = np.array([0.0, 0.0, dy, dy])
    return jnp.asarray(np.stack([(a - 1.0) * R, c * R + (b - 1.0) * Z], axis=1))


def check_homogeneous_energy():
    @settings(max_examples=100, deadline=None, derandomize=False)
    @given(radii, sizes, sizes, stretches, stretches, shears)
    def prop(x0, dx, dy, a, b, c):
        mat = material()
        ue = _homogeneous_ue(x0, dx, dy, a, b, c)
        e = float(axi.element_strain_energy(ue, jnp.zeros(4), jnp.asarray(x0), jnp.asarray(dx), jnp.asarray(dy), mat))
        F = jnp.asarray([[a, 0.0, 0.0], [c, b, 0.0], [0.0, 0.0, a]])
        vol = np.pi * ((x0 + dx) ** 2 - x0 ** 2) * dy
        expect = float(hom.psi(F, mat)) * vol
        assert e == pytest.approx(expect, rel=1e-11, abs=1e-14 * float(mat.mu) * vol)
    prop()


def check_axial_translation_is_free():
    @settings(max_examples=100, deadline=None, derandomize=False)
    @given(radii, sizes, sizes, st.floats(min_value=-1e-3, max_value=1e-3))
    def prop(x0, dx, dy, t):
        mat = material()
        ue = jnp.zeros((4, 2)).at[:, 1].set(t)
        args = (jnp.zeros(4), jnp.asarray(x0), jnp.asarray(dx), jnp.asarray(dy), mat)
        e = float(axi.element_strain_energy(ue, *args))
        g = np.asarray(jax.grad(lambda v: axi.element_strain_energy(v, *args))(ue))
        scale = float(mat.mu) * np.pi * ((x0 + dx) ** 2 - x0 ** 2) * dy
        assert abs(e) <= 1e-12 * scale
        assert np.max(np.abs(g)) <= 1e-9 * scale / max(dx, dy)
    prop()


def check_hessian_symmetric():
    @settings(max_examples=100, deadline=None, derandomize=False)
    @given(radii, sizes, sizes, st.lists(st.floats(min_value=-0.1, max_value=0.1), min_size=8, max_size=8))
    def prop(x0, dx, dy, v):
        mat = material(1e-6)
        ue = jnp.asarray(v).reshape(4, 2) * min(dx, dy)
        ue = ue.at[:, 0].multiply(jnp.where(x0 == 0.0, jnp.asarray([0.0, 1.0, 1.0, 0.0]), 1.0))  # u_R = 0 on the axis
        H = np.asarray(axi.element_hessian(ue, jnp.full(4, 0.3), jnp.asarray(x0), jnp.asarray(dx), jnp.asarray(dy),
                                           mat))
        assert np.all(np.isfinite(H))
        np.testing.assert_allclose(H, H.T, rtol=1e-10, atol=1e-12 * np.max(np.abs(H)))
    prop()


def test_homogeneous_energy_equals_psi_times_the_ring_volume():
    check_homogeneous_energy()


def test_axial_translation_stores_no_energy():
    check_axial_translation_is_free()


def test_element_hessian_is_symmetric_and_finite_on_the_axis():
    check_hessian_symmetric()


def test_gauss_points_sit_off_the_axis():
    R = np.asarray(axi.gauss_radii(jnp.asarray(0.0), jnp.asarray(1.0)))
    assert np.all(R > 0.2) and np.all(R < 0.8)
