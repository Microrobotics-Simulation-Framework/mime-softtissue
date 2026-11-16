"""The mean-dilatation switch of the 2-D kinematics (``volumetric="mean"``, axisymmetric and plane strain) against
independent oracles, with the planted error each check must catch.

- The element's Jbar against the deformed element's own volume: the image of a Q1 element is the quadrilateral through
  its four displaced nodes, so its ring volume is Pappus's 2 pi A r_c (the polygon's area and centroid, no quadrature),
  and in plane strain its area is the shoelace's. Checked through the energy at a purely volumetric material (mu = 0),
  so the whole path (Jbar, the weights' sum, the split) is under test, not one helper.
- Homogeneous deformation: both treatments store the same energy (any damage) and the same internal force (uniform
  damage) to round-off; through the solver, both reproduce the uniform-stretch closed form.

The plant: Jbar taken as the plain mean of the four Gauss points' J (the 2 pi R weight left out). Its detection rate is
measured in ``scripts`` (LOCK2D's work folder), not assumed."""
from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mime_softtissue.engine.kinematics import axisymmetric as axi
from mime_softtissue.engine.kinematics import plane_strain as ps
from mime_softtissue.engine.loadpath import run
from mime_softtissue.engine.material import Material, shear_and_bulk
from mime_softtissue.engine.solver import Solver
from mime_softtissue.engine.specimen import stretched_cylinder

E, NU = 50e3, 0.45


def material(k_res=1e-6, mu_factor=1.0):
    mu, kappa = shear_and_bulk(E, NU)
    one = jnp.asarray(1.0)
    return Material(mu=jnp.asarray(mu * mu_factor), kappa=jnp.asarray(kappa), gc=one, l=one, k_res=jnp.asarray(k_res),
                    stretch_limit=one)


@contextlib.contextmanager
def unweighted_mean():
    """The plant: the axisymmetric Jbar without the ring weight."""
    old = axi.mean_dilatation
    axi.mean_dilatation = lambda J, w: jnp.mean(J)
    try:
        yield
    finally:
        axi.mean_dilatation = old


# --- the oracle ------------------------------------------------------------------------------------------------------
def corners(x0, dx, dy):
    return np.array([[x0, 0.0], [x0 + dx, 0.0], [x0 + dx, dy], [x0, dy]])


def polygon_area_and_moment(p):
    """(area, integral of x over the area) of the polygon through ``p`` (n, 2), counterclockwise."""
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    return 0.5 * np.sum(cross), np.sum((x + xn) * cross) / 6.0


def ring_jbar(x0, dx, dy, ue):
    """The deformed ring's volume over the reference ring's (Pappus)."""
    ref = corners(x0, dx, dy)
    _, moment = polygon_area_and_moment(ref + np.asarray(ue))
    return moment / ((x0 + 0.5 * dx) * dx * dy)


def plane_jbar(dx, dy, ue):
    area, _ = polygon_area_and_moment(corners(0.0, dx, dy) + np.asarray(ue))
    return area / (dx * dy)


def psi_vol(J, kappa):
    return 0.25 * kappa * (J * J - 1.0 - 2.0 * np.log(J))


# --- strategies (physical scales: a body's elements, displacements a fraction of the element) ---------------------
sizes = st.floats(min_value=0.2e-3, max_value=2e-3)
radii = st.one_of(st.just(0.0), st.floats(min_value=0.1e-3, max_value=10e-3))
nodal = st.lists(st.floats(min_value=-0.15, max_value=0.15), min_size=8, max_size=8)
stretches = st.floats(min_value=0.7, max_value=1.4)
shears = st.floats(min_value=-0.3, max_value=0.3)
damage = st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=4, max_size=4)


def distorted(x0, dx, dy, v):
    ue = np.asarray(v).reshape(4, 2) * min(dx, dy)
    if x0 == 0.0:
        ue[[0, 3], 0] = 0.0  # u_R = 0 on the axis
    return ue


def volumetric_energy_error(x0, dx, dy, ue, mat):
    """|E_mean - V psi_vol(Jbar_oracle)| over its scale, at mu = 0, no damage and k_res = 0 (g = (1 - d)^2 + k_res
    multiplies the tension side, so an intact element in tension stores (1 + k_res) psi)."""
    jo = ring_jbar(x0, dx, dy, ue)
    V = 2.0 * np.pi * (x0 + 0.5 * dx) * dx * dy
    e = float(axi.element_strain_energy(jnp.asarray(ue), jnp.zeros(4), jnp.asarray(x0), jnp.asarray(dx),
                                        jnp.asarray(dy), mat, "mean"))
    kappa = float(mat.kappa)
    expect = V * psi_vol(jo, kappa)
    scale = V * kappa * (abs(jo - 1.0 / jo) + 1e-3)  # psi_vol's slope in J, so a Jbar error reads at its own size
    return abs(e - expect) / scale


def check_ring_jbar_is_the_deformed_volume():
    @settings(max_examples=100, deadline=None, derandomize=False)
    @given(radii, sizes, sizes, nodal)
    def prop(x0, dx, dy, v):
        ue = distorted(x0, dx, dy, v)
        assert volumetric_energy_error(x0, dx, dy, ue, material(k_res=0.0, mu_factor=0.0)) < 1e-10
    prop()


def check_plane_jbar_is_the_deformed_area():
    @settings(max_examples=100, deadline=None, derandomize=False)
    @given(sizes, sizes, nodal)
    def prop(dx, dy, v):
        mat = material(k_res=0.0, mu_factor=0.0)
        ue = distorted(1.0, dx, dy, v)
        jo = plane_jbar(dx, dy, ue)
        e = float(ps.element_strain_energy(jnp.asarray(ue), jnp.zeros(4), dx, dy, mat, False, "mean"))
        kappa = float(mat.kappa)
        expect = dx * dy * psi_vol(jo, kappa)
        assert abs(e - expect) <= 1e-10 * dx * dy * kappa * (abs(jo - 1.0 / jo) + 1e-3)
    prop()


def homogeneous_ue(x0, dx, dy, a, b, c):
    """u_R = (a - 1) R, u_Z = c R + (b - 1) Z (the axisymmetric homogeneous field)."""
    p = corners(x0, dx, dy)
    return jnp.asarray(np.stack([(a - 1.0) * p[:, 0], c * p[:, 0] + (b - 1.0) * p[:, 1]], axis=1))


def check_homogeneous_agreement():
    @settings(max_examples=100, deadline=None, derandomize=False)
    @given(radii, sizes, sizes, stretches, stretches, shears, damage, st.floats(min_value=0.0, max_value=1.0))
    def prop(x0, dx, dy, a, b, c, dq, d_uniform):
        mat = material()
        ue = homogeneous_ue(x0, dx, dy, a, b, c)
        args = (jnp.asarray(x0), jnp.asarray(dx), jnp.asarray(dy), mat)
        vol = np.pi * ((x0 + dx) ** 2 - x0 ** 2) * dy
        scale = float(mat.kappa) * vol
        dq = jnp.asarray(dq)
        e_g = float(axi.element_strain_energy(ue, dq, *args))
        e_m = float(axi.element_strain_energy(ue, dq, *args, "mean"))
        assert abs(e_g - e_m) <= 1e-12 * scale
        du = jnp.full(4, d_uniform)
        f_g = np.asarray(jax.grad(lambda v: axi.element_strain_energy(v, du, *args))(ue))
        f_m = np.asarray(jax.grad(lambda v: axi.element_strain_energy(v, du, *args, "mean"))(ue))
        assert np.max(np.abs(f_g - f_m)) <= 1e-11 * scale / min(dx, dy)
        p_g = np.asarray(axi.element_plus(ue, *args))
        p_m = np.asarray(axi.element_plus(ue, *args, "mean"))
        assert np.max(np.abs(p_g - p_m)) <= 1e-12 * float(mat.kappa)
        # plane strain: the same field with x0 = 0 and no hoop term
        F = jnp.asarray([[a, 0.0], [c, b]])
        uep = jnp.asarray(corners(0.0, dx, dy) @ (np.asarray(F) - np.eye(2)).T)
        g2 = float(ps.element_strain_energy(uep, dq, dx, dy, mat))
        m2 = float(ps.element_strain_energy(uep, dq, dx, dy, mat, False, "mean"))
        assert abs(g2 - m2) <= 1e-12 * float(mat.kappa) * dx * dy
    prop()


def check_hessian_symmetric_and_finite():
    @settings(max_examples=100, deadline=None, derandomize=False)
    @given(radii, sizes, sizes, nodal)
    def prop(x0, dx, dy, v):
        ue = jnp.asarray(distorted(x0, dx, dy, np.asarray(v) * 0.5))
        H = np.asarray(axi.element_hessian(ue, jnp.full(4, 0.3), jnp.asarray(x0), jnp.asarray(dx), jnp.asarray(dy),
                                           material(), "mean"))
        assert np.all(np.isfinite(H))
        np.testing.assert_allclose(H, H.T, rtol=1e-10, atol=1e-12 * np.max(np.abs(H)))
    prop()


def test_ring_jbar_is_the_deformed_ring_volume():
    check_ring_jbar_is_the_deformed_volume()


def test_ring_jbar_plant_is_caught():
    """One fixed draw on the axis column (the rate over many draws is measured outside the suite)."""
    ue = distorted(0.0, 1e-3, 1e-3, [0.0, 0.1, 0.1, -0.05, 0.12, 0.02, 0.0, 0.1])
    mat = material(k_res=0.0, mu_factor=0.0)
    assert volumetric_energy_error(0.0, 1e-3, 1e-3, ue, mat) < 1e-10
    with unweighted_mean():
        assert volumetric_energy_error(0.0, 1e-3, 1e-3, ue, mat) > 1e-3


def test_plane_jbar_is_the_deformed_area():
    check_plane_jbar_is_the_deformed_area()


def test_homogeneous_deformation_both_treatments_agree():
    check_homogeneous_agreement()


def test_mean_hessian_is_symmetric_and_finite():
    check_hessian_symmetric_and_finite()


@pytest.mark.parametrize("stretch", [1.3, 0.7])
def test_uniform_stretch_closed_form_with_either_treatment(stretch):
    """The cylinder's uniform stretch through the solver: homogeneous, so mean dilatation reproduces full
    integration's answer (and the closed form) to round-off."""
    out = {}
    for vol in ("gauss", "mean"):
        sp = stretched_cylinder(radius=2e-3, height=4e-3, nr=7, nz=5)
        r = run(Solver(sp, "elastic", volumetric=vol), material(k_res=0.0), np.linspace(0.0, (stretch - 1.0) * 4e-3,
                                                                                         6)[1:])
        out[vol] = (np.asarray(r["force"]), np.asarray(r["final_u"]))
    fg, ug = out["gauss"]
    fm, um = out["mean"]
    assert np.max(np.abs(fg - fm)) <= 1e-10 * np.max(np.abs(fg))
    assert np.max(np.abs(ug - um)) <= 1e-10 * np.max(np.abs(ug))


def test_plane_stress_refuses_the_mean():
    with pytest.raises(ValueError):
        ps.element_strain_energy(jnp.zeros((4, 2)), jnp.zeros(4), 1.0, 1.0, material(), True, "mean")
