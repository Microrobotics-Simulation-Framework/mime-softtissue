"""The fit's tests against closed forms (each would fail under the error it names)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime_softtissue.fit import homogeneous as hom

MU, KAPPA = 300.0, 650.0


def nh(kappa=KAPPA, **fibre):
    p = {"mu": jnp.asarray(MU), "kappa": jnp.asarray(kappa)}
    p.update({k: jnp.asarray(v) for k, v in fibre.items()})
    return hom.material_from(p)


@pytest.mark.parametrize("gamma", [1e-4, 0.3, 1.5])
def test_neo_hookean_simple_shear_is_linear(gamma):
    # J = 1 and J^-2/3 I1 - 3 = gamma^2: sigma = mu gamma exactly (a wrong I1 or a volumetric leak breaks it)
    assert float(hom.shear_stress(jnp.asarray(gamma), nh())) == pytest.approx(MU * gamma, rel=1e-13)


@pytest.mark.parametrize("geometry", ["cone_plate", "parallel_plate"])
@pytest.mark.parametrize("gamma0", [1e-3, 0.5])
def test_storage_modulus_of_a_linear_shear_law_is_mu(geometry, gamma0):
    assert float(hom.storage_modulus(jnp.asarray(gamma0), nh(), geometry)) == pytest.approx(MU, rel=1e-12)


def test_parallel_plate_reads_a_stiffening_law_below_cone_plate():
    # the rim's strain is the largest; the torque averages smaller strains, so a stiffening law reads softer
    m = nh(k1=100.0, k2=1.0)
    cone = float(hom.storage_modulus(jnp.asarray(0.8), m, "cone_plate"))
    plate = float(hom.storage_modulus(jnp.asarray(0.8), m, "parallel_plate"))
    assert plate < cone


def test_small_strain_fibre_shear_modulus():
    # one family is in tension on each half-cycle: G' -> mu + k1 sin^2(2 angle) as the amplitude -> 0; the next term
    # is of order k1 gamma0 (the gamma^2 sin^2 part of I4), 3e-7 here; much smaller amplitudes lose I4 - 1 to rounding
    m = nh(k1=100.0, k2=1.0)
    assert float(hom.storage_modulus(jnp.asarray(1e-6), m)) == pytest.approx(MU + 100.0, rel=1e-6)


def test_compression_small_strain_young_modulus():
    E = 9 * KAPPA * MU / (3 * KAPPA + MU)
    eps = 1e-7
    assert float(hom.compression_stress(jnp.asarray(eps), nh())) / eps == pytest.approx(E, rel=1e-5)


@pytest.mark.parametrize("strain", [0.2, 0.6])
def test_compression_incompressible_limit(strain):
    lam = 1.0 - strain
    expect = -MU * (lam - lam ** -2)
    got = float(hom.compression_stress(jnp.asarray(strain), nh(kappa=1e7 * MU)))
    assert got == pytest.approx(expect, rel=1e-5)


@pytest.mark.parametrize("strain", [0.05, 0.3, 0.6])
def test_lateral_solve_leaves_no_lateral_stress(strain):
    for m in (nh(), nh(k1=100.0, k2=1.0)):
        assert float(hom.lateral_residual(jnp.asarray(strain), m)) < 1e-10


def test_compression_derivative_matches_central_difference():
    def f(mu):
        return hom.compression_stress(jnp.asarray(0.4), hom.material_from({"mu": mu, "kappa": jnp.asarray(KAPPA),
                                                                            "k1": jnp.asarray(80.0),
                                                                            "k2": jnp.asarray(1.5)}))
    g = float(jax.grad(f)(jnp.asarray(MU)))
    h = 1e-3
    cd = (float(f(jnp.asarray(MU + h))) - float(f(jnp.asarray(MU - h)))) / (2 * h)
    assert g == pytest.approx(cd, rel=1e-7)


@pytest.mark.parametrize("geometry", ["cone_plate", "parallel_plate"])
def test_quarter_cycle_rule_equals_the_whole_cycle(geometry):
    m = nh(k1=100.0, k2=1.0)
    q = float(hom.storage_modulus(jnp.asarray(0.9), m, geometry))
    w = float(hom.storage_modulus(jnp.asarray(0.9), m, geometry, full_cycle=True))
    assert q == pytest.approx(w, rel=1e-9)


def test_compression_jacobian_is_finite_and_matches_central_difference():
    def f(th):
        return hom.compression_stress(jnp.asarray(0.35), hom.material_from(
            {"mu": jnp.exp(th[0]), "kappa": jnp.exp(th[1]), "k1": jnp.exp(th[2]), "k2": jnp.exp(th[3])}))
    th = jnp.log(jnp.asarray([200.0, 433.0, 100.0, 1.0]))
    J = np.asarray(jax.jacfwd(f)(th))
    cd = np.array([(float(f(th.at[i].add(1e-6))) - float(f(th.at[i].add(-1e-6)))) / 2e-6 for i in range(4)])
    assert np.all(np.isfinite(J))
    np.testing.assert_allclose(J, cd, rtol=1e-6, atol=1e-9 * np.max(np.abs(cd)))
