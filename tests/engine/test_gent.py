"""Gent's law (``Material.gent_jm``) against its closed forms in incompressible uniaxial and equibiaxial tension, alone and
through the incompressible plane-stress kinematics; its continuation past the limit; and the law without it unchanged.

The lines, stated before the checks: the nominal stress equals mu (l - l^-2) / (1 - x / Jm) (uniaxial, x = l^2 + 2/l - 3)
and mu (l - l^-5) / (1 - x / Jm) (equibiaxial, x = 2 l^2 + l^-4 - 3) to 1e-12 relative; the continuation meets the
logarithm with its value and slope at x* to 1e-12; a record without ``gent_jm`` gives the energy it gave before to the
last bit. The plant (the limit read 5 % high inside the logarithm) must fail every closed-form point."""
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime_softtissue.engine import material as law
from mime_softtissue.engine.kinematics import plane_strain as ps
from mime_softtissue.engine.material import GENT_CUT, Material

MU, JM = 7.2e3, 1.6
STRETCHES = (1.02, 1.2, 1.45, 1.6)  # x up to 0.88 Jm uniaxial; 1.6 equibiaxial is past the limit and dropped there


def material(jm=JM, kappa=0.0) -> Material:
    return Material(mu=jnp.asarray(MU), kappa=jnp.asarray(kappa), gc=jnp.asarray(1.0), l=jnp.asarray(1e-3),
                    k_res=jnp.asarray(1e-6), stretch_limit=jnp.asarray(10.0), gent_jm=None if jm is None else jnp.asarray(jm))


def incompressible_energy(I1, mat):
    plus, minus = law.energy_parts(jnp.asarray(1.0), I1, (), mat)
    return plus + minus


def uniaxial_stress(lam, mat):
    return float(jax.grad(lambda l: incompressible_energy(l ** 2 + 2 / l, mat))(jnp.asarray(lam)))


def equibiaxial_stress(lam, mat):
    """The nominal stress in each in-plane direction: half the energy's derivative in the common stretch."""
    return 0.5 * float(jax.grad(lambda l: incompressible_energy(2 * l ** 2 + l ** -4, mat))(jnp.asarray(lam)))


def uniaxial_closed(lam, jm=JM):
    return MU * (lam - lam ** -2) / (1 - (lam ** 2 + 2 / lam - 3) / jm)


def equibiaxial_closed(lam, jm=JM):
    return MU * (lam - lam ** -5) / (1 - (2 * lam ** 2 + lam ** -4 - 3) / jm)


def plane_stress_density(F, mat):
    plus, minus = ps.energy_parts(F, mat, True)
    return plus + minus


@contextmanager
def limit_read_high():
    """The plant: the limit read 5 % high inside the logarithm (a slip the small-strain modulus does not show)."""
    real = law.gent_energy

    def gent_energy(x, mu, jm):
        x_star = jm * (1.0 - GENT_CUT)
        inside = -0.5 * mu * jm * jnp.log1p(-jnp.minimum(x, x_star) / (1.05 * jm))
        return inside + (0.5 * mu / GENT_CUT) * jnp.maximum(x - x_star, 0.0)
    law.gent_energy = gent_energy
    try:
        yield
    finally:
        law.gent_energy = real


@pytest.mark.parametrize("lam", STRETCHES)
def test_uniaxial_stress_is_the_closed_form(lam):
    assert uniaxial_stress(lam, material()) == pytest.approx(uniaxial_closed(lam), rel=1e-12)


@pytest.mark.parametrize("lam", [s for s in STRETCHES if 2 * s ** 2 + s ** -4 - 3 < JM])
def test_equibiaxial_stress_is_the_closed_form(lam):
    assert equibiaxial_stress(lam, material()) == pytest.approx(equibiaxial_closed(lam), rel=1e-12)


@pytest.mark.parametrize("lam", [1.2, 1.45])
def test_uniaxial_through_the_plane_stress_kinematics(lam):
    """The deformation gradient's path: an incompressible sheet (F33 = 1 / det F2, no bulk term) stretched by ``lam``,
    its width relaxed by Newton to zero lateral stress (lam^-1/2), carries the closed-form stress."""
    mat = material()

    def w(a):
        return plane_stress_density(jnp.diag(jnp.stack([a, jnp.asarray(lam)])), mat)
    a = jnp.asarray(1.0)
    for _ in range(40):
        a = a - jax.grad(w)(a) / jax.hessian(w)(a)
    P = float(jax.grad(plane_stress_density)(jnp.diag(jnp.stack([a, jnp.asarray(lam)])), mat)[1, 1])
    assert float(a) == pytest.approx(lam ** -0.5, rel=1e-12)
    assert P == pytest.approx(uniaxial_closed(lam), rel=1e-12)


@pytest.mark.parametrize("lam", [1.1, 1.3])
def test_equibiaxial_through_the_plane_stress_kinematics(lam):
    mat = material()
    P = np.asarray(jax.grad(plane_stress_density)(lam * jnp.eye(2), mat))
    assert P[0, 0] == pytest.approx(equibiaxial_closed(lam), rel=1e-12)
    assert P[1, 1] == pytest.approx(equibiaxial_closed(lam), rel=1e-12)


def test_the_plant_fails_every_closed_form_point():
    """Caught n of n: every uniaxial and equibiaxial point misses its closed form by more than the line."""
    missed = []
    with limit_read_high():
        for lam in STRETCHES:
            missed.append(abs(uniaxial_stress(lam, material()) / uniaxial_closed(lam) - 1) > 1e-12)
            if 2 * lam ** 2 + lam ** -4 - 3 < JM:
                missed.append(abs(equibiaxial_stress(lam, material()) / equibiaxial_closed(lam) - 1) > 1e-12)
    assert len(missed) == 7 and all(missed)


def test_small_strain_shear_modulus_is_mu():
    """Simple shear by gamma at J = 1 (I1 = 3 + gamma^2): the stress is mu gamma at small gamma, as neo-Hookean's."""
    g = 1e-6
    tau = float(jax.grad(lambda t: incompressible_energy(3.0 + t ** 2, material()))(jnp.asarray(g)))
    assert tau == pytest.approx(MU * g, rel=1e-9)


def test_the_continuation_meets_the_logarithm_and_stays_finite():
    x_star = JM * (1 - GENT_CUT)
    f = lambda x: law.gent_energy(x, MU, JM)  # noqa: E731
    at_cut = -0.5 * MU * JM * np.log(GENT_CUT)
    slope = MU / 2 / GENT_CUT  # the logarithm's slope at x*
    assert float(f(jnp.asarray(x_star))) == pytest.approx(at_cut, rel=1e-12)
    below, above = x_star * (1 - 1e-9), x_star * (1 + 1e-9)
    assert float(jax.grad(f)(jnp.asarray(below))) == pytest.approx(slope, rel=1e-5)
    assert float(jax.grad(f)(jnp.asarray(above))) == pytest.approx(slope, rel=1e-12)
    for x in (above, 2.0 * JM, 30.0 * JM):  # linear past x*, from the logarithm's value there
        assert float(f(jnp.asarray(x))) == pytest.approx(at_cut + slope * (x - x_star), rel=1e-12)


@pytest.mark.parametrize("I1", [3.0, 3.4, 9.0])
@pytest.mark.parametrize("J", [0.9, 1.0, 1.3])
def test_without_the_limit_the_law_is_unchanged_to_the_last_bit(I1, J):
    """``gent_jm=None`` (every record without it) takes the neo-Hookean term as written before Gent was added."""
    mat = material(jm=None, kappa=3e4)._replace(c2=jnp.asarray(50.0), c3=jnp.asarray(0.0))
    Jj, I1j = jnp.asarray(J), jnp.asarray(I1)
    plus, minus = law.energy_parts(Jj, I1j, (), mat)
    x = Jj ** (-2.0 / 3.0) * I1j - 3.0  # the expression as it stood before Gent, in the same order
    iso = 0.5 * mat.mu * x + mat.c2 * x ** 2 + mat.c3 * x ** 3
    vol = 0.25 * mat.kappa * (Jj ** 2 - 1.0 - 2.0 * jnp.log(Jj))
    assert float(plus) == float(iso + jnp.where(Jj >= 1.0, vol, 0.0))
    assert float(minus) == float(jnp.where(Jj < 1.0, vol, 0.0))


# --- the two pairings Gent's term would make wrong, refused (``MaterialRefused``) -------------------------------------
PRONY = law.Prony(beta=jnp.asarray([0.5, 0.2]), tau=jnp.asarray([1.0, 30.0]))
C_STRETCHED = jnp.asarray([1.44, 1 / 1.2, 1 / 1.2, 0.0, 0.0, 0.0])  # incompressible uniaxial stretch 1.2, Voigt order


def test_gent_with_prony_branches_is_refused():
    """The branches are sized to the Yeoh law's slope (``branch_strain``), which does not follow Gent's: every entry to
    them refuses a material with both, and each alone is taken."""
    both = material()._replace(prony=PRONY)
    for call in (lambda: law.branch_strain(C_STRETCHED, both),
                 lambda: law.initial_strains((2,), both),
                 lambda: law.viscous_energies(C_STRETCHED, jnp.zeros((2, 6)), both, 0.1)):
        with pytest.raises(law.MaterialRefused) as refused:
            call()
        assert refused.value.code == "softtissue.gent-viscous"
    assert np.isfinite(np.asarray(law.branch_strain(C_STRETCHED, material(jm=None)._replace(prony=PRONY)))).all()
    assert np.isfinite(float(law.energy_parts(jnp.asarray(1.0), jnp.asarray(3.3), (), material())[0]))


def test_gent_with_the_3d_neo_residual_is_refused():
    """``residual="neo"`` strips Yeoh's terms past full damage but not Gent's excess over the neo-Hookean term: the 3-D
    engine refuses the pair when it is built, and takes each alone."""
    from mime_softtissue.engine import puncture3d as P

    blk = P.Block(half_width=1e-3, height=1e-3, h=0.5e-3, fine_xy=0.5e-3, fine_z=0.5e-3)
    tip = P.sharp_cone(0.35, 0.05e-3)
    gent = Material(mu=MU, kappa=50 * MU, gc=1.0, l=0.25e-3, k_res=1e-6, stretch_limit=10.0, gent_jm=JM)
    with pytest.raises(law.MaterialRefused) as refused:
        P.Puncture3D(blk, gent, tip, P.Settings(residual="neo"))
    assert refused.value.code == "softtissue.gent-neo-residual"
    assert P.Puncture3D(blk, gent, tip, P.Settings(residual="residual")).mat.gent_jm == JM
    assert P.Puncture3D(blk, gent._replace(gent_jm=None), tip, P.Settings(residual="neo")).mat.gent_jm is None
