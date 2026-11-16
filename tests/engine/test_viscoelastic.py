"""The viscoelastic branches (``material.Prony``): relaxation, creep, the energy account and the slow-rate limit.

The lines, stated before the checks:
- **relaxation:** after a step strain the force is the Prony closed form, F_inf + sum F_i prod (1 + r_i)^-1 for the
  backward-Euler steps, to 1e-9 at the material point and 1e-5 on the FE sheet with its width held (Newton's
  tolerance); against exp(-t / tau_i) to 1 % at r = dt / tau <= 0.01. The closed form is written out here (Yeoh's
  pure-shear stress and the branch energy's derivative), not taken from the law. Planted: tau_1 x 1.1, x 1.5, x 0.67
  in the model must each fail the 1e-9 line.
- **creep:** at small strain (stress 1e-4 of mu) under a constant load, one branch is the standard linear solid:
  strain(t) = s / E_inf - (s / E_inf - s / E_0) exp(-t / tau_c), tau_c = tau (1 + beta), to 1 % (r <= 1e-3).
  Planted: the internal strain never updated must fail it.
- **energy:** on a notched sheet loaded at a rate, |W - (U + D + Gamma)| <= 2 % of W up to the peak. Planted: the
  branches' dissipation D dropped must fail it (the run's D is checked to exceed 2 % of W, so the plant can fail).
- **slow rate:** at a rate 1e-6 times the relaxation's, the viscous run carries the elastic run's force to 1e-4 of the
  peak (the branches relax every step). Planted: the long-term modulus taken as the instantaneous one (1 + sum beta)
  must fail it.
- **every build:** the axisymmetric cylinder with its side held and the plane-strain sheet with its width held are the
  same uniaxial strain; their relaxing stresses agree to 1e-9 (the stress relaxing by over 15 %, so the branches are
  in it)."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime_softtissue.engine import material as law
from mime_softtissue.engine.loadpath import run
from mime_softtissue.engine.material import Material, Prony
from mime_softtissue.engine.solver import Solver
from mime_softtissue.engine.specimen import edge_notched_strip, stretched_cylinder
from mime_softtissue.fit.viscous import pure_shear_path

C1, C2 = 361.0, 438.0
BETA, TAU = (0.52, 0.13), (7.0, 63.0)
WIDTH, HALF, T = 40e-3, 5e-3, 3e-3


def material(beta=BETA, tau=TAU, c2=C2, gc=1.0, l=1e-3, kappa=0.0) -> Material:
    return Material(mu=jnp.asarray(2 * C1), kappa=jnp.asarray(kappa), gc=jnp.asarray(gc), l=jnp.asarray(l),
                    k_res=jnp.asarray(1e-6), stretch_limit=jnp.asarray(10.0), c2=jnp.asarray(c2),
                    prony=Prony(jnp.asarray(beta), jnp.asarray(tau)))


def long_term_stress(lam, c2=C2):
    x = lam ** 2 + lam ** -2 - 2
    return 2 * (lam - lam ** -3) * (C1 + 2 * c2 * x)


def branch_stress(lam, beta, c2=C2):
    """d/dlam of beta mu kappa^2 |E*|^2 in pure shear: E* = diag(0, l^2 - l^-2, l^-2 - l^2) / 4, kappa^2 = 1 + 2 c2 x / mu."""
    mu = 2 * C1
    x = lam ** 2 + lam ** -2 - 2
    dx = 2 * lam - 2 * lam ** -3
    s = lam ** 2 - lam ** -2
    ds = 2 * lam + 2 * lam ** -3
    return beta * mu * ((2 * c2 / mu) * dx * s ** 2 / 8 + (1 + 2 * c2 * x / mu) * 2 * s * ds / 8)


def closed_form_relaxation(lam, dts, beta=BETA, tau=TAU):
    """The step at dts[0], then holds: F_inf + sum F_i prod_k (1 + dt_k / tau_i)^-1."""
    out = long_term_stress(lam)
    for b, t in zip(beta, tau):
        out = out + branch_stress(lam, b) / np.cumprod(1 + np.asarray(dts) / t)
    return out


def step_and_hold(n=400, dt=0.05, first=1e-9):
    dts = np.full(n, dt)
    dts[0] = first
    return dts


@pytest.mark.parametrize("lam", [1.1, 1.3, 1.6])
def test_relaxation_at_the_material_point_is_the_prony_closed_form(lam):
    dts = step_and_hold()
    p = np.asarray(pure_shear_path(jnp.full(len(dts), lam), jnp.asarray(dts), material()).stress)
    assert np.max(np.abs(p / closed_form_relaxation(lam, dts) - 1)) <= 1e-9


def test_relaxation_follows_the_exponentials_when_the_steps_are_short():
    lam, dt = 1.3, 0.01 * min(TAU)
    dts = step_and_hold(n=2000, dt=dt)
    p = np.asarray(pure_shear_path(jnp.full(len(dts), lam), jnp.asarray(dts), material()).stress)
    t = np.cumsum(dts) - dts[0]
    exact = long_term_stress(lam) + sum(branch_stress(lam, b) * np.exp(-t / tau) for b, tau in zip(BETA, TAU))
    assert np.max(np.abs(p / exact - 1)) <= 0.01


@pytest.mark.parametrize("factor", [1.1, 1.5, 0.67])
def test_a_wrong_relaxation_time_fails_the_relaxation_check(factor):
    lam = 1.3
    dts = step_and_hold()
    wrong = material(tau=(TAU[0] * factor, TAU[1]))
    p = np.asarray(pure_shear_path(jnp.full(len(dts), lam), jnp.asarray(dts), wrong).stress)
    assert np.max(np.abs(p / closed_form_relaxation(lam, dts) - 1)) > 1e-9 * 1e3


def creep(mat, target, dts, update=law.updated_strains):
    """The material point (pure shear) under a constant nominal stress ``target``: each step's stretch solves
    d W_step / d lam = target by 20 Newton iterations; the internal strains then update by ``update``."""
    from mime_softtissue.fit import viscous as fv

    def step(carry, h):
        lam, gamma = carry

        def potential(q):
            return fv._long_term(q, mat) + law.viscous_energies(fv._cauchy_green(q), gamma, mat, h)[0]

        def newton(_, q):
            return q - (jax.grad(potential)(q) - target) / jax.hessian(potential)(q)

        lam = jax.lax.fori_loop(0, 20, newton, lam)
        return (lam, update(fv._cauchy_green(lam), gamma, mat, h)), lam - 1.0

    run_all = jax.jit(lambda d: jax.lax.scan(step, (jnp.asarray(1.0), law.initial_strains((), mat)), d)[1])
    return np.asarray(run_all(jnp.asarray(dts)))


def creep_case():
    mat = material(beta=(0.6,), tau=(5.0,), c2=0.0)
    e_inf = 4 * float(mat.mu)  # pure shear, incompressible plane stress, small strain: P = 4 mu e
    e_0 = e_inf * (1 + 0.6)
    target = 1e-4 * float(mat.mu)
    dts = np.full(1500, 5e-3)
    dts[0] = 1e-9
    t = np.cumsum(dts) - dts[0]
    exact = target / e_inf - (target / e_inf - target / e_0) * np.exp(-t / (5.0 * 1.6))
    return mat, target, dts, exact


def test_creep_is_the_standard_linear_solid_at_small_strain():
    mat, target, dts, exact = creep_case()
    strain = creep(mat, target, dts)
    assert np.max(np.abs(strain / exact - 1)) <= 0.01


def test_creep_without_the_internal_update_fails_the_creep_check():
    mat, target, dts, exact = creep_case()
    strain = creep(mat, target, dts, update=lambda c, gamma, m, h: gamma)
    assert np.max(np.abs(strain / exact - 1)) > 0.01


def held_width_sheet():
    sp = edge_notched_strip(width=WIDTH, half_height=HALF, thickness=T, notch=0.0, h=2e-3, fine=2e-3,
                            kinematics="plane_stress")
    free = sp.free.copy()
    free[:, :, 0] = 0.0
    return replace(sp, free=free)


def test_relaxation_on_the_fe_sheet_is_the_prony_closed_form():
    lam = 1.3
    delta = (lam - 1) * 2 * HALF
    dts = step_and_hold(n=60, dt=0.5)
    r = run(Solver(held_width_sheet(), "elastic"), material(), np.full(len(dts), delta), dts=dts)
    closed = closed_form_relaxation(lam, dts) * WIDTH * T
    assert np.max(np.abs(r["force"] / closed - 1)) <= 1e-5


def test_the_axisymmetric_and_plane_strain_builds_relax_alike():
    """Uniaxial strain (lateral stretches held at 1) two ways: the axisymmetric cylinder with its side held, and the
    plane-strain sheet with its width held; nominal stress per unit reference area."""
    mat = material(kappa=4 * C1)  # a small bulk modulus, so that the relaxing isochoric part is a fifth of the stress
    dts = step_and_hold(n=30, dt=1.0)
    stretch = 1.2
    cyl = stretched_cylinder(radius=1e-3, height=2e-3, nr=2, nz=2, confined=True)
    ra = run(Solver(cyl, "elastic"), mat, np.full(len(dts), (stretch - 1) * 2e-3), dts=dts)
    sheet = edge_notched_strip(width=4e-3, half_height=2e-3, thickness=1.0, notch=0.0, h=0.5e-3, fine=1e-3)
    free = sheet.free.copy()
    free[:, :, 0] = 0.0
    sheet = replace(sheet, free=free)
    rp = run(Solver(sheet, "elastic"), mat, np.full(len(dts), (stretch - 1) * 4e-3), dts=dts)
    axi = ra["force"] / (np.pi * 1e-3 ** 2)
    plane = rp["force"] / 4e-3
    assert np.max(np.abs(axi / plane - 1)) <= 1e-9
    assert axi[0] / axi[-1] > 1.15  # the check is not of a relaxed state (1.21 measured)


def test_the_branches_are_smooth_at_the_undeformed_state():
    """Every build's tangent exists at the identity with the branches on (no eigenvalues are taken)."""
    from mime_softtissue.engine.kinematics import axisymmetric as axi
    from mime_softtissue.engine.kinematics import plane_strain as ps

    mat = material(kappa=10.0 * C1)
    vis = law.ViscousStep(law.initial_strains((4,), mat), jnp.asarray(0.1))
    ue = jnp.zeros((4, 2))
    dq = jnp.zeros(4)
    for stress in (False, True):
        assert np.isfinite(np.asarray(ps.element_hessian(ue, dq, 1e-3, 1e-3, mat, stress, vis))).all()
    assert np.isfinite(np.asarray(axi.element_hessian(ue, dq, 0.0, 1e-3, 1e-3, mat, vis))).all()


def test_a_3d_cauchy_green_tensor_reads_as_the_plane_builds():
    """The law takes any build's C: a 3-D F with F33 = 1 gives the plane-strain energy."""
    from mime_softtissue.engine.kinematics import plane_strain as ps

    mat = material()
    F2 = jnp.array([[1.1, 0.2], [-0.05, 0.9]])
    F3 = jnp.eye(3).at[:2, :2].set(F2)
    c3 = law.voigt(F3.T @ F3)
    gamma = 0.01 * jnp.arange(12.0).reshape(2, 6)
    a = law.viscous_energies(c3, gamma, mat, 0.3)
    b = law.viscous_energies(ps.cauchy_green(F2), gamma, mat, 0.3)
    assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-14, atol=0)


def balance(r):
    d = np.concatenate([[0.0], r["delta"]])
    f = np.concatenate([[0.0], r["force"]])
    work = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(d))])[1:]
    return work, work - r["energy"] - r["crack_energy"] - np.cumsum(r["branch_dissipation"])


@pytest.fixture(scope="module")
def notched_visco_run():
    """A coarse notched sheet (h 0.5 mm, l 1 mm, notch 12 mm) at the clot tests' grip rate, to just past its peak."""
    sp = edge_notched_strip(width=WIDTH, half_height=HALF, thickness=T, notch=12e-3, h=0.5e-3, fine=3e-3,
                            kinematics="plane_stress")
    s = Solver(sp, "phase_field", tol_d=1e-5, max_staggered=300, newton_regularise=True, rate=0.1e-3)
    return run(s, material(gc=2.0), np.linspace(0, 8e-3, 81)[1:], stop_below=0.6, after_peak=2)


def test_energy_balance_with_the_branches_holds_to_the_line_up_to_the_peak(notched_visco_run):
    work, resid = balance(notched_visco_run)
    peak = int(np.argmax(notched_visco_run["force"]))
    assert peak > 10
    assert np.max(np.abs(resid[:peak + 1]) / work[:peak + 1]) <= 0.02


def test_energy_balance_without_the_dissipation_fails(notched_visco_run):
    r = dict(notched_visco_run)
    peak = int(np.argmax(r["force"]))
    work, _ = balance(r)
    assert np.sum(r["branch_dissipation"][:peak + 1]) > 0.02 * work[peak]
    r["branch_dissipation"] = 0.0 * r["branch_dissipation"]
    work, resid = balance(r)
    assert np.max(np.abs(resid[:peak + 1]) / work[:peak + 1]) > 0.02


def coarse_notched_sheet():
    return edge_notched_strip(width=WIDTH, half_height=HALF, thickness=T, notch=12e-3, h=1e-3, fine=3e-3,
                              kinematics="plane_stress")


SLOW_DELTAS = np.linspace(0, 6e-3, 31)[1:]


@pytest.fixture(scope="module")
def elastic_run():
    s = Solver(coarse_notched_sheet(), "phase_field", tol_d=1e-5, max_staggered=300, newton_regularise=True)
    return run(s, material()._replace(prony=None), SLOW_DELTAS)


def slow_run(long_term_scale=1.0):
    mat = material()
    mat = mat._replace(mu=mat.mu * long_term_scale, c2=mat.c2 * long_term_scale)
    slow = 1e-6 * (SLOW_DELTAS[1] - SLOW_DELTAS[0]) / max(TAU)  # each step takes 1e6 times the longest relaxation time
    s = Solver(coarse_notched_sheet(), "phase_field", tol_d=1e-5, max_staggered=300, newton_regularise=True, rate=slow)
    return run(s, mat, SLOW_DELTAS)


def test_at_a_vanishing_rate_the_viscous_run_is_the_elastic_run(elastic_run):
    rv = slow_run()
    assert np.max(np.abs(rv["force"] - elastic_run["force"])) <= 1e-4 * np.max(elastic_run["force"])


def test_the_instantaneous_modulus_as_long_term_fails_the_slow_rate_control(elastic_run):
    rv = slow_run(long_term_scale=1 + sum(BETA))
    assert np.max(np.abs(rv["force"] - elastic_run["force"])) > 1e-4 * np.max(elastic_run["force"])


def test_the_shipped_relaxation_data_is_the_packaged_file():
    """The calibration's data (doi 10.18738/T8/NA0WXY, decimated) ship with the package, byte for byte; each sample
    records the sha256 of the raw CSV it was decimated from."""
    import hashlib
    from importlib import resources

    raw = resources.files("mime_softtissue.studies").joinpath("data/rausch_visco_na0wxy.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "b5749d6d154df43fd323db4af30a675169f79c9a422841b81329f0412b8d50ef"
    import json

    samples = json.loads(raw)["samples"]
    assert sorted(samples) == ["1", "2", "4", "5", "6", "7"]
    assert all(len(s["raw_sha256"]) == 64 for s in samples.values())
