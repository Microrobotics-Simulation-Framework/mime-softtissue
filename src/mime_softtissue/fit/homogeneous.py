"""The plugin's material law (``engine.material``) at the homogeneous deformations of two lab tests, in 3-D:

- a rheometer's oscillatory shear: simple shear ``F = I + gamma e1 (x) e2`` (gap fixed), the shear stress
  ``d psi / d gamma``, and the storage modulus as the first harmonic of the stress over one cycle of amplitude
  ``gamma0``; on parallel plates the strain grows with radius and the instrument reads the torque, so the apparent
  stress is ``4 int_0^1 sigma(gamma_R s) s^2 ds`` (cone-plate: the stress itself);
- unconfined compression between frictionless platens: ``F = diag(l1, l2, l3)``, ``l3 = 1 - strain``, the lateral
  stretches solved for zero lateral stress, the nominal (first Piola) stress ``-d psi / d l3``, compressive positive.

Fibre families lie in the 1-2 plane at +-``angle`` to axis 1: in shear the plane of shear, in compression the plane
across the load. The undamaged energy is used (g = 1; the solver's undamaged material carries 1 + k_res on psi_plus,
1e-6 relative). Pure JAX, differentiable in every material number."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..engine import material as law
from ..engine.material import Material

_CYCLE = 4096  # one whole oscillation by the trapezoid rule, only to check the quarter-cycle rule (the odd stress
# has a kink in its second derivative at zero strain, so the trapezoid converges algebraically there)
_xq, _wq = np.polynomial.legendre.leggauss(16)
_QT, _QW = 0.25 * np.pi * (_xq + 1.0), 0.25 * np.pi * _wq  # Gauss-Legendre on [0, pi/2]
_x, _w = np.polynomial.legendre.leggauss(16)
_GL_S, _GL_W = 0.5 * (_x + 1.0), 0.5 * _w  # Gauss-Legendre on [0, 1]


def invariants_3d(F: jax.Array, mat: Material) -> tuple[jax.Array, jax.Array, tuple[jax.Array, ...]]:
    """J, I1 and the two fibre invariants of a 3 x 3 deformation gradient (fibres in the 1-2 plane)."""
    J = jnp.linalg.det(F)
    C = F.T @ F
    I1 = jnp.trace(C)
    fibres: list[jax.Array] = []
    if mat.fibre is not None:
        for sign in (1.0, -1.0):
            ang = sign * mat.fibre.angle
            a = jnp.stack([jnp.cos(ang), jnp.sin(ang), jnp.zeros_like(ang)])
            fibres.append(a @ C @ a)
    return J, I1, tuple(fibres)


def psi(F: jax.Array, mat: Material) -> jax.Array:
    """The undamaged free energy per unit reference volume at ``F``."""
    plus, minus = law.energy_parts(*invariants_3d(F, mat), mat)
    return plus + minus


def shear_stress(gamma: jax.Array, mat: Material, energy_law=None) -> jax.Array:
    """The shear stress of simple shear of amount ``gamma`` (equal to the nominal shear stress: J = 1)."""
    law_psi = energy_law or psi

    def energy(g):
        F = jnp.eye(3, dtype=jnp.result_type(g, float)).at[0, 1].set(g)
        return law_psi(F, mat)
    return jax.grad(energy)(gamma)


def _apparent_stress(gamma_r: jax.Array, mat: Material, geometry: str, energy_law=None) -> jax.Array:
    if geometry == "cone_plate":
        return shear_stress(gamma_r, mat, energy_law)
    if geometry == "parallel_plate":
        s = jnp.asarray(_GL_S, dtype=jnp.result_type(gamma_r, float))
        w = jnp.asarray(_GL_W, dtype=s.dtype)
        sig = jax.vmap(lambda si: shear_stress(gamma_r * si, mat, energy_law))(s)
        return 4.0 * jnp.sum(w * sig * s ** 2)
    raise ValueError("geometry is 'cone_plate' or 'parallel_plate'")


def storage_modulus(gamma0: jax.Array, mat: Material, geometry: str = "cone_plate", energy_law=None,
                    full_cycle: bool = False) -> jax.Array:
    """G' at strain amplitude ``gamma0``: (1 / pi gamma0) int_0^2pi sigma_app(gamma0 sin t) sin t dt. The stress is
    odd in gamma for an isotropic law and for the +-angle fibre pair (shearing back swaps the two families), so the
    integral is four times the quarter cycle's (Gauss-Legendre); ``full_cycle`` is the whole cycle by the trapezoid
    rule (the check of that symmetry)."""
    dt = jnp.result_type(gamma0, float)
    if full_cycle:
        st = jnp.sin(2.0 * jnp.pi * jnp.arange(_CYCLE) / _CYCLE).astype(dt)
        sig = jax.vmap(lambda s: _apparent_stress(gamma0 * s, mat, geometry, energy_law))(st)
        return 2.0 * jnp.sum(sig * st) / (_CYCLE * gamma0)
    st = jnp.sin(jnp.asarray(_QT, dtype=dt))
    sig = jax.vmap(lambda s: _apparent_stress(gamma0 * s, mat, geometry, energy_law))(st)
    return 4.0 * jnp.sum(jnp.asarray(_QW, dtype=dt) * sig * st) / (jnp.pi * gamma0)


def _lateral_guess(mat: Material, l3: jax.Array) -> jax.Array:
    nu = (3.0 * mat.kappa - 2.0 * mat.mu) / (2.0 * (3.0 * mat.kappa + mat.mu))
    return l3 ** (-nu)


def lateral_stretches(l3: jax.Array, mat: Material, iterations: int = 25, energy_law=None) -> tuple[jax.Array, jax.Array]:
    """(l1, l2) at axial stretch ``l3`` with zero lateral stress: Newton in (ln l1, ln l2), steps capped at 0.25."""
    law_psi = energy_law or psi

    def energy(x):
        return law_psi(jnp.diag(jnp.concatenate([jnp.exp(x), l3[None]])), mat)

    g0 = jnp.log(_lateral_guess(mat, l3))
    x0 = jnp.stack([g0, g0])

    def body(_, x):
        gr = jax.grad(energy)(x)
        H = jax.hessian(energy)(x)
        dx = -jnp.linalg.solve(H, gr)
        dx = dx * jnp.minimum(1.0, 0.25 / jnp.maximum(jnp.max(jnp.abs(dx)), 0.25))
        return x + dx

    # the loop is not differentiated; one Newton step from its answer is, which carries the implicit derivative
    # (the residual is zero there, so the step's own derivative is -H^-1 times the residual's)
    xs = jax.lax.stop_gradient(jax.lax.fori_loop(0, iterations, body, jax.lax.stop_gradient(x0)))
    x = xs - jnp.linalg.solve(jax.hessian(energy)(xs), jax.grad(energy)(xs))
    return jnp.exp(x[0]), jnp.exp(x[1])


def compression_stress(strain: jax.Array, mat: Material, energy_law=None) -> jax.Array:
    """Nominal compressive stress (force over the undeformed area, compressive positive) at engineering strain
    ``strain`` (compressive positive) in unconfined compression."""
    l3 = 1.0 - strain
    l1, l2 = lateral_stretches(l3, mat, energy_law=energy_law)
    law_psi = energy_law or psi
    return -jax.grad(lambda c: law_psi(jnp.diag(jnp.stack([l1, l2, c])), mat))(l3)


def lateral_residual(strain: jax.Array, mat: Material, energy_law=None) -> jax.Array:
    """The lateral nominal stresses left by the solve, relative to the axial one (a check on the solve)."""
    l3 = 1.0 - strain
    l1, l2 = lateral_stretches(l3, mat, energy_law=energy_law)
    law_psi = energy_law or psi
    g = jax.grad(lambda v: law_psi(jnp.diag(v), mat))(jnp.stack([l1, l2, l3]))
    return jnp.max(jnp.abs(g[:2])) / (jnp.abs(g[2]) + 1e-300)


def material_from(params: dict[str, Any], angle: Any = None) -> Material:
    """A Material with the fit's parameters (``mu``, ``kappa``, optionally ``k1``, ``k2``); the fracture numbers are
    irrelevant to these tests and set to placeholders."""
    from ..engine.material import Fibre
    fibre = None
    if "k1" in params:
        fibre = Fibre(k1=params["k1"], k2=params["k2"], angle=jnp.asarray(np.pi / 4 if angle is None else angle))
    one = jnp.ones((), dtype=jnp.result_type(params["mu"], float))
    return Material(mu=params["mu"], kappa=params["kappa"], gc=one, l=one, k_res=0.0 * one, stretch_limit=one,
                    fibre=fibre)
