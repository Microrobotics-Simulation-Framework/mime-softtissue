"""The viscoelastic law (``engine.material``'s Prony branches) at one material point of a pure-shear sheet: the width
held (stretch 1), the height stretched by ``stretch``, the thickness free (incompressible plane stress, so its stretch is
``1 / stretch``), through the engine's own functions (``kinematics.plane_strain.cauchy_green`` and ``energy_parts`` with
``stress=True``). Each step is the solver's: the nominal stress is the step's incremental potential's derivative, and
the internal strains are then updated in closed form. Undamaged (g = 1). Pure JAX, differentiable in every material
number; the calibration of ``studies.visco_clot`` and the closed-form checks run on it."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from ..engine import material as law
from ..engine.kinematics import plane_strain as ps
from ..engine.material import Material


class PointPath(NamedTuple):
    stress: jax.Array  # nominal stress (Pa) across the sheet's height, at the end of each step
    stored: jax.Array  # J/m^3, the stored energy (long-term and branches) at the end of each step
    dissipation: jax.Array  # J/m^3, the branches' dissipation over each step


def _sheet(stretch: jax.Array) -> jax.Array:
    one = jnp.ones_like(stretch)
    zero = jnp.zeros_like(stretch)
    return jnp.stack([jnp.stack([one, zero]), jnp.stack([zero, stretch])])


def _cauchy_green(stretch: jax.Array) -> jax.Array:
    return ps.cauchy_green(_sheet(stretch), stress=True)


def _long_term(stretch: jax.Array, mat: Material) -> jax.Array:
    plus, minus = ps.energy_parts(_sheet(stretch), mat, stress=True)
    return plus + minus


def pure_shear_path(stretch: Any, dt: Any, mat: Material) -> PointPath:
    """The material point along the stretches ``stretch`` (N,) reached in the times ``dt`` (N,) (s), from the
    undeformed, relaxed state."""
    stretch = jnp.asarray(stretch)
    dt = jnp.asarray(dt, dtype=stretch.dtype)
    if mat.prony is None:
        stress = jax.vmap(jax.grad(lambda s: _long_term(s, mat)))(stretch)
        stored = jax.vmap(lambda s: _long_term(s, mat))(stretch)
        return PointPath(stress, stored, jnp.zeros_like(stretch))
    gamma0 = law.initial_strains((), mat, stretch.dtype)

    def step(carry, inputs):
        gamma, c_prev = carry
        s, h = inputs

        def potential(q):
            return _long_term(q, mat) + law.viscous_energies(_cauchy_green(q), gamma, mat, h)[0]

        stress = jax.grad(potential)(s)
        c = _cauchy_green(s)
        new = law.updated_strains(c, gamma, mat, h)
        stored = _long_term(s, mat) + law.viscous_energies(c, new, mat, 0.0)[1]
        diss = law.branch_dissipation(c_prev, c, gamma, new, mat)
        return (new, c), (stress, stored, diss)

    c0 = _cauchy_green(jnp.ones((), stretch.dtype))
    _, (stress, stored, diss) = jax.lax.scan(step, (gamma0, c0), (stretch, dt))
    return PointPath(stress, stored, diss)


def with_scale(mat: Material, scale: Any) -> Material:
    """``mat`` with its long-term isochoric constants (mu, c2, c3) times ``scale``: the Prony terms, being ratios, and
    the branch strain's shape (c2 / mu, c3 / mu) are unchanged."""
    return mat._replace(mu=mat.mu * scale, c2=mat.c2 * scale, c3=mat.c3 * scale)
