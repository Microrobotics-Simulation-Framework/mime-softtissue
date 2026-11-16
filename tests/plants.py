"""Planted errors for the controls: each is a context manager that puts one defect into the engine for the duration
of a ``with`` block and removes it after. They live in tests, not in the engine (the prototype carried its plants as
switches in production code: ``stop_gradient_*`` fields and an environment variable inside a node).

A plant patches the name where the engine *uses* it, and must be applied before a ``Solver`` is built (its step is
jitted at construction).

A plant is a control, so it must be caught EVERY run: measure how often with MIME's `.github/probe/detection_rate.py`.
``PLANT_VARIABLE`` lets that probe plant ``PLANTED_FROM_THE_ENVIRONMENT`` into any test from outside (tests/conftest.py)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np

from mime_softtissue.engine import material as law
from mime_softtissue.engine.fracture import phase_field
from mime_softtissue.engine.kinematics import plane_strain as ps


@contextmanager
def patched(module, name: str, new) -> Iterator[None]:
    old = getattr(module, name)
    setattr(module, name, new)
    try:
        yield
    finally:
        setattr(module, name, old)


def I1_from_diagonal_only() -> Iterator[None]:
    """The first invariant read from the diagonal of F alone: not objective (a rotation changes it)."""
    def invariants(F, mat, stress=False):
        J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
        return J, F[0, 0] ** 2 + F[1, 1] ** 2 + 1.0, ()
    return patched(ps, "invariants", invariants)


def volumetric_term_not_stationary() -> Iterator[None]:
    """psi_vol with ``ln J`` where ``2 ln J`` belongs: zero at the identity, not stationary there."""
    real = law.energy_parts

    def energy_parts(J, I1, fibre_invariants, mat):
        plus, minus = real(J, I1, fibre_invariants, mat)
        Js = jnp.where(J > 0, J, 1.0)
        wrong = 0.25 * mat.kappa * jnp.log(Js)
        return jnp.where(J >= 1.0, plus + wrong, plus), jnp.where(J < 1.0, minus + wrong, minus)
    return patched(ps.law, "energy_parts", energy_parts)


def energy_offset_at_identity() -> Iterator[None]:
    """A constant added to the energy density: not zero at the identity."""
    real = law.energy_parts

    def energy_parts(J, I1, fibre_invariants, mat):
        plus, minus = real(J, I1, fibre_invariants, mat)
        return plus + 1e-3 * mat.mu, minus
    return patched(ps.law, "energy_parts", energy_parts)


def hessian_row_scaled() -> Iterator[None]:
    """The element Hessian's first row halved (an assembly slip): not symmetric, and no longer the derivative of the
    gradient. (A ``stop_gradient`` on a term does NOT do this: forward-over-reverse drops it consistently and the
    Hessian stays symmetric; the first plant tried, caught 0 of 10.)"""
    real = ps.element_hessian

    def element_hessian(ue, dq, dx, dy, mat, stress=False):
        return real(ue, dq, dx, dy, mat, stress).at[0, :].mul(0.5)
    return patched(ps, "element_hessian", element_hessian)


def assembly_upper_block_transposed() -> Iterator[None]:
    """The coupling to the next column stored transposed."""
    from mime_softtissue.engine.assembly import Assembly
    real = Assembly.blocks

    def blocks(self, ke):
        diag, upper = real(self, ke)
        return diag, jnp.swapaxes(upper, 1, 2)
    return patched(Assembly, "blocks", blocks)


def damage_not_clipped() -> Iterator[None]:
    """The damage solve without its clip to [0, 1]."""
    real_clip = jnp.clip

    class NoClip:
        def __getattr__(self, name):
            if name == "clip":
                return lambda x, lo, hi: x
            return getattr(jnp, name)
    return patched(phase_field, "jnp", NoClip())


def history_not_kept() -> Iterator[None]:
    """The history set to the current tension energy: damage can heal."""
    return patched(phase_field, "updated_history", lambda previous, plus: plus)


def gauss_weight(error: float) -> Iterator[None]:
    """One Gauss point of four carries weight (1 + error) in the element energy: a quadrature error."""
    weights = jnp.asarray([1.0 + error, 1.0, 1.0, 1.0])
    real = ps.element_strain_energy

    def element_strain_energy(ue, dq, dx, dy, mat, stress=False):
        F = ps.deformation_gradients(ue, dx, dy)
        plus, minus = jax.vmap(lambda f: ps.energy_parts(f, mat, stress))(F)
        g = law.degradation(dq, mat.k_res)
        return jnp.sum(weights * (g * plus + minus)) * dx * dy / 4.0
    from mime_softtissue.engine import solver as solver_module
    return patched(solver_module, "element_strain_energy", element_strain_energy)


def shear_modulus_scaled(factor: float) -> Iterator[None]:
    """The shear modulus off by ``factor`` in the material read from a record."""
    import reproduction_cases as rc
    from contextlib import ExitStack
    real = law.shear_and_bulk

    @contextmanager
    def both() -> Iterator[None]:
        with ExitStack() as stack:
            from mime_softtissue import _material_record as record
            wrong = lambda young, poisson: (real(young, poisson)[0] * factor, real(young, poisson)[1])
            stack.enter_context(patched(rc, "shear_and_bulk", wrong))
            stack.enter_context(patched(record, "shear_and_bulk", wrong))
            yield
    return both()


def gc_correction_left_out() -> Iterator[None]:
    """The input toughness without the phase field's grid-bias correction (in the case builders and in the entries)."""
    import reproduction_cases as rc
    from mime_softtissue import _material_record as record
    from contextlib import ExitStack

    @contextmanager
    def both() -> Iterator[None]:
        with ExitStack() as stack:
            stack.enter_context(patched(rc, "discretisation_toughness", lambda gc, h, l: gc))
            stack.enter_context(patched(record, "discretisation_toughness", lambda gc, h, l: gc))
            yield
    return both()


#: Set to anything to have tests/conftest.py plant the error in every test of the run, for MIME's detection_rate.py.
PLANT_VARIABLE = "MIME_PLANT_ERROR"
#: The plant that goes in: a first invariant that is not objective, which exactly the rotation-invariance properties catch.
PLANTED_FROM_THE_ENVIRONMENT = I1_from_diagonal_only
