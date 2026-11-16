"""The material law: a free-energy density of a compressible neo-Hookean solid, strain-stiffened by Yeoh's terms, with
optional fibre families, written in the deformation's invariants so that it does not know the dimension of the build that
feeds it (plane strain, incompressible plane stress, axisymmetric; a 3-D kinematics supplies the same invariants).

    psi(J, I1, I4) = g(d) psi_plus + psi_minus,                        g(d) = (1 - d)^2 + k_res
    psi_iso   = (mu / 2) x + c2 x^2 + c3 x^3,                          x = J^(-2/3) I1 - 3 >= 0
    psi_vol   = (kappa / 4) (J^2 - 1 - 2 ln J)
    psi_fibre = sum over families of (k1 / 2 k2) (exp(k2 <I4 - 1>^2) - 1)
    psi_plus  = psi_iso + psi_fibre + [J >= 1] psi_vol,                psi_minus = [J < 1] psi_vol

Damage degrades the isochoric, fibre and dilatational parts, never volumetric compression. With c2 = c3 = 0 the law is
neo-Hookean; the Yeoh terms are polynomials in x, so every derivative is smooth at the undeformed state (an Ogden law's
are not). No MIME import.

Gent's stiffening (optional, ``Material.gent_jm``): the neo-Hookean term (mu / 2) x is replaced by

    psi_gent = -(mu Jm / 2) ln(1 - x / Jm)          for x <= x* = Jm (1 - GENT_CUT),
    psi_gent(x*) + psi_gent'(x*) (x - x*)            beyond (continued linearly: C1, finite everywhere),

the same shear modulus mu at small strain, stiffening without bound as x nears Jm. The continuation keeps the energy
finite, so a damaged band can still open past the limit (k_res times a locking law would hold a crack shut), and a
Newton trial step past the limit reads a finite energy; an undamaged point reaches it only at a stress 1 / GENT_CUT
times the neo-Hookean one at the same x. Invariant-based like the rest (no eigenvalues); Yeoh's c2, c3 still add."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


class Fibre(NamedTuple):
    """Fibre families at +-``angle`` (radians, in the reference plane) to the x axis, taking no compression."""

    k1: Any
    k2: Any
    angle: Any


class Material(NamedTuple):
    """The numbers a step reads (a pytree: derivatives reach every field). ``gc`` and ``l`` are the fracture
    model's toughness and length scale; ``stretch_limit`` is read only by the element-deletion control. ``c2``, ``c3``:
    Yeoh's stiffening (Pa; C1 = mu / 2). ``gent_jm``: Gent's limit on x (None: no Gent term)."""

    mu: Any
    kappa: Any
    gc: Any
    l: Any
    k_res: Any
    stretch_limit: Any
    fibre: Fibre | None = None
    c2: Any = 0.0
    c3: Any = 0.0
    gent_jm: Any = None


#: Where Gent's logarithm hands over to its linear continuation: 1 - x* / Jm (the stress there is 1 / GENT_CUT times
#: the neo-Hookean one at the same x).
GENT_CUT = 1e-3


def gent_energy(x: jax.Array, mu: Any, jm: Any) -> jax.Array:
    """Gent's isochoric energy at ``x`` = J^(-2/3) I1 - 3, continued linearly past x* = jm (1 - GENT_CUT)."""
    x_star = jm * (1.0 - GENT_CUT)
    inside = -0.5 * mu * jm * jnp.log1p(-jnp.minimum(x, x_star) / jm)
    return inside + (0.5 * mu / GENT_CUT) * jnp.maximum(x - x_star, 0.0)


def shear_and_bulk(young: Any, poisson: Any) -> tuple[Any, Any]:
    """Shear and bulk moduli (3-D) from Young's modulus and Poisson's ratio."""
    return young / (2 * (1 + poisson)), young / (3 * (1 - 2 * poisson))


def energy_parts(J: jax.Array, I1: jax.Array, fibre_invariants: tuple[jax.Array, ...],
                 mat: Material) -> tuple[jax.Array, jax.Array]:
    """(psi_plus, psi_minus) at the invariants ``J``, ``I1`` and one ``I4`` a fibre family; infinite where J <= 0."""
    ok = J > 0
    Js = jnp.where(ok, J, 1.0)
    x = Js ** (-2.0 / 3.0) * I1 - 3.0
    base = 0.5 * mat.mu * x if mat.gent_jm is None else gent_energy(x, mat.mu, mat.gent_jm)
    iso = base + mat.c2 * x ** 2 + mat.c3 * x ** 3
    vol = 0.25 * mat.kappa * (Js ** 2 - 1.0 - 2.0 * jnp.log(Js))
    fibre = jnp.zeros((), dtype=J.dtype)
    if mat.fibre is not None:
        for I4 in fibre_invariants:
            stretch = jnp.maximum(I4 - 1.0, 0.0)
            fibre = fibre + mat.fibre.k1 / (2 * mat.fibre.k2) * (jnp.exp(mat.fibre.k2 * stretch ** 2) - 1.0)
    plus = iso + fibre + jnp.where(Js >= 1.0, vol, 0.0)
    minus = jnp.where(Js < 1.0, vol, 0.0)
    bad = jnp.asarray(jnp.inf, dtype=J.dtype)
    return jnp.where(ok, plus, bad), jnp.where(ok, minus, bad)


def degradation(d: Any, k_res: Any) -> Any:
    """g(d), the share of the undamaged stiffness left at damage ``d``."""
    return (1.0 - d) ** 2 + k_res
