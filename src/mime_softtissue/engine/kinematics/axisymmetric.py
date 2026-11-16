"""Axisymmetric kinematics on bilinear (Q1) elements in the (R, Z) half-plane, 2 x 2 Gauss points, no twist.

The grid's x is the radius R (the axis at x = 0 when the specimen reaches it), its y the axial Z; a node carries
(u_R, u_Z). At a Gauss point the deformation gradient in the cylindrical frame is

    F = [[1 + du_R/dR, du_R/dZ, 0], [du_Z/dR, 1 + du_Z/dZ, 0], [0, 0, F_tt]],   F_tt = 1 + u_R / R  (the hoop stretch)

and the element's energy is sum over Gauss points of 2 pi R_g (dR dZ / 4) psi. Gauss points lie inside the element,
so R_g > 0 even in the column on the axis (R_g = dR (1 - 1/sqrt 3) / 2 there), where u_R = 0 is prescribed and
u_R / R stays finite. Fibre families lie in the (R, Z) plane at +-angle to R (as plane strain's lie in (x, y)).

Same functions as ``plane_strain`` with the element's left radius ``x0`` added. The pieces a planted error replaces
(``hoop_stretch``, ``gauss_weights``) are module functions looked up at trace time."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .. import material as law
from .plane_strain import DN_CENTRE, N_GAUSS, _GAUSS, gauss_gradients
from ..material import Material

XI_GAUSS = np.asarray(_GAUSS)[:, 0]  # the Gauss points' reference radial coordinate
N_CENTRE = np.full(4, 0.25)


def gauss_radii(x0: jax.Array, dx: jax.Array) -> jax.Array:
    """R at the four Gauss points of an element whose left edge is at ``x0``: (4,)."""
    return x0 + 0.5 * dx * (1.0 + jnp.asarray(XI_GAUSS))


def gauss_weights(x0: jax.Array, dx: jax.Array, dy: jax.Array) -> jax.Array:
    """The quadrature weights 2 pi R_g dR dZ / 4: (4,)."""
    return 2.0 * jnp.pi * gauss_radii(x0, dx) * dx * dy / 4.0


def hoop_stretch(ue: jax.Array, n: jax.Array, radius: jax.Array) -> jax.Array:
    """F_tt = 1 + u_R / R at a point with shape-function values ``n`` (4,) and radius ``radius``."""
    return 1.0 + (n @ ue[:, 0]) / radius


def invariants(F2: jax.Array, Ftt: jax.Array, mat: Material) -> tuple[jax.Array, jax.Array, tuple[jax.Array, ...]]:
    """J, I1 and the fibre invariants from the in-plane block ``F2`` (2 x 2) and the hoop stretch ``Ftt``."""
    J = (F2[0, 0] * F2[1, 1] - F2[0, 1] * F2[1, 0]) * Ftt
    C = F2.T @ F2
    I1 = C[0, 0] + C[1, 1] + Ftt * Ftt
    fibres: list[jax.Array] = []
    if mat.fibre is not None:
        for sign in (1.0, -1.0):
            a = jnp.stack([jnp.cos(sign * mat.fibre.angle), jnp.sin(sign * mat.fibre.angle)])
            fibres.append(a @ C @ a)
    return J, I1, tuple(fibres)


def deformation(ue: jax.Array, x0: jax.Array, dx: jax.Array, dy: jax.Array) -> tuple[jax.Array, jax.Array]:
    """(F2, F_tt) at the four Gauss points: (4, 2, 2) and (4,)."""
    G = gauss_gradients(dx, dy)
    F2 = jnp.eye(2, dtype=ue.dtype)[None] + jnp.einsum("ai,gaj->gij", ue, G)
    R = gauss_radii(x0, dx)
    Ftt = jax.vmap(lambda n, r: hoop_stretch(ue, n, r))(jnp.asarray(N_GAUSS, dtype=ue.dtype), R)
    return F2, Ftt


def energy_parts(F2: jax.Array, Ftt: jax.Array, mat: Material) -> tuple[jax.Array, jax.Array]:
    return law.energy_parts(*invariants(F2, Ftt, mat), mat)


def cauchy_green(F2: jax.Array, Ftt: jax.Array) -> jax.Array:
    """The right Cauchy-Green tensor in Voigt order (6,): the in-plane block's and the hoop's C_tt = F_tt^2."""
    C = F2.T @ F2
    zero = jnp.zeros_like(Ftt)
    return jnp.stack([C[0, 0], C[1, 1], Ftt * Ftt, C[0, 1], zero, zero])


def viscous_parts(F2: jax.Array, Ftt: jax.Array, mat: Material,
                  visco: law.ViscousStep) -> tuple[jax.Array, jax.Array]:
    """The branches' (incremental potential, stored energy) at the four Gauss points."""
    return law.viscous_energies(jax.vmap(cauchy_green)(F2, Ftt), visco.gamma, mat, visco.dt)


def element_strain_energy(ue: jax.Array, dq: jax.Array, x0: jax.Array, dx: jax.Array, dy: jax.Array,
                          mat: Material, visco: law.ViscousStep | None = None) -> jax.Array:
    """Degraded strain energy of one ring element (the whole ring, 2 pi included): ``ue`` (4, 2), ``dq`` (4,).
    ``visco``: the viscous branches' step, added to psi_plus as in ``plane_strain``."""
    F2, Ftt = deformation(ue, x0, dx, dy)
    plus, minus = jax.vmap(lambda f, t: energy_parts(f, t, mat))(F2, Ftt)
    if visco is not None:
        plus = plus + viscous_parts(F2, Ftt, mat, visco)[0]
    g = law.degradation(dq, mat.k_res)
    return jnp.sum(gauss_weights(x0, dx, dy) * (g * plus + minus))


def element_hessian(ue: jax.Array, dq: jax.Array, x0: jax.Array, dx: jax.Array, dy: jax.Array,
                    mat: Material, visco: law.ViscousStep | None = None) -> jax.Array:
    return jax.hessian(lambda v: element_strain_energy(v.reshape(4, 2), dq, x0, dx, dy, mat, visco))(ue.reshape(8))


def element_plus(ue: jax.Array, x0: jax.Array, dx: jax.Array, dy: jax.Array, mat: Material,
                 visco: law.ViscousStep | None = None) -> jax.Array:
    """psi_plus at the four Gauss points (the history field's source); with ``visco`` the branches' stored energy."""
    F2, Ftt = deformation(ue, x0, dx, dy)
    plus = jax.vmap(lambda f, t: energy_parts(f, t, mat)[0])(F2, Ftt)
    if visco is not None:
        plus = plus + viscous_parts(F2, Ftt, mat, visco)[1]
    return plus


def element_cauchy_green(ue: jax.Array, x0: jax.Array, dx: jax.Array, dy: jax.Array) -> jax.Array:
    """C in Voigt order at the four Gauss points: (4, 6)."""
    return jax.vmap(cauchy_green)(*deformation(ue, x0, dx, dy))


def element_stored(ue: jax.Array, dq: jax.Array, x0: jax.Array, dx: jax.Array, dy: jax.Array,
                   mat: Material, visco: law.ViscousStep | None = None) -> jax.Array:
    """The stored energy at each Gauss point times its weight: (4,) (a readout; with ``visco`` at ``dt`` = 0 the
    branches' stored energy at its strains)."""
    F2, Ftt = deformation(ue, x0, dx, dy)
    plus, minus = jax.vmap(lambda f, t: energy_parts(f, t, mat))(F2, Ftt)
    if visco is not None:
        plus = plus + viscous_parts(F2, Ftt, mat, visco)[1]
    return gauss_weights(x0, dx, dy) * (law.degradation(dq, mat.k_res) * plus + minus)


def centre_stretch(ue: jax.Array, x0: jax.Array, dx: jax.Array, dy: jax.Array) -> jax.Array:
    """The largest principal stretch at the element's centre: the in-plane block's or the hoop's."""
    G = jnp.asarray(DN_CENTRE) * jnp.stack([2.0 / dx, 2.0 / dy])
    F = jnp.eye(2, dtype=ue.dtype) + ue.T @ G
    C = F.T @ F
    tr, det = C[0, 0] + C[1, 1], C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    top = 0.5 * tr + jnp.sqrt(jnp.maximum(0.25 * tr * tr - det, 0.0))
    Ftt = hoop_stretch(ue, jnp.asarray(N_CENTRE, dtype=ue.dtype), x0 + 0.5 * dx)
    return jnp.maximum(jnp.sqrt(jnp.maximum(top, 1.0)), Ftt)


def nodal_ring_area(x: np.ndarray) -> np.ndarray:
    """The annulus each node column owns on a horizontal face: pi ((R + dR_right/2)^2 - (R - dR_left/2)^2), the axis
    node's disc pi (dR/2)^2 (the contact penalty's nodal weight)."""
    d = np.diff(x)
    left = np.concatenate([[0.0], d / 2])
    right = np.concatenate([d / 2, [0.0]])
    outer = x + right
    inner = np.maximum(x - left, 0.0)
    return np.pi * (outer ** 2 - inner ** 2)
