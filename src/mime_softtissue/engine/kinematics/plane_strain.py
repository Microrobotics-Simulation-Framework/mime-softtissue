"""Plane-strain kinematics on bilinear (Q1) quadrilateral elements, 2 x 2 Gauss points: the shape functions, the
deformation gradient at a Gauss point, its invariants for the material law, and the element's energy.

``stress=True`` is the incompressible plane-stress build on the same elements (a thin sheet: F33 = 1 / det F, so J = 1
and I1 = C11 + C22 + det(C)^-1); the energy is then per unit reference thickness of the sheet, as plane strain's is per
unit length out of plane.

This module is one build of the solid's kinematics. An axisymmetric build and a 3-D build would sit beside it as
independent modules with the same functions (``invariants``, ``element_strain_energy``, ``element_plus``,
``centre_stretch``, ``gauss_gradients`` and the node/degree-of-freedom layout the assembly reads); the solver's
dimension-specific names are all imported here (``quasistatic``)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .. import material as law
from ..material import Material

# Q1 reference nodes in the order (i, j), (i+1, j), (i+1, j+1), (i, j+1).
_XI = np.array([-1.0, 1.0, 1.0, -1.0])
_ETA = np.array([-1.0, -1.0, 1.0, 1.0])
_G = 1.0 / np.sqrt(3.0)
_GAUSS = np.array([(-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G)])
COLUMN = np.array([0, 1, 1, 0])  # the node's column offset in its element
ROW = np.array([0, 0, 1, 1])


def _shape(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    n = (1 + _XI * xi) * (1 + _ETA * eta) / 4
    dn = np.stack([_XI * (1 + _ETA * eta) / 4, _ETA * (1 + _XI * xi) / 4], axis=1)  # d/dxi, d/deta
    return n, dn


N_GAUSS = np.stack([_shape(*p)[0] for p in _GAUSS])  # (4 gauss, 4 nodes)
DN_GAUSS = np.stack([_shape(*p)[1] for p in _GAUSS])  # (4 gauss, 4 nodes, 2)
DN_CENTRE = _shape(0.0, 0.0)[1]


def invariants(F: jax.Array, mat: Material, stress: bool = False) -> tuple[jax.Array, jax.Array, tuple[jax.Array, ...]]:
    """J, I1 and the fibre invariants of a 2 x 2 deformation gradient in plane strain (F33 = 1), or with ``stress`` in
    incompressible plane stress (F33 = 1 / det F; J is 1, or det F where det F <= 0 so that the law reads it inverted)."""
    area = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    C = F.T @ F
    # In plane stress the sheet's thickness stretch is 1 / area: J = 1 where the element is not inverted.
    safe = jnp.where(area > 0, area, 1.0)
    jac = jnp.where(area > 0, jnp.ones_like(area), area) if stress else area
    i1 = C[0, 0] + C[1, 1] + (1.0 / (safe * safe) if stress else 1.0)
    fibres: list[jax.Array] = []
    if mat.fibre is not None:
        for sign in (1.0, -1.0):
            a = jnp.stack([jnp.cos(sign * mat.fibre.angle), jnp.sin(sign * mat.fibre.angle)])
            fibres.append(a @ C @ a)
    return jac, i1, tuple(fibres)


def energy_parts(F: jax.Array, mat: Material, stress: bool = False) -> tuple[jax.Array, jax.Array]:
    """(psi_plus, psi_minus) at a 2 x 2 deformation gradient."""
    return law.energy_parts(*invariants(F, mat, stress), mat)


def gauss_gradients(dx: jax.Array, dy: jax.Array) -> jax.Array:
    """dN/dX at each Gauss point of a dx-by-dy element: (4 gauss, 4 nodes, 2)."""
    scale = jnp.stack([2.0 / dx, 2.0 / dy])
    return jnp.asarray(DN_GAUSS) * scale


def deformation_gradients(ue: jax.Array, dx: jax.Array, dy: jax.Array) -> jax.Array:
    """F at the four Gauss points of an element with nodal displacements ``ue`` (4, 2): (4, 2, 2)."""
    G = gauss_gradients(dx, dy)
    return jnp.eye(2, dtype=ue.dtype)[None] + jnp.einsum("ai,gaj->gij", ue, G)


def element_strain_energy(ue: jax.Array, dq: jax.Array, dx: jax.Array, dy: jax.Array, mat: Material,
                          stress: bool = False) -> jax.Array:
    """Degraded strain energy of one element (per unit thickness): ``ue`` (4, 2) nodal displacements, ``dq`` (4,)
    damage at the Gauss points."""
    F = deformation_gradients(ue, dx, dy)
    plus, minus = jax.vmap(lambda f: energy_parts(f, mat, stress))(F)
    g = law.degradation(dq, mat.k_res)
    return jnp.sum(g * plus + minus) * dx * dy / 4.0


def element_hessian(ue: jax.Array, dq: jax.Array, dx: jax.Array, dy: jax.Array, mat: Material,
                    stress: bool = False) -> jax.Array:
    """The element energy's Hessian in its eight nodal displacements (the tangent stiffness the Newton solve uses)."""
    return jax.hessian(lambda v: element_strain_energy(v.reshape(4, 2), dq, dx, dy, mat, stress))(ue.reshape(8))


def element_plus(ue: jax.Array, dx: jax.Array, dy: jax.Array, mat: Material, stress: bool = False) -> jax.Array:
    """psi_plus at the four Gauss points (the history field's source)."""
    F = deformation_gradients(ue, dx, dy)
    return jax.vmap(lambda f: energy_parts(f, mat, stress)[0])(F)


def centre_stretch(ue: jax.Array, dx: jax.Array, dy: jax.Array) -> jax.Array:
    """The largest principal stretch at the element's centre (plane strain; the out-of-plane stretch is 1)."""
    G = jnp.asarray(DN_CENTRE) * jnp.stack([2.0 / dx, 2.0 / dy])
    F = jnp.eye(2, dtype=ue.dtype) + ue.T @ G
    C = F.T @ F
    tr, det = C[0, 0] + C[1, 1], C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    top = 0.5 * tr + jnp.sqrt(jnp.maximum(0.25 * tr * tr - det, 0.0))
    return jnp.sqrt(jnp.maximum(top, 1.0))
