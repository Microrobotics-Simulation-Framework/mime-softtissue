"""The energy flowing into a crack tip's region, from a run's own fields: the domain J-integral at finite strain.

    J = integral over A of ( P_ij u_i,1 - W delta_1j ) q_,j dA

in the reference configuration (Eshelby's energy-momentum tensor), with ``W`` the degraded energy density, ``P`` its
first Piola stress and ``q`` 1 inside the radius ``r1`` of ``centre``, 0 outside ``r2``, linear between. Where the
damage does not vary along the crack's direction in the annulus (no damage there, or a straight wake), this is the
contour integral at the annulus: the energy released per unit crack advance of everything inside it. On the strip's
half model the crack faces and the ligament's symmetry add nothing, so the whole specimen's J is twice the half's. The
crack is taken to run along +x."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .. import material as law
from ..kinematics.plane_strain import energy_parts, gauss_gradients
from ..material import Material


def j_integral(solver: Any, u: Any, d: Any, mat: Material, centre: tuple[float, float], r1: float,
               r2: float, notch: Any = None) -> float:
    """J (J/m^2) of the whole specimen at the state ``(u, d)`` of ``solver``'s half model."""
    g = solver.grid
    X, Y = np.meshgrid(g.x, g.y, indexing="ij")
    q = np.clip((r2 - np.hypot(X - centre[0], Y - centre[1])) / (r2 - r1), 0.0, 1.0)
    dx, dy = solver.sizes(notch)
    dq = solver.damage_at_gauss(jnp.asarray(d))
    eye = jnp.eye(2)

    def density(f: Any, de: Any) -> Any:
        plus, minus = energy_parts(f, mat)
        return law.degradation(de, mat.k_res) * plus + minus

    def element(ue: Any, qe: Any, de: Any, ex: Any, ey: Any) -> Any:
        grads = gauss_gradients(ex, ey)  # (4 gauss, 4 nodes, 2)
        F = eye[None] + jnp.einsum("ai,gaj->gij", ue, grads)
        grad_q = jnp.einsum("a,gaj->gj", qe, grads)
        W = jax.vmap(density)(F, de)
        P = jax.vmap(jax.grad(density))(F, de)
        du_dx1 = F[:, :, 0] - eye[None, :, 0]
        integrand = jnp.einsum("gij,gi,gj->g", P, du_dx1, grad_q) - W * grad_q[:, 0]
        return jnp.sum(integrand) * ex * ey / 4.0

    I, J = solver.I, solver.J
    half = jnp.sum(jax.vmap(element)(jnp.asarray(u)[I, J], jnp.asarray(q)[I, J], dq, dx, dy))
    return 2.0 * float(half)


def tip(grid_x: np.ndarray, d_line: np.ndarray, level: float, start: float) -> float:
    """The largest x on the crack plane at which the nodal damage ``d_line`` reaches ``level`` (linear between nodes),
    or ``start`` (the notch's tip) where it reaches it nowhere ahead of it."""
    ahead = grid_x >= start - 1e-12
    above = np.where(ahead & (d_line >= level))[0]
    if len(above) == 0:
        return start
    k = above[-1]
    if k + 1 >= len(grid_x):
        return float(grid_x[k])
    d0, d1 = d_line[k], d_line[k + 1]
    return float(grid_x[k] + (grid_x[k + 1] - grid_x[k]) * (d0 - level) / (d0 - d1))
