"""The AT2 phase field: surface energy ``Gc ( d^2 / 2l + l/2 |grad d|^2 )`` per unit reference area, nodal damage
``d`` in [0, 1] on the grid's Q1 nodes, irreversible through the history field ``H = max over the path of
psi_plus`` (Miehe, Welschinger, Hofacker 2010). The damage subproblem for a fixed history is linear."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..assembly import Assembly
from ..grid import Grid
from ..kinematics.plane_strain import N_GAUSS, gauss_gradients
from ..solvers import blocktri


def discretisation_toughness(gc: float, h: float, l: float) -> float:
    """The input toughness that makes the grid's phase field dissipate ``gc``: ``gc / (1 + h / 2l)`` (the AT2 grid
    bias, Bourdin, Francfort, Marigo 2008)."""
    return gc / (1.0 + h / (2.0 * l))


def solve_damage(history: jax.Array, gc, l, dx: jax.Array, dy: jax.Array, grid: Grid, assembly: Assembly,
                 I: jax.Array, J: jax.Array, weights: jax.Array | None = None, viscosity: float = 0.0,
                 d_prev: jax.Array | None = None) -> jax.Array:
    """The damage minimising sum (1-d)^2 H + Gc (d^2/2l + l/2 |grad d|^2) [+ eta/2 (d - d_prev)^2] for ``history``
    (n_el, 4). ``weights`` (n_el, 4): the Gauss weights (an axisymmetric build's 2 pi R dR dZ / 4); by default the
    plane element's dx dy / 4. ``viscosity``: eta, a viscous regularisation of the damage against the previous
    step's ``d_prev`` (nodal)."""
    N = jnp.asarray(N_GAUSS)
    G = jax.vmap(gauss_gradients)(dx, dy)  # (n_el, 4g, 4n, 2)
    if weights is not None or viscosity:
        wq = jnp.broadcast_to((dx * dy / 4.0)[:, None], history.shape) if weights is None else weights
        coef = wq * (2.0 * (history + gc / (2.0 * l)) + viscosity)
        mass = jnp.einsum("eg,ga,gb->eab", coef, N, N)
        stiff = gc * l * jnp.einsum("eg,egai,egbi->eab", wq, G, G)
        src = wq * 2.0 * history
        if viscosity:
            src = src + wq * viscosity * damage_at_gauss(d_prev, I, J)
        ke = mass + stiff
        fe = jnp.einsum("eg,ga->ea", src, N)
        rhs = jnp.zeros((grid.nx + 1, grid.ny + 1), dtype=history.dtype).at[I, J].add(fe)
        diag, upper = assembly.blocks(ke)
        d = blocktri.solve(diag, upper, rhs).reshape(grid.nx + 1, grid.ny + 1)
        return jnp.clip(d, 0.0, 1.0)
    w = (dx * dy / 4.0)[:, None, None]
    mass = jnp.einsum("eg,ga,gb->eab", 2.0 * (history + gc / (2.0 * l)), N, N)
    stiff = gc * l * jnp.einsum("egai,egbi->eab", G, G)
    ke = w * (mass + stiff)
    fe = (w[:, :, 0] * jnp.einsum("eg,ga->ea", 2.0 * history, N))
    rhs = jnp.zeros((grid.nx + 1, grid.ny + 1), dtype=history.dtype).at[I, J].add(fe)
    diag, upper = assembly.blocks(ke)
    d = blocktri.solve(diag, upper, rhs).reshape(grid.nx + 1, grid.ny + 1)
    return jnp.clip(d, 0.0, 1.0)


def damage_at_gauss(d: jax.Array, I: jax.Array, J: jax.Array) -> jax.Array:
    return d[I, J] @ jnp.asarray(N_GAUSS).T


def crack_energy(d: jax.Array, gc, l, dx: jax.Array, dy: jax.Array, I: jax.Array, J: jax.Array,
                 weights: jax.Array | None = None) -> jax.Array:
    """The surface term, per unit thickness of the model (or the whole ring's, with axisymmetric ``weights``)."""
    de = d[I, J]
    dq = de @ jnp.asarray(N_GAUSS).T
    G = jax.vmap(gauss_gradients)(dx, dy)
    grad = jnp.einsum("ea,egai->egi", de, G)
    w = (dx * dy / 4.0)[:, None] if weights is None else weights
    return jnp.sum(w * gc * (dq ** 2 / (2 * l) + 0.5 * l * jnp.sum(grad ** 2, -1)))


def updated_history(previous: jax.Array, plus: jax.Array) -> jax.Array:
    """The history after a state whose tension energy is ``plus``: never below ``previous``."""
    return jnp.maximum(previous, plus)
