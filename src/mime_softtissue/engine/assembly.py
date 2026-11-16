"""Index maps from element matrices to the block-tridiagonal storage of ``solvers.blocktri``, for a field with
``ndof`` components a node, the blocks being the grid's node columns; and the constraint of fixed degrees of
freedom."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .grid import Grid
from .kinematics.plane_strain import COLUMN, ROW


class Assembly:
    def __init__(self, grid: Grid, ndof: int) -> None:
        self.ndof, self.nb, self.m = ndof, grid.nx + 1, ndof * (grid.ny + 1)
        i, j = grid.element_columns()
        local = np.arange(4 * ndof)
        node, comp = local // ndof, local % ndof
        col, loc = COLUMN[node], ndof * ROW[node] + comp
        row_in_block = ndof * j[:, None] + loc[None, :]  # (n_el, nloc)
        p, q = np.meshgrid(local, local, indexing="ij")
        same = col[p] == col[q]
        up = (col[p] == 0) & (col[q] == 1)
        m = self.m
        rp, rq = row_in_block[:, p], row_in_block[:, q]  # (n_el, nloc, nloc)
        block = i[:, None, None] + col[p][None]
        self.d_index = ((block * m + rp) * m + rq)[:, same]
        self.d_pick = same
        self.u_index = ((i[:, None, None] * m + rp) * m + rq)[:, up]
        self.u_pick = up

    def blocks(self, ke: jax.Array) -> tuple[jax.Array, jax.Array]:
        nb, m = self.nb, self.m
        diag = jnp.zeros(nb * m * m, dtype=ke.dtype).at[self.d_index.ravel()].add(ke[:, self.d_pick].ravel())
        upper = jnp.zeros((nb - 1) * m * m, dtype=ke.dtype).at[self.u_index.ravel()].add(ke[:, self.u_pick].ravel())
        return diag.reshape(nb, m, m), upper.reshape(nb - 1, m, m)


def constrain(diag: jax.Array, upper: jax.Array, free: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The system with fixed degrees of freedom (``free`` 0) replaced by identity rows and columns; ``free`` (nb, m)."""
    f = free.astype(diag.dtype)
    diag = diag * f[:, :, None] * f[:, None, :] + jax.vmap(jnp.diag)(1.0 - f)
    upper = upper * f[:-1, :, None] * f[1:, None, :]
    return diag, upper
