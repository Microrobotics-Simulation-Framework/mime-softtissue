"""A direct solve of a block-tridiagonal system in pure JAX: block LU by a forward and a backward ``lax.scan``.

A structured grid numbered column by column gives a stiffness matrix whose only non-zero blocks are a column's own
(``diag``, one per column) and its coupling to the next column (``upper``); the lower blocks are the upper's
transposes (the matrix is symmetric). Cost: about four block sizes cubed a column. Differentiable in both modes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve


def solve(diag: jax.Array, upper: jax.Array, rhs: jax.Array) -> jax.Array:
    """``x`` with ``A x = rhs``; ``diag`` (n, m, m), ``upper`` (n-1, m, m) the block coupling column i to i+1 (rows
    in i, columns in i+1), ``rhs`` (n, m)."""

    first = lu_factor(diag[0])

    def forward(carry, block):
        factor, y_prev = carry
        d, u_prev, b = block  # u_prev couples the previous column to this one
        z = lu_solve(factor, u_prev)
        s = d - u_prev.T @ z
        y = b - u_prev.T @ lu_solve(factor, y_prev)
        new = lu_factor(s)
        return (new, y), (new, y)

    (_, _), (factors, ys) = jax.lax.scan(forward, (first, rhs[0]), (diag[1:], upper, rhs[1:]))
    lus = jnp.concatenate([first[0][None], factors[0]], axis=0)
    pivots = jnp.concatenate([first[1][None], factors[1]], axis=0)
    ys = jnp.concatenate([rhs[0][None], ys], axis=0)
    x_last = lu_solve((lus[-1], pivots[-1]), ys[-1])

    def backward(x_next, block):
        lu, piv, y, u = block
        x = lu_solve((lu, piv), y - u @ x_next)
        return x, x

    _, xs = jax.lax.scan(backward, x_last, (lus[:-1], pivots[:-1], ys[:-1], upper), reverse=True)
    return jnp.concatenate([xs, x_last[None]], axis=0)


def matvec(diag: jax.Array, upper: jax.Array, x: jax.Array) -> jax.Array:
    """``A x`` for the same storage (a check on :func:`solve`)."""
    y = jnp.einsum("nij,nj->ni", diag, x)
    y = y.at[:-1].add(jnp.einsum("nij,nj->ni", upper, x[1:]))
    return y.at[1:].add(jnp.einsum("nji,nj->ni", upper, x[:-1]))
