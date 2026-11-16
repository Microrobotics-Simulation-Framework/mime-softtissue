"""A monolithic quasi-Newton solve of one load step of the phase-field solid (after Kristensen and Martinez-Paneda
2020): limited-memory BFGS on the coupled residual (displacement and damage), its initial matrix the two block
solves the staggered scheme already has (the displacement's tangent and the damage's matrix, the coupling dropped),
refreshed every ``refresh`` iterations; a backtracking line search on the residual's size in the initial matrix's
diagonal scaling; damage kept in [0, 1] by projection, the residual taken in the bounds' sense (zero where a bound
is active and the residual pushes past it). Pure JAX, fixed shapes."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from . import blocktri


def solve(residual: Callable[[jax.Array], jax.Array], blocks: Callable[[jax.Array], tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]],
          z0: jax.Array, n_u: int, shape_u: tuple[int, ...], shape_d: tuple[int, ...], *, memory: int = 20, max_iter: int = 1000, refresh: int = 100, tol_u: float, tol_d: float
          ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``(z, iterations, converged)``. ``z`` = (displacement, damage) flattened, the first ``n_u`` the displacement;
    ``blocks(z)`` gives ((diag_u, upper_u), (diag_d, upper_d)) of the initial matrix at ``z``, the displacement's
    already constrained, in block shapes ``shape_u`` and ``shape_d``. Converged when a step moves no displacement by more than ``tol_u`` and no
    damage by more than ``tol_d`` with a step the line search did not cut below a quarter."""
    n = z0.shape[0]
    dt = z0.dtype

    def project(z):
        return z.at[n_u:].set(jnp.clip(z[n_u:], 0.0, 1.0))

    def bound_residual(z, r):
        d, rd = z[n_u:], r[n_u:]
        stuck = ((d <= 0.0) & (rd > 0.0)) | ((d >= 1.0) & (rd < 0.0))
        return r.at[n_u:].set(jnp.where(stuck, 0.0, rd))

    def factor(z):
        (du, uu), (dd, ud) = blocks(z)
        scale = jnp.concatenate([jax.vmap(jnp.diag)(du).ravel(), jax.vmap(jnp.diag)(dd).ravel()])
        return (du, uu), (dd, ud), jnp.where(scale > 0, scale, 1.0)

    def h0(fac, v):
        (du, uu), (dd, ud), _ = fac
        a = blocktri.solve(du, uu, v[:n_u].reshape(shape_u)).ravel()
        b = blocktri.solve(dd, ud, v[n_u:].reshape(shape_d)).ravel()
        return jnp.concatenate([a, b])

    def direction(fac, g, S, Y, rho, k):
        m = S.shape[0]
        count = jnp.minimum(k, m)

        def first(i, c):
            q, al = c
            j = (k - 1 - i) % m
            use = i < count
            a = jnp.where(use, rho[j] * jnp.dot(S[j], q), 0.0)
            return q - a * Y[j], al.at[j].set(a)

        q, al = jax.lax.fori_loop(0, m, first, (g, jnp.zeros(m, dt)))
        r = h0(fac, q)

        def second(i, r):
            j = (k - count + i) % m
            use = i < count
            b = rho[j] * jnp.dot(Y[j], r)
            return jnp.where(use, r + S[j] * (al[j] - b), r)

        return -jax.lax.fori_loop(0, m, second, r)

    def merit(fac, r):
        return jnp.sum(r * r / fac[2])

    r0 = bound_residual(z0, residual(z0))
    fac0 = factor(z0)

    def body(c):
        z, r, fac, S, Y, rho, k, kk, _, _, _ = c
        fac = jax.lax.cond((kk > 0) & (kk % refresh == 0), lambda _: factor(z), lambda _: fac, None)
        reset = (kk > 0) & (kk % refresh == 0)
        k = jnp.where(reset, 0, k)
        p = direction(fac, r, S, Y, rho, k)
        m0 = merit(fac, r)

        def worse(s):
            a, _, rt, i = s
            return (~(merit(fac, rt) < (1.0 - 1e-4 * a) * m0) | ~jnp.all(jnp.isfinite(rt))) & (i < 12)

        def halve(s):
            a, _, _, i = s
            zt = project(z + 0.5 * a * p)
            return 0.5 * a, zt, bound_residual(zt, residual(zt)), i + 1

        zt = project(z + p)
        a, zt, rt, _ = jax.lax.while_loop(worse, halve, (jnp.ones((), dt), zt, bound_residual(zt, residual(zt)), 0))
        s = zt - z
        y = rt - r
        sy = jnp.dot(s, y)
        ok = sy > 1e-12 * jnp.sqrt(jnp.dot(s, s) * jnp.dot(y, y))
        j = k % S.shape[0]
        S_next = jnp.where(ok, S.at[j].set(s), S)
        Y_next = jnp.where(ok, Y.at[j].set(y), Y)
        rho = jnp.where(ok, rho.at[j].set(1.0 / jnp.where(ok, sy, 1.0)), rho)
        k = k + ok.astype(k.dtype)
        du = jnp.max(jnp.abs(s[:n_u]))
        dd = jnp.max(jnp.abs(s[n_u:]))
        return zt, rt, fac, S_next, Y_next, rho, k, kk + 1, du, dd, a

    def done(du, dd, a):  # a small step that was not cut short by the line search
        return (du <= tol_u) & (dd <= tol_d) & (a >= 0.25)

    def cond(c):
        *_, kk, du, dd, a = c
        return (kk < max_iter) & ~done(du, dd, a)

    init = (z0, r0, fac0, jnp.zeros((memory, n), dt), jnp.zeros((memory, n), dt), jnp.zeros(memory, dt),
            jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32), jnp.asarray(jnp.inf, dt), jnp.asarray(jnp.inf, dt),
            jnp.ones((), dt))
    z, _, _, _, _, _, _, kk, du, dd, a = jax.lax.while_loop(cond, body, init)
    return z, kk, done(du, dd, a)
