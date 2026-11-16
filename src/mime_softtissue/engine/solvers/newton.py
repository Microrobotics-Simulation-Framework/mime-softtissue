"""Damped Newton minimisation of an energy over free degrees of freedom: the step by the direct block-tridiagonal
solve, halved until the energy does not rise. Differentiable in forward mode (a ``lax.while_loop`` of
differentiable steps); ``energy`` and ``hessian_blocks`` close over whatever the caller differentiates."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from ..assembly import constrain
from . import blocktri


def minimise(energy: Callable[[jax.Array], jax.Array],
             hessian_blocks: Callable[[jax.Array], tuple[jax.Array, jax.Array]], u: jax.Array, free: jax.Array,
             free_blocks: jax.Array, *, max_iter: int, step_tol: float,
             regularise: bool = False, stall_step: float = 0.0) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``(u, iterations, converged)``: the displacement found by Newton from ``u``, stopping when the largest step is
    under ``step_tol`` (converged) or after ``max_iter`` iterations (not). ``regularise``: where the Newton step is
    not a descent direction (an indefinite tangent), the tangent is shifted by tau times its mean diagonal on the free
    degrees of freedom, tau from 1e-6 up by tens, until it is (a Levenberg-type safeguard; off by default).
    ``stall_step`` (m, 0 off): a stagnation stop, the solve converged where an accepted step is under it and changes
    the energy by no more than its rounding (1e-13 of it): the floor a damaged, penalised system reaches above
    ``step_tol``."""
    nb = free_blocks.shape[0]

    def body(c):
        u, k, _ = c
        r = jax.grad(energy)(u) * free
        diag, upper = constrain(*hessian_blocks(u), free_blocks)
        rb = r.reshape(nb, -1)
        du = -blocktri.solve(diag, upper, rb).reshape(u.shape) * free
        if regularise:
            fb = free_blocks.astype(diag.dtype)
            dvals = jax.vmap(jnp.diag)(diag)
            scale = jnp.sum(jnp.abs(dvals) * fb) / jnp.maximum(jnp.sum(fb), 1.0)
            shift = jax.vmap(jnp.diag)(fb * scale)

            def not_descent(s):
                _, d, n = s
                slope = jnp.sum(r * d)
                return (slope >= -1e-14 * jnp.sqrt(jnp.sum(r * r) * jnp.sum(d * d))) & (n < 14)

            def bump(s):
                tau, _, n = s
                tau = jnp.where(tau == 0, 1e-6, 10.0 * tau)
                d = -blocktri.solve(diag + tau * shift, upper, rb).reshape(u.shape) * free
                return tau, d, n + 1

            _, du, _ = jax.lax.while_loop(not_descent, bump, (jnp.zeros((), u.dtype), du, 0))
        e0 = energy(u)
        slack = 1e-10 * jnp.abs(e0) + 1e-30

        def worse(s):
            _, e, n = s
            return ((~jnp.isfinite(e)) | (e > e0 + slack)) & (n < 20)

        def halve(s):
            a, _, n = s
            return a / 2, energy(u + (a / 2) * du), n + 1

        a, e_a, _ = jax.lax.while_loop(worse, halve, (jnp.ones((), u.dtype), energy(u + du), 0))
        size = a * jnp.max(jnp.abs(du))
        if stall_step:
            stalled = (size <= stall_step) & (jnp.abs(e_a - e0) <= 1e-13 * jnp.abs(e0))
            size = jnp.where(stalled, 0.0, size)
        if regularise:  # a line search that found no admissible point takes no step (the solve then reads capped)
            a = jnp.where(worse((a, e_a, 0)), 0.0, a)
            return u + a * du, k + 1, jnp.where(a > 0, size, jnp.inf)
        return u + a * du, k + 1, size

    def cond(c):
        _, k, size = c
        return (k < max_iter) & (size > step_tol)

    u, k, size = jax.lax.while_loop(cond, body, (u, 0, jnp.asarray(jnp.inf, u.dtype)))
    return u, jnp.asarray(k), size <= step_tol
