"""The matrix-free geometric multigrid of the 3-D build (rediscretised or Galerkin coarse operators, Chebyshev
smoothing on node-block Jacobi, a dense coarsest solve) and preconditioned CG. From spike 3D-PUNCTURE's ``fem3d.py``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .fem import Grid


def coarsen_gauss(a: jax.Array, axes: tuple[bool, bool, bool], harmonic: bool = False) -> jax.Array:
    """A Gauss-point array (2, 2, 2, nx, ny, nz) rediscretised on the coarse grid: each coarse Gauss point takes the
    mean (arithmetic, or harmonic) of the fine Gauss points inside the fine element it lies in, per coarsened axis
    (an odd axis's last element is kept as it is)."""
    if harmonic:
        return 1.0 / coarsen_gauss(1.0 / a, axes, False)
    for ax, f in enumerate(axes):
        if not f:
            continue
        n = a.shape[3 + ax]
        tail = None
        if n % 2:
            tail = jax.lax.slice_in_dim(a, n - 1, n, axis=3 + ax)
            a = jax.lax.slice_in_dim(a, 0, n - 1, axis=3 + ax)
            n -= 1
        m = jnp.mean(a, axis=ax)                      # element axis now at 2 + ax
        m = jnp.moveaxis(m, 2 + ax, -1)
        m = m.reshape(m.shape[:-1] + (n // 2, 2))     # (..., E, S): fine element 2E + S holds coarse side S
        m = jnp.moveaxis(m, -1, ax)                   # S to the side position
        a = jnp.moveaxis(m, -1, 3 + ax)               # E back to the element position
        if tail is not None:
            a = jnp.concatenate([a, tail], axis=3 + ax)
    return a


def inject(f: jax.Array, axes: tuple[bool, bool, bool]) -> jax.Array:
    for ax, c in enumerate(axes):
        if c:
            n = f.shape[ax] - 1
            even = jax.lax.slice_in_dim(f, 0, n + 1, stride=2, axis=ax)
            f = even if n % 2 == 0 else jnp.concatenate([even, jax.lax.slice_in_dim(f, n, n + 1, axis=ax)], axis=ax)
    return f


def prolong(v: jax.Array, fine: Grid, axes: tuple[bool, bool, bool]) -> jax.Array:
    """Trilinear interpolation (rectilinear weights) of a coarse nodal field onto the fine nodes."""
    for ax, f in enumerate(axes):
        if not f:
            continue
        c = (fine.x, fine.y, fine.z)[ax]
        n = len(c) - 1
        tail = None
        if n % 2:
            tail = jax.lax.slice_in_dim(v, v.shape[ax] - 1, v.shape[ax], axis=ax)
            v = jax.lax.slice_in_dim(v, 0, v.shape[ax] - 1, axis=ax)
            c = c[:-1]
        t = (c[1::2] - c[0:-1:2]) / (c[2::2] - c[0:-1:2])
        v = jnp.moveaxis(v, ax, 0)
        tt = jnp.asarray(t, dtype=v.dtype).reshape((-1,) + (1,) * (v.ndim - 1))
        odd = (1.0 - tt) * v[:-1] + tt * v[1:]
        inter = jnp.stack([v[:-1], odd], axis=1).reshape((2 * (v.shape[0] - 1),) + v.shape[1:])
        v = jnp.moveaxis(jnp.concatenate([inter, v[-1:]], axis=0), 0, ax)
        if tail is not None:
            v = jnp.concatenate([v, tail], axis=ax)
    return v


# ------------------------------------------------------------------------------------------------- the multigrid ---
@dataclass
class GMGOptions:
    degree: int = 2            # Chebyshev degree, pre and post
    smoothing_range: float = 15.0
    lam_factor: float = 1.1
    power_its: int = 12
    min_cells: int = 2         # stop coarsening an axis below this many cells
    max_coarse: int = 400     # the coarsest level's unknowns, solved densely
    harmonic: bool = False     # coarse g by harmonic rather than arithmetic mean
    max_levels: int = 12
    coarse_state: str = "inject"  # the coarse levels' linearisation point: the injected state, or "zero" (small strain)
    galerkin: bool = False     # coarse operators R A P by coloured probing (27-point stencils) instead of rediscretised


class Level:
    def __init__(self, grid, free, K, opts, key):
        self.grid, self.free, self.K = grid, free, K
        self.ncomp = free.shape[-1]
        self.shape = free.shape
        self.n = int(np.prod(self.shape))
        self.D_inv = self._block_diagonal_inverse()
        self.lam = self._lam_max(opts, key)

    def A(self, v):
        return self.free * self.K(self.free * v) + (1.0 - self.free) * v

    def _block_diagonal_inverse(self):
        """The node-block diagonal by 8 x ncomp coloured probes (nodes two apart in every index share no element),
        in a fori_loop so the operator is compiled once."""
        c = self.ncomp
        nx, ny, nz = self.shape[:3]
        dt = self.free.dtype
        ix = jnp.asarray(np.arange(nx) % 2)[:, None, None]
        iy = jnp.asarray(np.arange(ny) % 2)[None, :, None]
        iz = jnp.asarray(np.arange(nz) % 2)[None, None, :]
        comps = jnp.arange(c)

        def body(i, D):
            col, j = i // c, i % c
            a, b, cc = col // 4, (col // 2) % 2, col % 2
            mask = ((ix == a) & (iy == b) & (iz == cc)).astype(dt)
            onehot = (comps == j).astype(dt)
            w = mask[..., None] * self.A(mask[..., None] * onehot)
            return D + w[..., :, None] * onehot

        D = jax.lax.fori_loop(0, 8 * c, body, jnp.zeros(self.shape[:3] + (c, c), dt))
        return jnp.linalg.inv(D)

    def M(self, r):
        return jnp.einsum("...ij,...j->...i", self.D_inv, r)

    def _lam_max(self, opts, key):
        v = jax.random.normal(key, self.shape, self.free.dtype) * self.free
        def body(_, c):
            v, lam = c
            w = self.M(self.A(v))
            nw = jnp.sqrt(jnp.sum(w * w))
            return w / nw, nw / jnp.sqrt(jnp.sum(v * v))
        v = v / jnp.sqrt(jnp.sum(v * v))
        _, lam = jax.lax.fori_loop(0, opts.power_its, body, (v, jnp.ones((), v.dtype)))
        return lam


class Coarsest:
    def __init__(self, grid, free, K):
        self.grid, self.free, self.K = grid, free, K
        self.shape = free.shape
        n = int(np.prod(self.shape))
        A = jax.vmap(lambda e: self.A(e.reshape(self.shape)).ravel())(jnp.eye(n, dtype=free.dtype)).T
        self.lu = jax.scipy.linalg.lu_factor(A)

    def A(self, v):
        return self.free * self.K(self.free * v) + (1.0 - self.free) * v

    def solve(self, b):
        return jax.scipy.linalg.lu_solve(self.lu, b.ravel()).reshape(self.shape)


OFFSETS = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]


def stencil_apply(S: jax.Array, v: jax.Array) -> jax.Array:
    """y[J] = sum_o S[J, o] v[J + o] over the 27 neighbours (zero outside): S (nx, ny, nz, 27, c, c), v (nx, ny, nz, c)."""
    nx, ny, nz = v.shape[:3]
    vp = jnp.pad(v, ((1, 1), (1, 1), (1, 1), (0, 0)))
    out = jnp.zeros_like(v)
    for o, (a, b, c) in enumerate(OFFSETS):
        sl = vp[1 + a:1 + a + nx, 1 + b:1 + b + ny, 1 + c:1 + c + nz]
        out = out + jnp.einsum("...ij,...j->...i", S[..., o, :, :], sl)
    return out


def probe_stencil(A: Callable, shape: tuple, dtype) -> jax.Array:
    """The 27-point block stencil of a linear operator on nodal fields whose coupling reaches one node in each index
    (a Galerkin coarse operator of trilinear transfers), by 27 x c coloured probes (period 3 in each index)."""
    nx, ny, nz, c = shape
    ix = jnp.asarray(np.arange(nx))[:, None, None]
    iy = jnp.asarray(np.arange(ny))[None, :, None]
    iz = jnp.asarray(np.arange(nz))[None, None, :]
    comps = jnp.arange(c)

    def body(i, S):
        col, j = i // c, i % c
        p, q, r = col // 9, (col // 3) % 3, col % 3
        mask = ((ix % 3 == p) & (iy % 3 == q) & (iz % 3 == r)).astype(S.dtype)
        onehot = (comps == j).astype(S.dtype)
        y = A(mask[..., None] * onehot)
        ox, oy, oz = (p - ix + 1) % 3 - 1, (q - iy + 1) % 3 - 1, (r - iz + 1) % 3 - 1
        o = (ox + 1) * 9 + (oy + 1) * 3 + (oz + 1)
        o = jnp.broadcast_to(o, (nx, ny, nz))
        oh27 = (o[..., None] == jnp.arange(27)).astype(S.dtype)
        return S + oh27[..., :, None, None] * y[..., None, :, None] * onehot[None, None, :]

    return jax.lax.fori_loop(0, 27 * c, body, jnp.zeros((nx, ny, nz, 27, c, c), dtype))


def build_hierarchy(make_energy: Callable, coeffs: dict, u_lin: jax.Array, grid: Grid, free: jax.Array,
                    opts: GMGOptions, gauss_keys: tuple[str, ...], fine_K=None):
    """Levels from fine to coarse. ``make_energy(grid, coeffs)`` returns u -> energy; ``coeffs`` holds Gauss-point
    arrays under ``gauss_keys`` (coarsened by the mean) and scalars. Each coarse level linearises its own energy's
    gradient at the injected state (rediscretised coarse operators), or with ``opts.galerkin`` is R A P of the level
    above, probed into a 27-point stencil."""
    levels, transfers = [], []
    key = jax.random.PRNGKey(0)
    g, c, u, f = grid, dict(coeffs), u_lin, free
    K = None
    for depth in range(opts.max_levels):
        if depth == 0 and fine_K is not None:
            K = fine_K
        elif depth == 0 or not opts.galerkin:
            K = jax.linearize(jax.grad(make_energy(g, c)), u)[1]
        else:
            fine_lv = levels[-1]
            fg, fax = transfers[-1]
            cfree = f

            def Ac(v, fine_lv=fine_lv, fg=fg, fax=fax, cfree=cfree):
                restrict = jax.linear_transpose(lambda w: prolong(w, fg, fax), jnp.zeros(cfree.shape, v.dtype))
                y = fine_lv.free * fine_lv.K(fine_lv.free * prolong(cfree * v, fg, fax))
                return cfree * restrict(y)[0] + (1.0 - cfree) * v

            S = probe_stencil(Ac, f.shape, f.dtype)
            K = (lambda S: (lambda v: stencil_apply(S, v)))(S)
        axes = g.coarse_axes(opts.min_cells)
        n = int(np.prod(f.shape))
        if n <= opts.max_coarse or not any(axes) or depth == opts.max_levels - 1:
            levels.append(Coarsest(g, f, K))
            break
        key, sub = jax.random.split(key)
        levels.append(Level(g, f, K, opts, sub))
        transfers.append((g, axes))
        c = {k: (coarsen_gauss(v, axes, opts.harmonic and k == "g") if k in gauss_keys else v) for k, v in c.items()}
        g, u, f = g.coarse(axes), inject(u, axes), inject(f, axes)
        if opts.coarse_state == "zero":
            u = jnp.zeros_like(u)
    return levels, transfers


def chebyshev(lv: Level, b, x, opts: GMGOptions, x_zero: bool):
    lmax = opts.lam_factor * lv.lam
    lmin = lmax / opts.smoothing_range
    theta, delta = 0.5 * (lmax + lmin), 0.5 * (lmax - lmin)
    sigma = theta / delta
    rho = 1.0 / sigma
    r = b if x_zero else b - lv.A(x)
    d = lv.M(r) / theta
    x = d if x_zero else x + d
    for _ in range(1, opts.degree):
        rho_new = 1.0 / (2.0 * sigma - rho)
        r = r - lv.A(d)
        d = rho_new * rho * d + (2.0 * rho_new / delta) * lv.M(r)
        x = x + d
        rho = rho_new
    return x


def vcycle(levels, transfers, opts: GMGOptions, b, depth: int = 0):
    lv = levels[depth]
    if isinstance(lv, Coarsest):
        return lv.solve(b)
    fine, axes = transfers[depth]
    x = chebyshev(lv, b, None, opts, x_zero=True)
    r = b - lv.A(x)
    coarse_free = levels[depth + 1].free
    restrict = jax.linear_transpose(lambda v: prolong(v, fine, axes), jnp.zeros(coarse_free.shape, b.dtype))
    rc = coarse_free * restrict(r)[0]
    ec = vcycle(levels, transfers, opts, rc, depth + 1)
    x = x + lv.free * prolong(coarse_free * ec, fine, axes)
    return chebyshev(lv, b, x, opts, x_zero=False)


# ----------------------------------------------------------------------------------------------------------- PCG ---
def pcg(A, M, b, x0, rtol, maxit, atol=0.0, flexible=False):
    """Preconditioned CG; stops at ||r|| <= max(rtol ||b||, atol), at ``maxit``, or on non-positive curvature
    (flagged). Returns x, iterations, the relative-residual history (maxit+1,), the flag. 3D-COST: ``flexible`` takes
    the Polak-Ribiere beta (r_new . (z_new - z_old) / rz), for a preconditioner that is not exactly linear (float32);
    without it the arithmetic is the original's."""
    bnorm = jnp.sqrt(jnp.sum(b * b))
    r = b - A(x0)
    z = M(r)
    rz = jnp.sum(r * z)
    hist = jnp.full((maxit + 1,), jnp.nan, b.dtype).at[0].set(jnp.sqrt(jnp.sum(r * r)) / bnorm)
    stop = jnp.maximum(rtol * bnorm, atol)

    def cond(c):
        x, r, z, p, rz, k, hist, neg = c
        return (k < maxit) & (jnp.sqrt(jnp.sum(r * r)) > stop) & (~neg) & jnp.isfinite(rz)

    def body(c):
        x, r, z, p, rz, k, hist, neg = c
        Ap = A(p)
        pAp = jnp.sum(p * Ap)
        neg = pAp <= 0
        alpha = jnp.where(neg, 0.0, rz / jnp.where(neg, 1.0, pAp))
        x = x + alpha * p
        r = r - alpha * Ap
        z_new = M(r)
        rz_new = jnp.sum(r * z_new)
        beta = ((rz_new - jnp.sum(r * z)) if flexible else rz_new) / rz
        p = z_new + beta * p
        hist = hist.at[k + 1].set(jnp.sqrt(jnp.sum(r * r)) / bnorm)
        return x, r, z_new, p, rz_new, k + 1, hist, neg

    x, r, z, p, rz, k, hist, neg = jax.lax.while_loop(cond, body, (x0, r, z, z, rz, 0, hist, jnp.zeros((), bool)))
    return x, k, hist, neg

def gmg_solver(levels, transfers, opts):
    return lambda r: vcycle(levels, transfers, opts, r)


