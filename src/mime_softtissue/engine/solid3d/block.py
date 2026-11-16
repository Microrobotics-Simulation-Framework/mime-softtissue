"""A block pressed by a rigid tip in 3-D, a quarter by symmetry (x >= 0, y >= 0; the tip's axis along z through the
origin), its bottom fixed, its sides and top free: each load step a staggered AT2 solve at the tip's depth.

The tip is a signed distance ``distance(tip, points, depth)`` (negative inside) evaluated at the nodes' deformed
positions, with a nodal penalty of stiffness ``tip["stiffness"]`` (Pa/m) weighted by each node's tributary area in plan
(the axisymmetric build's ring area). Frictionless. ``tip`` is the caller's mapping (traced): the shape's numbers and the
penalty's stiffness. The solid is the plugin's law (``engine.material``) through ``solid3d.fem``.

The solver is spike 3D-COST's ``all2``, the configuration that held its lines at h = 0.1 mm at about a sixth of the
baseline's cost: a graded grid (``ratio``, ``cap``, ``fine_xy``, ``fine_z``), an inexact inner Newton (each staggered
iteration's Newton stops at a step of tol + min(``inexact`` x the last damage change, 1e-6) x height, then one Newton to
the full tolerance), and the multigrid preconditioners built and applied in float32 (``mixed``), the outer Newton and
PCG in the state's dtype. Its fourth lever, adaptive load steps, is the caller's: a MIME node takes one load step a base
step. Newton is regularised (the tangent shifted by tau times its mean diagonal, tau from 1e-6 up by tens, at most 14
times, until the step descends; a PCG that meets negative curvature counts as not descending) and its step halved until
the energy does not rise."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..grid import graded_rows
from ..material import Material
from . import fem as F
from . import multigrid as MG

F32 = jnp.float32


class State(NamedTuple):
    u: jax.Array  # (nx+1, ny+1, nz+1, 3)
    d: jax.Array  # (nx+1, ny+1, nz+1)
    history: jax.Array  # (2, 2, 2, nx, ny, nz)
    delta: jax.Array  # the tip's depth at the last step


class StepOut(NamedTuple):
    force: jax.Array  # N, the whole block (the quarter's x 4), pushing the tip out (+z)
    energy: jax.Array  # J, stored, whole block
    crack_energy: jax.Array  # J, the phase field's surface term, whole block
    contact_energy: jax.Array  # J, the penalty's, whole block
    staggered: jax.Array
    newton: jax.Array
    newton_capped: jax.Array
    krylov: jax.Array
    shifts: jax.Array
    damage_krylov: jax.Array
    residual: jax.Array
    d_max: jax.Array


def rows(h: float, length: float, fine: float, ratio: float, cap: float, from_top: bool) -> np.ndarray:
    """Node coordinates from 0 to ``length``: spacing ``h`` within ``fine`` of 0 (or of ``length`` with
    ``from_top``), growing by ``ratio`` up to ``cap`` h beyond."""
    r = graded_rows(length, h, fine, ratio, cap)
    if from_top:
        r = length - r[::-1]
    r = np.asarray(r, float)
    r[0], r[-1] = 0.0, length
    return r


def tributary(x: np.ndarray) -> np.ndarray:
    d = np.diff(x)
    return np.concatenate([[0.0], d / 2]) + np.concatenate([d / 2, [0.0]])


def _cast(tree: Any, dtype: Any) -> Any:
    return jax.tree_util.tree_map(lambda v: jnp.asarray(v, dtype) if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating)
                                  else v, tree)


@dataclass
class Block3D:
    """The quarter block, ``half_width`` across in x and y from the axis and ``height`` tall; spacing ``h`` (m) within
    ``fine_xy`` of the axis and ``fine_z`` of the top. ``distance(tip, points, depth)``: the tip's signed distance at
    ``points`` (..., 3) in the block's frame (origin at the bottom face's centre) with the tip moved ``depth`` below its
    reference; ``tip`` is the caller's pytree (traced)."""

    half_width: float
    height: float
    h: float
    distance: Callable[..., jax.Array]
    fine_xy: float = 0.8e-3
    fine_z: float = 1.0e-3
    ratio: float = 1.5
    cap: float = 16.0
    inexact: float = 1e-3
    mixed: bool = True
    rtol: float = 1e-8
    maxit: int = 200
    rtol_d: float = 1e-10
    max_newton: int = 100
    max_staggered: int = 400
    tol_d: float = 1e-5
    opts: MG.GMGOptions = field(default_factory=lambda: MG.GMGOptions(galerkin=True, degree=4))

    def __post_init__(self) -> None:
        x = rows(self.h, self.half_width, self.fine_xy, self.ratio, self.cap, False)
        z = rows(self.h, self.height, self.fine_z, self.ratio, self.cap, True)
        self.grid = F.Grid(x, x.copy(), z)
        nx, ny, nz = self.grid.nodes
        free = np.ones((nx, ny, nz, 3))
        free[0, :, :, 0] = 0.0  # x = 0: symmetry
        free[:, 0, :, 1] = 0.0  # y = 0: symmetry
        free[:, :, 0, :] = 0.0  # the bottom, fixed
        self.free_np = free
        X = np.stack(np.meshgrid(x, x, z, indexing="ij"), axis=-1)
        self.X_np = X
        w = tributary(x)
        self.w_np = np.broadcast_to((w[:, None] * w[None, :])[:, :, None], (nx, ny, nz)).copy()
        self.tol = 1e-11 * self.height

    @property
    def unknowns(self) -> int:
        return 3 * int(np.prod(self.grid.nodes))

    # ---- energies -------------------------------------------------------------------------------------------
    def phi(self, u: jax.Array, depth: jax.Array, tip: Any) -> jax.Array:
        return self.distance(tip, jnp.asarray(self.X_np, u.dtype) + u, depth)

    def contact(self, u: jax.Array, depth: jax.Array, tip: Any) -> jax.Array:
        gap = jnp.minimum(self.phi(u, depth, tip), 0.0)
        return 0.5 * jnp.asarray(tip["stiffness"], u.dtype) * jnp.sum(jnp.asarray(self.w_np, u.dtype) * gap * gap)

    def total(self, u: jax.Array, g: jax.Array, depth: jax.Array, mat: Material, tip: Any) -> jax.Array:
        return F.solid_energy(u, g, self.grid, mat) + self.contact(u, depth, tip)

    # ---- linear solves --------------------------------------------------------------------------------------
    def _solve(self, K: Callable, r: jax.Array, u: jax.Array, g: jax.Array, mat: Material, Kp: Callable | None):
        free = jnp.asarray(self.free_np, r.dtype)

        def A(v: jax.Array) -> jax.Array:
            return free * K(free * v) + (1.0 - free) * v

        low = F32 if self.mixed else u.dtype
        mat_l = _cast(mat, low)

        def make(gr: F.Grid, c: dict[str, jax.Array]) -> Callable:
            return lambda v: F.solid_energy(v, c["g"], gr, mat_l)

        levels, transfers = MG.build_hierarchy(make, {"g": g.astype(low)}, u.astype(low), self.grid,
                                               jnp.asarray(self.free_np, low), self.opts, ("g",),
                                               fine_K=Kp if self.mixed else K)
        P = MG.gmg_solver(levels, transfers, self.opts)

        def Pc(v: jax.Array) -> jax.Array:
            return P(v.astype(low)).astype(v.dtype)

        x, k, _, neg = MG.pcg(A, Pc, r, jnp.zeros_like(r), self.rtol, self.maxit)
        lv = levels[0]
        if isinstance(lv, MG.Coarsest):
            diag = jnp.ones_like(r)
        else:
            diag = jnp.diagonal(jnp.linalg.inv(lv.D_inv), axis1=-2, axis2=-1).astype(r.dtype)
        return x, k, neg, diag

    def newton(self, u: jax.Array, g: jax.Array, depth: jax.Array, mat: Material, tip: Any, tol: jax.Array):
        free = jnp.asarray(self.free_np, u.dtype)

        def E_(v: jax.Array) -> jax.Array:
            return self.total(v, g, depth, mat, tip)

        gradE = jax.grad(E_)
        if self.mixed:
            g32, z32, m32, t32 = g.astype(F32), depth.astype(F32), _cast(mat, F32), _cast(tip, F32)
            gradE32 = jax.grad(lambda v: self.total(v, g32, z32, m32, t32))
        zi = jnp.zeros((), jnp.int32)

        def body(c: tuple) -> tuple:
            u, k, size, kry, shifts = c
            r = gradE(u) * free
            K = jax.linearize(gradE, u)[1]
            Kp = jax.linearize(gradE32, u.astype(F32))[1] if self.mixed else None
            x, kk, neg, diag = self._solve(K, r, u, g, mat, Kp)
            kk = kk.astype(jnp.int32)
            du = -x * free
            scale = jnp.sum(jnp.abs(diag) * free) / jnp.maximum(jnp.sum(free), 1.0)

            def not_descent(s: tuple) -> jax.Array:
                tau, d, ng, n, kk = s
                slope = jnp.sum(r * d)
                return (ng | (slope >= -1e-14 * jnp.sqrt(jnp.sum(r * r) * jnp.sum(d * d)))) & (n < 14)

            def bump(s: tuple) -> tuple:
                tau, _, _, n, kk = s
                tau = jnp.where(tau == 0, 1e-6, 10.0 * tau)

                def Ks(v: jax.Array) -> jax.Array:
                    return K(v) + (tau * scale) * v

                Kps = (lambda v: Kp(v) + (tau * scale).astype(F32) * v) if Kp is not None else None
                x2, k2, ng2, _ = self._solve(Ks, r, u, g, mat, Kps)
                return tau, -x2 * free, ng2, n + 1, kk + k2.astype(jnp.int32)

            _, du, _, nb, kk = jax.lax.while_loop(not_descent, bump, (jnp.zeros((), u.dtype), du, neg, zi, kk))
            shifts = shifts + nb
            e0 = E_(u)
            slack = 1e-10 * jnp.abs(e0) + 1e-30

            def worse(s: tuple) -> jax.Array:
                a, e, n = s
                return ((~jnp.isfinite(e)) | (e > e0 + slack)) & (n < 20)

            def halve(s: tuple) -> tuple:
                a, _, n = s
                return a / 2, E_(u + (a / 2) * du), n + 1

            a, e_a, _ = jax.lax.while_loop(worse, halve, (jnp.ones((), u.dtype), E_(u + du), 0))
            bad = (~jnp.isfinite(e_a)) | (e_a > e0 + slack)
            a = jnp.where(bad, 0.0, a)
            size = jnp.where(a > 0, a * jnp.max(jnp.abs(du)), jnp.inf)
            return u + a * du, k + 1, size, kry + kk, shifts

        def cond(c: tuple) -> jax.Array:
            return (c[1] < self.max_newton) & (c[2] > tol)

        u, k, size, kry, shifts = jax.lax.while_loop(cond, body, (u, zi, jnp.asarray(jnp.inf, u.dtype), zi, zi))
        return u, k, size <= tol, kry, shifts

    def solve_damage(self, H: jax.Array, mat: Material) -> tuple[jax.Array, jax.Array]:
        gr = self.grid
        low = F32 if self.mixed else H.dtype
        gc, l = mat.gc, mat.l

        def Ed(d: jax.Array) -> jax.Array:
            return F.damage_energy(d[..., 0], H, gr, gc, l)

        def make(g2: F.Grid, c: dict[str, jax.Array]) -> Callable:
            return lambda d: F.damage_energy(d[..., 0], c["H"], g2, jnp.asarray(gc, low), jnp.asarray(l, low))

        d0 = jnp.zeros(gr.nodes + (1,), H.dtype)
        gradE = jax.grad(Ed)
        K = jax.linearize(gradE, d0)[1]
        dopts = MG.GMGOptions(galerkin=False)
        levels, transfers = MG.build_hierarchy(make, {"H": H.astype(low)}, d0.astype(low), gr,
                                               jnp.ones(gr.nodes + (1,), low), dopts, ("H",))
        P = MG.gmg_solver(levels, transfers, dopts)

        def Pc(v: jax.Array) -> jax.Array:
            return P(v.astype(low)).astype(v.dtype)

        b = -gradE(d0)
        x, k, _, _ = MG.pcg(K, Pc, b, jnp.zeros_like(b), self.rtol_d, self.maxit)
        return jnp.clip(x[..., 0], 0.0, 1.0), k

    # ---- the step -------------------------------------------------------------------------------------------
    def initial(self, dtype: Any = jnp.float64) -> State:
        gr = self.grid
        return State(jnp.zeros(gr.nodes + (3,), dtype), jnp.zeros(gr.nodes, dtype),
                     jnp.zeros((2, 2, 2) + gr.shape, dtype), jnp.zeros((), dtype))

    def _tol(self, err: jax.Array) -> jax.Array:
        if self.inexact <= 0:
            return jnp.asarray(self.tol, err.dtype)
        e = jnp.where(jnp.isfinite(err), err, 1.0)
        return self.tol + jnp.minimum(self.inexact * e, 1e-6) * self.height

    def advance(self, state: State, depth: jax.Array, mat: Material, tip: Any) -> tuple[State, StepOut]:
        """One load step to the tip's ``depth``: the proportional predictor (the last state where it inverts an
        element), the staggered loop, then (inexact) one Newton to the full tolerance."""
        depth = jnp.asarray(depth, state.u.dtype)
        prev = state.delta
        u = state.u * jnp.where(prev > 0, depth / jnp.where(prev > 0, prev, 1.0), 1.0)
        e_pred = F.solid_energy(u, F.degradation(state.d, mat.k_res), self.grid, mat)
        u = jnp.where(jnp.isfinite(e_pred), u, state.u)
        zi = jnp.zeros((), jnp.int32)

        def body(c: tuple) -> tuple:
            u, d, hist, k, nk, err, d_used, capped, kry, shifts, dk = c
            g = F.degradation(d, mat.k_res)
            u, n, ok, k1, k4 = self.newton(u, g, depth, mat, tip, self._tol(err))
            plus, _ = F.plus_minus(u, self.grid, mat)
            hist = jnp.maximum(state.history, plus)
            d_new, kd = self.solve_damage(hist, mat)
            return (u, d_new, hist, k + 1, nk + n, jnp.max(jnp.abs(d_new - d)), d, capped + (~ok).astype(jnp.int32),
                    kry + k1, shifts + k4, dk + kd.astype(jnp.int32))

        def cond(c: tuple) -> jax.Array:
            return (c[3] < self.max_staggered) & (c[5] > self.tol_d)

        init = (u, state.d, state.history, zi, zi, jnp.asarray(jnp.inf, u.dtype), state.d, zi, zi, zi, zi)
        u, d, hist, k, nk, err, d_used, capped, kry, shifts, dk = jax.lax.while_loop(cond, body, init)
        g = F.degradation(d_used, mat.k_res)
        if self.inexact > 0:  # the last equilibrium to the full tolerance
            u, n, ok, k1, k4 = self.newton(u, g, depth, mat, tip, jnp.asarray(self.tol, u.dtype))
            nk, capped, kry, shifts = nk + n, capped + (~ok).astype(jnp.int32), kry + k1, shifts + k4
        force = 4.0 * jax.grad(lambda z: self.contact(u, z, tip))(depth)
        out = StepOut(force=force, energy=4.0 * F.solid_energy(u, g, self.grid, mat),
                      crack_energy=4.0 * F.crack_energy(d, self.grid, mat.gc, mat.l),
                      contact_energy=4.0 * self.contact(u, depth, tip), staggered=k, newton=nk, newton_capped=capped,
                      krylov=kry, shifts=shifts, damage_krylov=dk, residual=err, d_max=jnp.max(d))
        return State(u, d, hist, depth), out


def with_options(block: Block3D, **changes: Any) -> Block3D:
    """The same block with some settings changed (a new grid where the spacing changes)."""
    return replace(block, **changes)
