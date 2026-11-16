"""A rigid tip pressed into a soft block in 3-D: the 3-D kinematics option (``kinematics.solid3d``), the plugin's material
law (Yeoh in the AT2 split), the plugin's contact law (``contact``: a tip is an ``Indenter``, its signed distance
evaluated at R = |(x, y) - tip axis|, a nodal penalty, frictionless), and the staggered phase-field step.

The solver is the 3-D spikes' (3D-PUNCTURE, 3D-COST's ``all2``, 3D-STAGGER): a regularised Newton (the tangent shifted by
tau times its mean diagonal until the step descends, the step halved until the energy does not rise) whose linear solves
are PCG with one geometric-multigrid V-cycle (Galerkin coarse operators, Chebyshev on node-block Jacobi), the
preconditioner built and applied in float32 (``mixed``); an inexact inner Newton inside the staggered loop
(``inexact``); load steps adapted to the staggered count (``adaptive``); Anderson acceleration of the staggered loop on the
nodal damage (``anderson``); a step limit (``stagger_cap``, ``newton_cap``): a step that does not converge is rejected, never
accepted, and retried at half the increment.

Domains: ``quarter`` (x, y >= 0, symmetry planes x = 0 and y = 0, the tip's axis on their line), ``half`` (y >= 0, symmetry
plane y = 0, the axis at x = ``axis_x``), ``full``. The bottom face is fixed, the sides free, the top free but for the tip.
No MIME import."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import contact as ct
from .fracture import deletion
from .fracture.phase_field import discretisation_toughness
from .grid import graded_rows
from .kinematics import solid3d as S
from .material import Material, degradation
from .solvers import multigrid3d as MG

F32 = jnp.float32
SYMMETRIES = {"quarter": 4.0, "half": 2.0, "full": 1.0}


# ------------------------------------------------------------------------------------------------------ the tips ---
def blunt_punch(radius: float, fillet: float) -> ct.Indenter:
    """A flat-ended cylinder, its edge rounded by ``fillet`` (the contact law's ``punch``)."""
    return ct.Indenter("punch", radius=radius, fillet=fillet, top=0.0)


def sphere_tipped_needle(radius: float) -> ct.Indenter:
    """A cylinder of ``radius`` ending in a hemisphere of the same radius: the ``punch`` with its fillet equal to its
    radius."""
    return ct.Indenter("punch", radius=radius, fillet=radius, top=0.0)


def sharp_cone(half_angle: float, apex_radius: float, shaft_radius: float = 1.0) -> ct.Indenter:
    """A cone of ``half_angle`` (radians), its apex rounded by ``apex_radius``; its shaft (default 1 m) beyond any block,
    so only the cone touches (the contact law's ``cone``)."""
    return ct.Indenter("cone", radius=shaft_radius, fillet=apex_radius, top=0.0, half_angle=half_angle)


def needle(radius: float, half_angle: float, apex_radius: float) -> ct.Indenter:
    """A cylinder of ``radius`` ending in a conical point of ``half_angle``, its apex rounded by ``apex_radius`` (the
    contact law's ``cone`` with a shaft inside the block)."""
    return ct.Indenter("cone", radius=radius, fillet=apex_radius, top=0.0, half_angle=half_angle)


# ------------------------------------------------------------------------------------------- the block, settings ---
@dataclass(frozen=True)
class Block:
    """The block (m): ``2 half_width`` square in plan, ``height`` tall; the grid's spacing ``h`` within ``fine_xy`` of the
    tip's axis (in x and in y) and within ``fine_z`` of the top, growing by ``ratio`` up to ``cap`` h beyond."""

    half_width: float = 5e-3
    height: float = 5e-3
    h: float = 0.1e-3
    symmetry: str = "quarter"
    fine_xy: float = 0.8e-3
    fine_z: float = 1.0e-3
    ratio: float = 1.5
    cap: float = 16.0
    axis_x: float = 0.0  # the tip's axis in x (half and full domains: an offset from the grid's centre line)


@dataclass(frozen=True)
class Settings:
    """The solver's settings; the defaults are 3D-COST's ``all2`` (which held its lines against the uniform-step
    baseline at a sixth of its cost) with 3D-STAGGER's Anderson depth 3 and a step limit."""

    rtol: float = 1e-8
    maxit: int = 200
    rtol_damage: float = 1e-10
    max_newton: int = 100
    max_staggered: int = 400
    tol_damage: float = 1e-5
    newton_tol: float = 1e-11  # times the block's height
    chebyshev_degree: int = 4
    galerkin: bool = True
    penalty_factor: float = 100.0  # the contact stiffness, times E0 / h
    inexact: float = 1e-3
    mixed: bool = True
    anderson: int = 3
    stagger_cap: int = 60
    newton_cap: int = 60
    adaptive: bool = True
    step: float = 0.04e-3  # the largest step (adaptive) or the step (fixed), m
    first_step: float = 0.02e-3
    min_step: float = 0.0025e-3
    grow_at: int = 8
    shrink_at: int = 25
    volumetric: str = "mean"
    #: The material's tension part past full damage: "residual" keeps k_res times the whole law (round 1's arithmetic);
    #: "neo" keeps k_res times the neo-Hookean part only, so a fully damaged point carries no Yeoh stiffening.
    residual: str = "residual"
    trace: bool = False  # print a line a staggered pass (depth, pass, damage change, Newton count, converged, Krylov)
    #: "phase_field" (AT2) or "deletion": the plugin's control without a length scale, an element deleted for good once
    #: the largest principal stretch at its centre passes ``Material.stretch_limit`` (mesh-dependent by construction).
    fracture: str = "phase_field"
    #: The damage's bounds 0 <= d <= 1: "clip" (the spikes' arithmetic: the unconstrained solve clipped; it overshoots
    #: 1 where one node is driven hard, e.g. under a sharp tip, and the clipped field is not a minimiser) or
    #: "active_set" (both bounds held exactly, ``_damage_bounded``; identical where nothing leaves [0, 1]).
    damage_bound: str = "active_set"
    #: Where the tip's penalty acts: "nodes" (the contact law's nodal penalty, weighted by plan area) or "gauss" (a
    #: volume penalty at the Gauss points, 100 E0 / h^2: catches a sharp tip between nodes).
    contact_at: str = "nodes"


class State(NamedTuple):
    u: jax.Array
    d: jax.Array  # nodal damage (with deletion: the largest of the node's elements')
    history: jax.Array
    depth: jax.Array
    deleted: jax.Array  # (nx, ny, nz): 1 where an element is deleted (the deletion control only)


class StepOut(NamedTuple):
    force: jax.Array
    energy: jax.Array
    crack_energy: jax.Array
    contact_energy: jax.Array
    staggered: jax.Array
    newton: jax.Array
    newton_capped: jax.Array
    krylov: jax.Array
    shifts: jax.Array
    damage_krylov: jax.Array
    residual: jax.Array
    converged: jax.Array
    aa_restarts: jax.Array
    d_max: jax.Array
    penetration: jax.Array  # the deepest node inside the tip, over h


# ------------------------------------------------------------------------------------------- Anderson (3D-STAGGER) ---
class AAState(NamedTuple):
    DF: jax.Array
    DG: jax.Array
    nh: jax.Array
    pos: jax.Array
    fp: jax.Array
    gp: jax.Array
    fn_prev: jax.Array
    started: jax.Array


def aa_init(m: int, n: int, dtype: Any) -> AAState:
    z = jnp.zeros((m, n), dtype)
    zi = jnp.zeros((), jnp.int32)
    return AAState(z, z, zi, zi, jnp.zeros((n,), dtype), jnp.zeros((n,), dtype), jnp.asarray(jnp.inf, dtype),
                   jnp.zeros((), bool))


def aa_update(s: AAState, x: jax.Array, g: jax.Array, f: jax.Array, m: int, reg: float = 1e-10):
    """One Anderson update (type II, mixing 1) from the iterate x, its map value g = G(x) and residual f = g - x (flat):
    restarted when ||f|| grows, column-scaled least squares with a Tikhonov term, the next iterate projected onto [0, 1],
    a non-finite update replaced by g. Returns (next iterate, state, restarted)."""
    fn = jnp.sqrt(jnp.sum(f * f))
    restart = s.started & (fn > s.fn_prev)
    DF = jnp.where(restart, 0.0, s.DF)
    DG = jnp.where(restart, 0.0, s.DG)
    nh = jnp.where(restart, 0, s.nh)
    pos = jnp.where(restart, 0, s.pos)
    store = s.started & ~restart
    DF = jnp.where(store, DF.at[pos].set(f - s.fp), DF)
    DG = jnp.where(store, DG.at[pos].set(g - s.gp), DG)
    nh = jnp.where(store, jnp.minimum(nh + 1, m), nh)
    pos = jnp.where(store, (pos + 1) % m, pos)
    mask = (jnp.arange(m) < nh).astype(f.dtype)
    A = DF * mask[:, None]
    Gm = A @ A.T
    b = A @ f
    dg = jnp.diagonal(Gm)
    sc = jnp.where(mask > 0, jnp.sqrt(jnp.where(dg > 0, dg, 1.0)), 1.0)
    mm = mask[:, None] * mask[None, :]
    Gs = mm * Gm / (sc[:, None] * sc[None, :]) + (1.0 - mm) * jnp.eye(m, dtype=f.dtype) + reg * jnp.eye(m, dtype=f.dtype)
    gam = jnp.linalg.solve(Gs, mask * b / sc) / sc * mask
    xn = g - (DG * mask[:, None]).T @ gam
    bad = ~(jnp.all(jnp.isfinite(xn)) & jnp.all(jnp.isfinite(gam)))
    xn = jnp.where((nh > 0) & ~bad, xn, g)
    return jnp.clip(xn, 0.0, 1.0), AAState(DF, DG, nh, pos, f, g, fn, jnp.ones((), bool)), restart


def _pass_line(depth, k, err, n, ok, kry, shifts) -> None:
    print(f"    pass depth {float(depth) * 1e3:.4f} mm k {int(k)} err {float(err):.2e} newton {int(n)} ok {bool(ok)} "
          f"krylov {int(kry)} shifts {int(shifts)} t {time.strftime('%H:%M:%S')}", flush=True)


# ------------------------------------------------------------------------------------------------------ the rows ---
def axis_rows(length: float, h: float, fine: float, ratio: float, cap: float, from_top: bool = False) -> np.ndarray:
    r = np.asarray(graded_rows(length, h, fine, ratio, cap), float)
    if from_top:
        r = length - r[::-1]
    r[0], r[-1] = 0.0, length
    return r


def tributary(x: np.ndarray) -> np.ndarray:
    d = np.diff(x)
    return np.concatenate([[0.0], d / 2]) + np.concatenate([d / 2, [0.0]])


def initial_modulus(mat: Material) -> float:
    """Young's modulus at small strain, 9 kappa mu / (3 kappa + mu)."""
    mu, kappa = float(mat.mu), float(mat.kappa)
    return 9.0 * kappa * mu / (3.0 * kappa + mu)


# ----------------------------------------------------------------------------------------------------- the solve ---
class Puncture3D:
    """One block, material, tip and settings: the jitted step and the run. ``material.gc`` is the toughness the run
    should dissipate (its input is corrected for the grid here)."""

    def __init__(self, block: Block, material: Material, tip: ct.Indenter, settings: Settings = Settings()):
        if block.symmetry not in SYMMETRIES:
            raise ValueError(f"symmetry is one of {tuple(SYMMETRIES)}")
        if block.symmetry == "quarter" and block.axis_x != 0.0:
            raise ValueError("a quarter domain puts the tip's axis on its two symmetry planes (axis_x = 0)")
        self.block, self.settings = block, settings
        self.multiplier = SYMMETRIES[block.symmetry]
        b, s = block, settings
        half = axis_rows(b.half_width, b.h, b.fine_xy, b.ratio, b.cap)
        full_x = np.concatenate([-half[::-1], half[1:]])
        x = half if b.symmetry == "quarter" else full_x
        y = half if b.symmetry in ("quarter", "half") else full_x
        z = axis_rows(b.height, b.h, b.fine_z, b.ratio, b.cap, from_top=True)
        self.grid = S.Grid3(x, y, z)
        nx, ny, nz = self.grid.nodes
        free = np.ones((nx, ny, nz, 3))
        if b.symmetry == "quarter":
            free[0, :, :, 0] = 0.0
        if b.symmetry in ("quarter", "half"):
            free[:, 0, :, 1] = 0.0
        free[:, :, 0, :] = 0.0
        self.free = jnp.asarray(free)
        self.free32 = self.free.astype(F32)
        X = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
        self.X = jnp.asarray(X)
        self.X32 = self.X.astype(F32)
        w = tributary(x)[:, None] * tributary(y)[None, :]
        self.w = jnp.asarray(np.broadcast_to(w[:, :, None], (nx, ny, nz)).copy())
        self.w32 = self.w.astype(F32)
        # Python floats, so a float32 preconditioner is not promoted by a numpy scalar (3D-COST's dtype finding)
        self.tip = replace(tip, top=float(b.height), **{k: float(v) for k, v in tip.geometry().items()})
        self.axis = (b.axis_x, 0.0)
        floats = {k: (None if v is None else (v if isinstance(v, tuple) else float(v)))
                  for k, v in material._asdict().items() if k != "fibre"}
        self.gc_physical = floats["gc"]
        self.mat = Material(**{**floats, "gc": discretisation_toughness(floats["gc"], b.h, floats["l"])})
        self.kpen = s.penalty_factor * initial_modulus(self.mat) / b.h
        if s.contact_at == "gauss":  # a volume penalty: the material inside the tip, at the Gauss points
            self.kpen = s.penalty_factor * initial_modulus(self.mat) / b.h ** 2
            self.w = S.gauss_weight(self.grid)[None, None, None] * jnp.ones((2, 2, 2, 1, 1, 1))
            self.w32 = self.w.astype(F32)
        self.tol = s.newton_tol * b.height
        self.opts = MG.GMGOptions(galerkin=s.galerkin, degree=s.chebyshev_degree)
        self.dopts = MG.GMGOptions(galerkin=False)
        if s.fracture not in ("phase_field", "deletion"):
            raise ValueError("fracture is phase_field or deletion")
        self.step = jax.jit(self._step if s.fracture == "phase_field" else self._step_deletion)
        self.unknowns = 3 * nx * ny * nz

    # ---- energies --------------------------------------------------------------------------------------------
    def solid(self, u, g):
        e = S.solid_energy(u, g, self.grid, self.mat, self.settings.volumetric)
        if self.settings.residual == "neo":  # past full damage only the neo-Hookean part keeps k_res
            m = self.mat
            stiff = m._replace(mu=0.0, kappa=0.0)
            k = jnp.asarray(m.k_res, u.dtype)
            e = e - k * S.solid_energy(u, jnp.ones_like(g), self.grid, stiff, self.settings.volumetric)
        return e

    def phi(self, X, u, depth):
        p = X + u
        R = ct._norm(p[..., 0] - self.axis[0], p[..., 1] - self.axis[1])
        return ct.signed_distance(self.tip, R, p[..., 2], depth)

    def _gauss_points(self, X, u):
        """Positions at the Gauss points (2, 2, 2, nx, ny, nz, 3) from nodal reference positions and displacements."""
        return jnp.stack([S.gauss_values(X[..., c] + u[..., c]) for c in range(3)], axis=-1)

    def _gap(self, X, u, depth):
        if self.settings.contact_at == "gauss":
            return jnp.minimum(self.phi(self._gauss_points(X, u), 0.0, depth), 0.0)
        return jnp.minimum(self.phi(X, u, depth), 0.0)

    def contact(self, u, depth):
        gap = self._gap(self.X, u, depth)
        return 0.5 * self.kpen * jnp.sum(self.w * gap * gap)

    def contact32(self, u, depth):
        gap = self._gap(self.X32, u, depth)
        return 0.5 * self.kpen * jnp.sum(self.w32 * gap * gap)

    def total(self, u, g, depth):
        return self.solid(u, g) + self.contact(u, depth)

    def total32(self, u, g, depth):
        return self.solid(u, g) + self.contact32(u, depth)

    # ---- linear solves ---------------------------------------------------------------------------------------
    def _solve(self, K, r, u, g, Kp=None):
        free = self.free
        A = lambda v: free * K(free * v) + (1.0 - free) * v
        vol = self.settings.volumetric
        mat = self.mat
        if self.settings.mixed:
            mk = lambda gr, c: (lambda v: S.solid_energy(v, c["g"], gr, mat, vol))
            levels, transfers = MG.build_hierarchy(mk, {"g": g.astype(F32)}, u.astype(F32), self.grid, self.free32,
                                                   self.opts, ("g",), fine_K=Kp)
            P32 = MG.gmg_solver(levels, transfers, self.opts)
            Pc = lambda v: P32(v.astype(F32)).astype(v.dtype)
        else:
            mk = lambda gr, c: (lambda v: S.solid_energy(v, c["g"], gr, mat, vol))
            levels, transfers = MG.build_hierarchy(mk, {"g": g}, u, self.grid, self.free, self.opts, ("g",), fine_K=K)
            Pc = MG.gmg_solver(levels, transfers, self.opts)
        x, k, _, neg = MG.pcg(A, Pc, r, jnp.zeros_like(r), self.settings.rtol, self.settings.maxit)
        lv = levels[0]
        if isinstance(lv, MG.Coarsest):
            diag = jnp.ones_like(r)
        else:
            diag = jnp.diagonal(jnp.linalg.inv(lv.D_inv), axis1=-2, axis2=-1).astype(r.dtype)
        return x, k, neg, diag

    def newton(self, u, g, depth, tol):
        """The regularised Newton at fixed damage; with ``newton_cap`` a pass ends at the cap or at an iteration that
        finds no descent. Returns (u, iterations, converged, (krylov, shifts))."""
        s = self.settings
        free = self.free
        cap = s.newton_cap if s.newton_cap > 0 else s.max_newton
        stop_stuck = s.newton_cap > 0
        E_ = lambda v: self.total(v, g, depth)
        gradE = jax.grad(E_)
        if s.mixed:
            g32, z32 = g.astype(F32), depth.astype(F32)
            gradE32 = jax.grad(lambda v: self.total32(v, g32, z32))
        zi = jnp.zeros((), jnp.int32)

        def body(c):
            u, k, size, kry, shifts, stuck = c
            r = gradE(u) * free
            K = jax.linearize(gradE, u)[1]
            Kp = jax.linearize(gradE32, u.astype(F32))[1] if s.mixed else None
            x, kk, neg, diag = self._solve(K, r, u, g, Kp)
            kk = kk.astype(jnp.int32)
            du = -x * free
            scale = jnp.sum(jnp.abs(diag) * free) / jnp.maximum(jnp.sum(free), 1.0)

            def not_descent(st):
                tau, d, ng, n, kk = st
                slope = jnp.sum(r * d)
                return (ng | (slope >= -1e-14 * jnp.sqrt(jnp.sum(r * r) * jnp.sum(d * d)))) & (n < 14)

            def bump(st):
                tau, _, _, n, kk = st
                tau = jnp.where(tau == 0, 1e-6, 10.0 * tau)
                Ks = lambda v: K(v) + (tau * scale) * v
                Kps = (lambda v: Kp(v) + (tau * scale).astype(F32) * v) if s.mixed else None
                x2, k2, ng2, _ = self._solve(Ks, r, u, g, Kps)
                return tau, -x2 * free, ng2, n + 1, kk + k2.astype(jnp.int32)

            _, du, _, nb, kk = jax.lax.while_loop(not_descent, bump, (jnp.zeros((), u.dtype), du, neg, zi, kk))
            e0 = E_(u)
            slack = 1e-10 * jnp.abs(e0) + 1e-30

            def worse(st):
                a, e, n = st
                return ((~jnp.isfinite(e)) | (e > e0 + slack)) & (n < 20)

            def halve(st):
                a, _, n = st
                return a / 2, E_(u + (a / 2) * du), n + 1

            a, e_a, _ = jax.lax.while_loop(worse, halve, (jnp.ones((), u.dtype), E_(u + du), 0))
            bad = (~jnp.isfinite(e_a)) | (e_a > e0 + slack)
            a = jnp.where(bad, 0.0, a)
            size = jnp.where(a > 0, a * jnp.max(jnp.abs(du)), jnp.inf)
            return u + a * du, k + 1, size, kry + kk, shifts + nb, bad

        def cond(c):
            go = (c[1] < cap) & (c[2] > tol)
            return (go & ~c[5]) if stop_stuck else go

        init = (u, zi, jnp.asarray(jnp.inf, u.dtype), zi, zi, jnp.zeros((), bool))
        u, k, size, kry, shifts, _ = jax.lax.while_loop(cond, body, init)
        return u, k, size <= tol, (kry, shifts)

    def solve_damage(self, H):
        gr, m = self.grid, self.mat
        Ed = lambda d: S.damage_energy(d[..., 0], H, gr, m.gc, m.l)
        mk = lambda g2, c: (lambda d: S.damage_energy(d[..., 0], c["H"], g2, m.gc, m.l))
        d0 = jnp.zeros(gr.nodes + (1,), H.dtype)
        gradE = jax.grad(Ed)
        K = jax.linearize(gradE, d0)[1]
        if self.settings.mixed:
            free = jnp.ones(gr.nodes + (1,), F32)
            levels, transfers = MG.build_hierarchy(mk, {"H": H.astype(F32)}, d0.astype(F32), gr, free, self.dopts, ("H",))
            P32 = MG.gmg_solver(levels, transfers, self.dopts)
            Pc = lambda v: P32(v.astype(F32)).astype(v.dtype)
        else:
            free = jnp.ones(gr.nodes + (1,), H.dtype)
            levels, transfers = MG.build_hierarchy(mk, {"H": H}, d0, gr, free, self.dopts, ("H",))
            Pc = MG.gmg_solver(levels, transfers, self.dopts)
        b = -gradE(d0)
        x, k, _, _ = MG.pcg(K, Pc, b, jnp.zeros_like(b), self.settings.rtol_damage, self.settings.maxit)
        if self.settings.damage_bound == "clip":
            return jnp.clip(x[..., 0], 0.0, 1.0), k
        return self._damage_bounded(K, b, H, x, k)

    def _damage_bounded(self, K, b, H, x, k):
        """The damage minimiser under 0 <= d <= 1 by a primal-dual active set: nodes outside are held at their bound and
        the rest re-solved; a node held at 1 is released when its multiplier b - K d turns negative, one held at 0 when it
        turns positive; until the sets stop changing (at most 8 rounds). Where nothing leaves [0, 1] this returns the
        unconstrained solve unchanged."""
        gr, m, s = self.grid, self.mat, self.settings
        dt = H.dtype
        mk = lambda g2, c: (lambda d: S.damage_energy(d[..., 0], c["H"], g2, m.gc, m.l))

        def solve(hi, lo):
            fr = (~(hi | lo))[..., None].astype(dt)
            fixed = hi[..., None].astype(dt)
            A = lambda v: fr * K(fr * v) + (1.0 - fr) * v
            rhs = fr * (b - K(fixed)) + fixed
            if s.mixed:
                lv, tr = MG.build_hierarchy(mk, {"H": H.astype(F32)}, jnp.zeros(gr.nodes + (1,), F32), gr,
                                            fr.astype(F32), self.dopts, ("H",))
                P32 = MG.gmg_solver(lv, tr, self.dopts)
                Pc = lambda v: P32(v.astype(F32)).astype(v.dtype)
            else:
                lv, tr = MG.build_hierarchy(mk, {"H": H}, jnp.zeros(gr.nodes + (1,), dt), gr, fr, self.dopts, ("H",))
                Pc = MG.gmg_solver(lv, tr, self.dopts)
            y, kk, _, _ = MG.pcg(A, Pc, rhs, fixed, s.rtol_damage, s.maxit)
            return y, kk

        def body(c):
            x, hi, lo, k, it, _ = c
            lam = (b - K(x))[..., 0]
            held = hi | lo
            hi_new = (hi & (lam >= 0.0)) | (~held & (x[..., 0] > 1.0))
            lo_new = (lo & (lam <= 0.0)) | (~held & (x[..., 0] < 0.0))
            y, kk = solve(hi_new, lo_new)
            return y, hi_new, lo_new, k + kk.astype(k.dtype), it + 1, jnp.any(hi_new != hi) | jnp.any(lo_new != lo)

        def cond(c):
            return c[5] & (c[4] < 8)

        none = jnp.zeros(gr.nodes, bool)
        go = jnp.any(x[..., 0] > 1.0) | jnp.any(x[..., 0] < 0.0)
        x, _, _, k, _, _ = jax.lax.while_loop(cond, body, (x, none, none, k, jnp.zeros((), jnp.int32), go))
        return jnp.clip(x[..., 0], 0.0, 1.0), k

    # ---- the step --------------------------------------------------------------------------------------------
    def initial(self, dtype: Any = jnp.float64) -> State:
        gr = self.grid
        return State(jnp.zeros(gr.nodes + (3,), dtype), jnp.zeros(gr.nodes, dtype),
                     jnp.zeros((2, 2, 2) + gr.shape, dtype), jnp.zeros((), dtype), jnp.zeros(gr.shape, dtype))

    def _tol(self, err):
        s = self.settings
        if s.inexact <= 0:
            return jnp.asarray(self.tol)
        e = jnp.where(jnp.isfinite(err), err, 1.0)
        return self.tol + jnp.minimum(s.inexact * e, 1e-6) * self.block.height

    def _pass(self, u, x, hist0, depth, tol):
        """One staggered pass from the damage iterate ``x``: Newton at g(x), the history max(H_start, psi_plus), the
        damage solve."""
        m = self.mat
        g = S.degradation(x, m.k_res)
        u, n, ok, (kry, shifts) = self.newton(u, g, depth, tol)
        p = S.tension_energy(u, self.grid, m, self.settings.volumetric)
        hist = jnp.maximum(hist0, p)
        d_new, kd = self.solve_damage(hist)
        return u, n, ok, kry, shifts, hist, d_new, kd

    def _step(self, state: State, depth):
        s = self.settings
        m = self.mat
        depth = jnp.asarray(depth, state.u.dtype)
        prev = state.depth
        u0 = state.u * jnp.where(prev > 0, depth / jnp.where(prev > 0, prev, 1.0), 1.0)
        u0 = jnp.where(jnp.isfinite(self.solid(u0, S.degradation(state.d, m.k_res))), u0, state.u)
        limit = s.stagger_cap > 0
        cap = s.stagger_cap if limit else s.max_staggered
        M = max(s.anderson, 1)
        nd = int(np.prod(self.grid.nodes))
        zi = jnp.zeros((), jnp.int32)
        inf = jnp.asarray(jnp.inf, u0.dtype)

        def body(c):
            u, n, ok, kry, shifts, hist, d_new, kd = self._pass(c["u"], c["x"], state.history, depth, self._tol(c["err"]))
            f = d_new - c["x"]
            err = jnp.max(jnp.abs(f))
            if s.trace:
                jax.debug.callback(_pass_line, depth, c["k"] + 1, err, n, ok, kry, shifts)
            if s.anderson > 0:
                xn, aa, rs = aa_update(c["aa"], c["x"].ravel(), d_new.ravel(), f.ravel(), s.anderson)
                xn = xn.reshape(c["x"].shape)
            else:
                xn, aa, rs = d_new, c["aa"], jnp.zeros((), bool)
            return dict(u=u, x=xn, d=d_new, d_used=c["x"], hist=hist, k=c["k"] + 1, nk=c["nk"] + n, err=err,
                        capped=c["capped"] + (~ok).astype(jnp.int32), kry=c["kry"] + kry, shifts=c["shifts"] + shifts,
                        dk=c["dk"] + kd.astype(jnp.int32), conv=err <= s.tol_damage,
                        failed=jnp.asarray(limit) & ~ok, aa=aa, rs=c["rs"] + rs.astype(jnp.int32))

        def cond(c):
            return (c["k"] < cap) & (~c["conv"]) & (~c["failed"])

        init = dict(u=u0, x=state.d, d=state.d, d_used=state.d, hist=state.history, k=zi, nk=zi, err=inf, capped=zi,
                    kry=zi, shifts=zi, dk=zi, conv=jnp.zeros((), bool), failed=jnp.zeros((), bool),
                    aa=aa_init(M, nd, u0.dtype), rs=zi)
        c = jax.lax.while_loop(cond, body, init)
        u, d, d_used = c["u"], c["d"], c["d_used"]
        converged = c["conv"] & ~c["failed"] & (c["capped"] == 0)
        nk, capped, kry, shifts = c["nk"], c["capped"], c["kry"], c["shifts"]
        if s.inexact > 0:  # the last equilibrium to the full tolerance (skipped on a step the limit rejects)
            g = S.degradation(d_used, m.k_res)
            tol_f = jnp.where(converged, jnp.asarray(self.tol, u.dtype), jnp.inf)
            u, n, ok, (k1, k4) = self.newton(u, g, depth, tol_f)
            nk, capped, kry, shifts = nk + n, capped + (~ok).astype(jnp.int32), kry + k1, shifts + k4
            converged = converged & ok
        g = S.degradation(d_used, m.k_res)
        mult = self.multiplier
        force = mult * jax.grad(lambda z: self.contact(u, z))(depth)
        out = StepOut(force=force, energy=mult * self.solid(u, g),
                      crack_energy=mult * S.crack_energy(d, self.grid, m.gc, m.l),
                      contact_energy=mult * self.contact(u, depth), staggered=c["k"], newton=nk, newton_capped=capped,
                      krylov=kry, shifts=shifts, damage_krylov=c["dk"], residual=c["err"], converged=converged,
                      aa_restarts=c["rs"], d_max=jnp.max(d),
                      penetration=jnp.min(self._gap(self.X, u, depth)) / self.block.h)
        return State(u, d, c["hist"], depth, state.deleted), out

    # ---- the deletion control --------------------------------------------------------------------------------
    def _deletion_g(self, deleted):
        return jnp.broadcast_to(degradation(deleted, self.mat.k_res), (2, 2, 2) + deleted.shape)

    def _step_deletion(self, state: State, depth):
        """Newton at the deleted set, then delete every element whose centre's largest principal stretch passes the
        limit (``fracture.deletion``), repeated until none is newly deleted (at most ``stagger_cap`` or
        ``max_staggered`` rounds; a step that is still deleting there is not converged)."""
        s, m = self.settings, self.mat
        depth = jnp.asarray(depth, state.u.dtype)
        prev = state.depth
        u0 = state.u * jnp.where(prev > 0, depth / jnp.where(prev > 0, prev, 1.0), 1.0)
        u0 = jnp.where(jnp.isfinite(self.solid(u0, self._deletion_g(state.deleted))), u0, state.u)
        cap = s.stagger_cap if s.stagger_cap > 0 else s.max_staggered
        zi = jnp.zeros((), jnp.int32)

        def body(c):
            u, n, ok, (kry, shifts) = self.newton(c["u"], self._deletion_g(c["de"]), depth, jnp.asarray(self.tol))
            Fc = S.element_mean(S.deformation(u, self.grid))
            stretch = jnp.sqrt(jnp.linalg.eigvalsh(jnp.swapaxes(Fc, -1, -2) @ Fc)[..., -1])
            new = deletion.newly_deleted(stretch, c["de"], m.stretch_limit)
            return dict(u=u, de=jnp.where(new, 1.0, c["de"]), k=c["k"] + 1, nk=c["nk"] + n, kry=c["kry"] + kry,
                        shifts=c["shifts"] + shifts, capped=c["capped"] + (~ok).astype(jnp.int32), more=jnp.any(new))

        def cond(c):
            return (c["k"] < cap) & c["more"] & (c["capped"] == 0)

        c = jax.lax.while_loop(cond, body, dict(u=u0, de=state.deleted, k=zi, nk=zi, kry=zi, shifts=zi, capped=zi,
                                                more=jnp.ones((), bool)))
        u, de = c["u"], c["de"]
        converged = (~c["more"]) & (c["capped"] == 0)
        nx, ny, nz = self.grid.shape
        pad = jnp.pad(de, 1)
        d = jnp.max(jnp.stack([pad[a:a + nx + 1, b:b + ny + 1, e:e + nz + 1] for a in (0, 1) for b in (0, 1)
                               for e in (0, 1)]), axis=0)
        mult = self.multiplier
        vol = jnp.sum(de * 8.0 * S.gauss_weight(self.grid, u.dtype))
        force = mult * jax.grad(lambda z: self.contact(u, z))(depth)
        out = StepOut(force=force, energy=mult * self.solid(u, self._deletion_g(de)),
                      # no surface energy: the deleted volume over h, times Gc, so that crack_area reads volume / h
                      crack_energy=mult * vol / self.block.h * self.gc_physical,
                      contact_energy=mult * self.contact(u, depth), staggered=c["k"], newton=c["nk"],
                      newton_capped=c["capped"], krylov=c["kry"], shifts=c["shifts"], damage_krylov=zi,
                      residual=jnp.zeros((), u.dtype), converged=converged, aa_restarts=zi, d_max=jnp.max(de),
                      penetration=jnp.min(self._gap(self.X, u, depth)) / self.block.h)
        return State(u, d, state.history, depth, de), out

    # ---- readouts --------------------------------------------------------------------------------------------
    def crack_extent(self, d: np.ndarray, threshold: float = 0.9) -> tuple[float, float, float]:
        """The damaged region's reach (m) from the tip's axis in x and in y, and below the top: the largest |x - axis|,
        |y| and (top - z) over nodes with d >= ``threshold`` (0 where none)."""
        on = np.asarray(d) >= threshold
        if not on.any():
            return 0.0, 0.0, 0.0
        X = np.asarray(self.X)
        return (float(np.max(np.abs(X[..., 0][on] - self.axis[0]))), float(np.max(np.abs(X[..., 1][on]))),
                float(np.max(self.block.height - X[..., 2][on])))


# ------------------------------------------------------------------------------------------------ the surface ---
@dataclass
class Surface:
    """The block's boundary as triangles over the grid's boundary nodes: ``rest`` (V, 3) m, ``faces`` (F, 3),
    ``face_kind`` (F,) one of ``KINDS``, ``node`` (V, 3) the grid index of each vertex; per saved frame ``points``
    (T, V, 3) m and ``damage`` (T, V), at ``depth`` (T,) m. Symmetry faces show the crack inside the block."""

    KINDS = ("top", "bottom", "side", "symmetry")
    rest: np.ndarray
    faces: np.ndarray
    face_kind: np.ndarray
    node: np.ndarray
    symmetry_planes: tuple[str, ...]
    points: list[np.ndarray] = field(default_factory=list)
    damage: list[np.ndarray] = field(default_factory=list)
    depth: list[float] = field(default_factory=list)


def boundary_surface(p: Puncture3D) -> Surface:
    nx, ny, nz = p.grid.nodes
    sym = {"quarter": ("x", "y"), "half": ("y",), "full": ()}[p.block.symmetry]
    index: dict[tuple[int, int, int], int] = {}
    nodes: list[tuple[int, int, int]] = []
    faces: list[tuple[int, int, int]] = []
    kinds: list[int] = []

    def vid(t):
        if t not in index:
            index[t] = len(nodes)
            nodes.append(t)
        return index[t]

    def sheet(fixed_axis, value, kind):
        a, b = [ax for ax in range(3) if ax != fixed_axis]
        na, nb = (nx, ny, nz)[a], (nx, ny, nz)[b]
        for i in range(na - 1):
            for j in range(nb - 1):
                q = []
                for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    t = [0, 0, 0]
                    t[fixed_axis], t[a], t[b] = value, i + di, j + dj
                    q.append(vid(tuple(t)))
                faces.extend([(q[0], q[1], q[2]), (q[0], q[2], q[3])])
                kinds.extend([kind, kind])

    k_of = {k: i for i, k in enumerate(Surface.KINDS)}
    sheet(2, nz - 1, k_of["top"])
    sheet(2, 0, k_of["bottom"])
    sheet(0, 0, k_of["symmetry"] if "x" in sym else k_of["side"])
    sheet(0, nx - 1, k_of["side"])
    sheet(1, 0, k_of["symmetry"] if "y" in sym else k_of["side"])
    sheet(1, ny - 1, k_of["side"])
    node = np.asarray(nodes, int)
    X = np.asarray(p.X)
    return Surface(rest=X[node[:, 0], node[:, 1], node[:, 2]], faces=np.asarray(faces, int), face_kind=np.asarray(kinds),
                   node=node, symmetry_planes=sym)


def mirrored(surface: Surface, keep_symmetry_faces: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The whole block's boundary drawn from a symmetric run, for display: (points (T, V', 3), faces (F', 3), damage
    (T, V'), display_only (F',) True on the reflected copies). Each symmetry plane reflects the surface (x -> -x, y -> -y
    about the tip's axis at 0); the symmetry faces themselves are dropped unless ``keep_symmetry_faces`` (they lie inside
    the whole block)."""
    sym_kind = Surface.KINDS.index("symmetry")
    keep = np.ones(len(surface.faces), bool) if keep_symmetry_faces else surface.face_kind != sym_kind
    faces0 = surface.faces[keep]
    P0 = np.asarray(surface.points)
    D0 = np.asarray(surface.damage)
    flips = [(1.0, 1.0)]
    if "x" in surface.symmetry_planes:
        flips.append((-1.0, 1.0))
    if "y" in surface.symmetry_planes:
        flips += [(fx, -1.0) for fx, _ in list(flips)]
    pts, fcs, dmg, disp = [], [], [], []
    nv = P0.shape[1]
    for i, (fx, fy) in enumerate(flips):
        Pk = P0.copy()
        Pk[..., 0] *= fx
        Pk[..., 1] *= fy
        f = faces0 + i * nv
        if fx * fy < 0:  # a single reflection turns the triangles inside out
            f = f[:, ::-1]
        pts.append(Pk)
        dmg.append(D0)
        fcs.append(f)
        disp.append(np.full(len(f), i > 0))
    return np.concatenate(pts, axis=1), np.concatenate(fcs), np.concatenate(dmg, axis=1), np.concatenate(disp)


def yeoh_material(c1: float, c2: float, gc: float, l: float, bulk_ratio: float = 50.0, k_res: float = 1e-6) -> Material:
    """A nearly incompressible Yeoh solid for the 3-D build: mu = 2 C1, the Yeoh C2 as ``c2``, kappa = ``bulk_ratio`` mu
    (50: Poisson's ratio about 0.49), ``gc`` the toughness to dissipate (J/m^2), ``l`` the phase field's length (m)."""
    mu = 2.0 * c1
    return Material(mu=mu, kappa=bulk_ratio * mu, gc=gc, l=l, k_res=k_res, stretch_limit=10.0, c2=c2)


# --------------------------------------------------------------------------------------------------- the run ---
@dataclass
class PunctureRun:
    """What a run returns: ``rows`` (one entry a converged load step, arrays keyed by name: ``depth_m``, ``force_N``,
    ``crack_area_m2``, ``crack_energy_J``, ``energy_J``, ``contact_energy_J``, ``d_max``, ``extent_x_m``,
    ``extent_y_m``, ``extent_z_m``, and the step's counts), ``surface`` (the block's boundary at the saved frames),
    ``counts`` (the run's totals), ``meta`` (the grid, the settings, the case)."""

    rows: dict[str, np.ndarray]
    surface: Surface
    counts: dict[str, Any]
    meta: dict[str, Any]

    def energy_balance(self, floor: float = 0.05) -> float:
        """max over steps of |W - (stored + crack + contact)| / W, W the work of the tip force (trapezoid from 0), over the
        steps where W is at least ``floor`` of the run's largest W (the first contacts' tiny W would make any rounding a
        large ratio)."""
        z, f = self.rows["depth_m"], self.rows["force_N"]
        if len(z) == 0:
            return float("nan")
        zz = np.concatenate([[0.0], z])
        ff = np.concatenate([[0.0], f])
        W = np.cumsum(0.5 * (ff[1:] + ff[:-1]) * np.diff(zz))
        E = self.rows["energy_J"] + self.rows["crack_energy_J"] + self.rows["contact_energy_J"]
        ok = W >= floor * np.max(W)
        return float(np.max(np.abs(W[ok] - E[ok]) / W[ok])) if ok.any() and np.max(W) > 0 else float("nan")

    def peak(self) -> tuple[float, float]:
        """(the largest force N, its depth m)."""
        i = int(np.argmax(self.rows["force_N"]))
        return float(self.rows["force_N"][i]), float(self.rows["depth_m"][i])

    def save(self, path: str) -> None:
        """``path``.npz (rows and surface arrays) and ``path``.json (counts and meta)."""
        import json
        sf = self.surface
        np.savez_compressed(path + ".npz", **{f"row_{k}": v for k, v in self.rows.items()}, rest=sf.rest,
                            faces=sf.faces, face_kind=sf.face_kind, node=sf.node,
                            points=np.asarray(sf.points, np.float32).reshape(-1, *sf.rest.shape),
                            damage=np.asarray(sf.damage, np.float32).reshape(-1, sf.rest.shape[0]),
                            frame_depth=np.asarray(sf.depth))
        with open(path + ".json", "w") as fh:
            json.dump({"counts": self.counts, "meta": self.meta, "symmetry_planes": list(sf.symmetry_planes)}, fh,
                      indent=1, default=float)


def load_run(path: str) -> PunctureRun:
    import json
    z = np.load(path + ".npz")
    j = json.load(open(path + ".json"))
    rows = {k[4:]: z[k] for k in z.files if k.startswith("row_")}
    sf = Surface(rest=z["rest"], faces=z["faces"], face_kind=z["face_kind"], node=z["node"],
                 symmetry_planes=tuple(j["symmetry_planes"]), points=list(z["points"]), damage=list(z["damage"]),
                 depth=list(z["frame_depth"]))
    return PunctureRun(rows, sf, j["counts"], j["meta"])


ROW_KEYS = ("depth_m", "force_N", "crack_area_m2", "crack_energy_J", "energy_J", "contact_energy_J", "d_max",
            "extent_x_m", "extent_y_m", "extent_z_m", "staggered", "newton", "krylov", "shifts", "damage_krylov",
            "aa_restarts", "penetration_over_h", "step_s")


def run_puncture(block: Block, material: Material, tip: ct.Indenter, depth: float, settings: Settings = Settings(),
                 frames: int = 24, wall: float | None = None, output_depths: Sequence[float] = (),
                 on_step: Callable[[dict[str, float]], None] | None = None) -> PunctureRun:
    """Press ``tip`` into ``block`` to ``depth`` (m) and return the run (``PunctureRun``). ``material.gc`` is the
    toughness to dissipate (J/m^2). ``frames``: about how many surface frames to keep, evenly in depth (always the last).
    ``wall``: s; no new step starts after it. ``output_depths``: depths (m) the adaptive steps land on. ``on_step``
    receives each accepted row (a progress hook)."""
    p = Puncture3D(block, material, tip, settings)
    s = settings
    surf = boundary_surface(p)
    node = surf.node
    st = p.initial()
    rows: dict[str, list[float]] = {k: [] for k in ROW_KEYS}
    rejected: list[dict[str, float]] = []
    t0 = time.time()
    status = "depth reached"
    z = 0.0
    dz = s.first_step if s.adaptive else s.step
    next_frame = 0.0
    frame_every = depth / max(frames, 1)
    stops = sorted(set(list(output_depths) + [depth]))

    def keep_frame(state: State, at: float) -> None:
        u = np.asarray(state.u)
        surf.points.append((surf.rest + u[node[:, 0], node[:, 1], node[:, 2]]).astype(np.float32))
        surf.damage.append(np.asarray(state.d)[node[:, 0], node[:, 1], node[:, 2]].astype(np.float32))
        surf.depth.append(at)

    keep_frame(st, 0.0)
    while z < depth - 1e-12:
        target = min(z + dz, depth)
        for od in stops:
            if z + 1e-12 < od < target - 1e-12:
                target = od
        ts = time.time()
        st_new, o = p.step(st, jnp.asarray(target))
        o = jax.device_get(o)
        ok = bool(o.converged) and bool(np.isfinite(o.force))
        if not ok:
            rejected.append({"depth_m": target, "staggered": int(o.staggered), "newton": int(o.newton),
                             "step_s": time.time() - ts})
            half = (target - z) / 2
            if half < s.min_step - 1e-15:
                status = "stopped: a step below the minimum did not converge"
                break
            dz = half
            if wall is not None and time.time() - t0 > wall:
                status = "wall time"
                break
            continue
        st = st_new
        z = target
        if s.adaptive:
            if int(o.staggered) <= s.grow_at:
                dz = min(dz * 1.5, s.step)
            elif int(o.staggered) >= s.shrink_at:
                dz = max(dz / 2, s.min_step)
        d_np = np.asarray(st.d)
        ex, ey, ez = p.crack_extent(d_np)
        row = {"depth_m": z, "force_N": float(o.force), "crack_area_m2": float(o.crack_energy) / p.gc_physical,
               "crack_energy_J": float(o.crack_energy), "energy_J": float(o.energy),
               "contact_energy_J": float(o.contact_energy), "d_max": float(o.d_max), "extent_x_m": ex,
               "extent_y_m": ey, "extent_z_m": ez, "staggered": int(o.staggered), "newton": int(o.newton),
               "krylov": int(o.krylov), "shifts": int(o.shifts), "damage_krylov": int(o.damage_krylov),
               "aa_restarts": int(o.aa_restarts), "penetration_over_h": float(o.penetration),
               "step_s": time.time() - ts}
        for k in ROW_KEYS:
            rows[k].append(row[k])
        if on_step is not None:
            on_step(row)
        if z >= next_frame + frame_every - 1e-12 or z >= depth - 1e-12:
            keep_frame(st, z)
            next_frame = z
        if wall is not None and time.time() - t0 > wall:
            status = "wall time"
            break
    if surf.depth[-1] != z:
        keep_frame(st, z)
    R = {k: np.asarray(v) for k, v in rows.items()}
    counts = {"steps": len(R["depth_m"]), "rejected": len(rejected), "staggered": int(R["staggered"].sum()),
              "newton": int(R["newton"].sum()), "krylov": int(R["krylov"].sum()), "shifts": int(R["shifts"].sum()),
              "damage_krylov": int(R["damage_krylov"].sum()), "wall_s": time.time() - t0, "status": status,
              "rejected_rows": rejected, "unknowns": p.unknowns, "cells": list(p.grid.shape),
              "jax": jax.__version__, "devices": [str(dv) for dv in jax.devices()]}
    meta = {"block": asdict(block), "settings": asdict(settings), "tip": {"kind": p.tip.kind, **{k: float(v) for k, v in
            p.tip.geometry().items()}}, "material": {k: v for k, v in p.mat._asdict().items() if k != "fibre"},
            "gc_physical": p.gc_physical, "penalty": p.kpen, "multiplier": p.multiplier, "depth_m": depth,
            "x_m": p.grid.x.tolist(), "y_m": p.grid.y.tolist(), "z_m": p.grid.z.tolist()}
    run = PunctureRun(R, surf, counts, meta)
    run.counts["energy_balance"] = run.energy_balance()
    return run
