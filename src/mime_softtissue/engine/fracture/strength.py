"""A strength surface in the phase field: crack nucleation set by the material's strength, separately from the
regularisation length l, with Gc for the growth of a crack (Kumar, Bourdin, Francfort and Lopez-Pamies, J. Mech. Phys.
Solids 142 (2020) 104027), in the large-deformation, fully explicit form of Kamarei, Kumar and Lopez-Pamies, J. Mech.
Phys. Solids 188 (2024) 105683 (arXiv 2402.06785), eqs. (2), (10), (15), (16), (20) and (26).

The damage d = 1 - z minimises, for a fixed history ``H`` (psi_plus's maximum over the path) and a driving force
``c_e`` frozen at the staggered iterate,

    sum w [ (1 - d)^2 H [I1 >= 0] + c_e d + (3 delta Gc / 8) (d / l + l |grad d|^2) + eta/2 (d - d_n)^2 ],

over d_n <= d <= 1 (irreversible). That is the 2024 paper's phase-field equation (20) written as a minimisation: its
driving force's ``z (1 - sqrt(I1^2) / I1) W`` term, which cancels the elastic driving under a compressive Biot stress,
is the indicator ``[I1 >= 0]`` here, and ``c_e = beta2 sqrt(J2) + beta1 I1`` of the degraded Biot stress (its
principal values the derivatives of the degraded energy in the principal stretches). The strength surface is
Drucker-Prager in the Biot stress, set by the uniaxial tensile strength ``s_ts`` and the hydrostatic strength ``s_hs``:
sqrt(J2) + gamma1 I1 + gamma0 = 0, gamma1 = s_ts / (sqrt3 (3 s_hs - s_ts)), gamma0 = -sqrt3 s_hs s_ts / (3 s_hs - s_ts).

Differences from the papers, declared: c_e's Biot stress is degraded by the load step's starting damage, frozen over
the staggered iterations (degraded by the current iterate, a uniform bar's damage swings between two values and the
iteration never settles; the two agree until damage starts and as the step shrinks); the elastic driving is this
engine's history field (the papers use the current
energy; with a monotone load the two agree until damage starts); the energy split is this engine's (volumetric
compression is not degraded; the papers degrade all of W); an active set enforces the bounds (the 2024 paper uses a
penalty). ``coefficients`` computes W_ts and W_hs for this engine's own law (neo-Hookean or Gent, compressible), so the
uniaxial-tension and hydrostatic points of the surface hold exactly at any l and h."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .. import material as law
from ..assembly import Assembly, constrain
from ..grid import Grid
from ..kinematics.plane_strain import N_GAUSS, gauss_gradients
from ..solvers import blocktri


class Strength(NamedTuple):
    """The strength surface's numbers a step reads (a pytree): the strengths (Pa, Biot), the coefficient delta and
    the driving force's beta1 (1/1), beta2 (1/1); ``coefficients`` fills them from the strengths."""

    s_ts: Any
    s_hs: Any
    delta: Any
    beta1: Any
    beta2: Any


class Tension(NamedTuple):
    """A tension-only strength surface (a pytree, an option beside ``Strength``): the largest principal Biot stress
    reaching ``s_ts`` (Pa), F(S) = max(S_i) - s_ts, so that no state whose principal stresses are all <= 0 (uniaxial
    or confined compression) ever reaches it. ``tension_coefficients`` fills ``delta`` and the driving force's
    ``beta`` (1/1): c_e = beta max(S_i)."""

    s_ts: Any
    delta: Any
    beta: Any
    gate: Any = 0.0  # 0: the elastic driving acts where I1 >= 0 (the papers'); 1: where max(S_i) > 1e-3 s_ts (crack opening)


def _energy(stretches: jax.Array, dq: Any, mat: law.Material) -> jax.Array:
    """The degraded energy at principal ``stretches`` (3,): g(d) psi_plus + psi_minus."""
    J = stretches[0] * stretches[1] * stretches[2]
    I1 = jnp.sum(stretches * stretches)
    plus, minus = law.energy_parts(J, I1, (), mat)
    return law.degradation(dq, mat.k_res) * plus + minus


def biot_principal(stretches: jax.Array, dq: Any, mat: law.Material) -> jax.Array:
    """The degraded Biot stress's principal values at principal ``stretches`` (3,): d(energy)/d(stretch)."""
    return jax.grad(_energy)(stretches, dq, mat)


def principal_stretches(F2: jax.Array, Ftt: jax.Array) -> jax.Array:
    """(3,) principal stretches of the block-diagonal deformation [[F2, 0], [0, Ftt]] (axisymmetric: Ftt the hoop
    stretch; plane strain: Ftt = 1)."""
    C = F2.T @ F2
    half = 0.5 * (C[0, 0] + C[1, 1])
    det = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    root = jnp.sqrt(jnp.maximum(half * half - det, 0.0))
    return jnp.stack([jnp.sqrt(jnp.maximum(half + root, 1e-300)), jnp.sqrt(jnp.maximum(half - root, 1e-300)), Ftt])


def driving(s: jax.Array, st: Strength | Tension) -> tuple[jax.Array, jax.Array]:
    """(c_e, tension) at one point from the Biot stress's principal values ``s`` (3,): c_e = beta2 sqrt(J2) + beta1 I1
    (``Strength``, Drucker-Prager) or beta max(S_i) (``Tension``); tension: I1 >= 0."""
    if isinstance(st, Tension):
        opening = jnp.max(s) > 1e-3 * st.s_ts
        return st.beta * jnp.max(s), jnp.where(st.gate > 0.5, opening, jnp.sum(s) >= 0.0)
    I1 = jnp.sum(s)
    J2 = ((s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2) / 6.0
    return st.beta2 * jnp.sqrt(jnp.maximum(J2, 0.0)) + st.beta1 * I1, I1 >= 0.0


def surface(s: Any, s_ts: float, s_hs: float) -> Any:
    """The Drucker-Prager function F(S) at Biot principal values ``s`` (..., 3): >= 0 violates the strength."""
    s = np.asarray(s)
    I1 = s.sum(-1)
    J2 = ((s[..., 0] - s[..., 1]) ** 2 + (s[..., 1] - s[..., 2]) ** 2 + (s[..., 2] - s[..., 0]) ** 2) / 6.0
    g1 = s_ts / (np.sqrt(3.0) * (3 * s_hs - s_ts))
    g0 = -np.sqrt(3.0) * s_hs * s_ts / (3 * s_hs - s_ts)
    return np.sqrt(J2) + g1 * I1 + g0


def _bisect(f, lo: float, hi: float, tol: float = 1e-13) -> float:
    flo = f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo < tol * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


def uniaxial_state(mat: law.Material, s: float) -> tuple[float, float, float]:
    """(stretch, lateral stretch, W) of uniform uniaxial tension whose Biot stress is ``s`` (the lateral stress
    zero): the root nearest the undeformed state."""
    zero = jnp.zeros(())

    def lateral(lam: float) -> float:
        g = lambda ll: float(biot_principal(jnp.asarray([lam, ll, ll]), zero, mat)[1])
        return _bisect(g, 0.05, 1.0 + 1e-12)

    def axial(lam: float) -> float:
        ll = lateral(lam)
        return float(biot_principal(jnp.asarray([lam, ll, ll]), zero, mat)[0]) - s

    hi = 1.5
    while axial(hi) < 0:
        hi *= 2.0
        if hi > 1e6:
            raise ValueError("no uniaxial state reaches the strength")
    lam = _bisect(axial, 1.0, hi)
    ll = lateral(lam)
    return lam, ll, float(_energy(jnp.asarray([lam, ll, ll]), zero, mat))


def hydrostatic_state(mat: law.Material, s: float) -> tuple[float, float]:
    """(stretch, W) of uniform hydrostatic tension whose Biot stress is ``s``."""
    zero = jnp.zeros(())
    f = lambda lam: float(biot_principal(jnp.asarray([lam, lam, lam]), zero, mat)[0]) - s
    hi = 1.5
    while f(hi) < 0:
        hi *= 2.0
        if hi > 1e6:
            raise ValueError("no hydrostatic state reaches the strength")
    lam = _bisect(f, 1.0, hi)
    return lam, float(_energy(jnp.asarray([lam, lam, lam]), zero, mat))


def coefficients(mat: law.Material, s_ts: float, s_hs: float, h: float, gc: float | None = None,
                 l: float | None = None, delta_factor: float = 1.0) -> Strength:
    """The strength surface's numbers for the law ``mat`` (its mu, kappa, Gent limit; its gc and l unless given),
    the strengths ``s_ts`` and ``s_hs`` (Biot, Pa; 3 s_hs > s_ts) and the cell size ``h``: delta by the 2024 paper's
    eq. (26) (with its first-order element correction) times ``delta_factor`` (a calibration, as the 2020 paper
    calibrates delta against a pre-cracked test), beta1 and beta2 by its eq. (16) from that delta."""
    if not 3.0 * s_hs > s_ts:
        raise ValueError("the Drucker-Prager surface needs 3 s_hs > s_ts")
    if mat.fibre is not None:
        raise ValueError("the strength surface reads an isotropic law (no fibre families)")
    gc = float(mat.gc if gc is None else gc)
    l = float(mat.l if l is None else l)
    elastic = mat._replace(k_res=0.0)
    _, _, w_ts = uniaxial_state(elastic, s_ts)
    _, w_hs = hydrostatic_state(elastic, s_hs)
    r3 = np.sqrt(3.0)
    corr = 1.0 + 3.0 * h / (8.0 * l)
    delta = (corr ** -2 * (s_ts + (1.0 + 2.0 * r3) * s_hs) / ((8.0 + 3.0 * r3) * s_hs) * 3.0 * gc / (16.0 * w_ts * l)
             + 0.4 / corr) * delta_factor
    q = delta * gc / (8.0 * l)
    beta1 = -q / s_hs + 2.0 * w_hs / (3.0 * s_hs)
    beta2 = -r3 * (3.0 * s_hs - s_ts) / (s_hs * s_ts) * q + 2.0 * r3 * w_ts / s_ts - 2.0 * w_hs / (r3 * s_hs)
    return Strength(s_ts=float(s_ts), s_hs=float(s_hs), delta=float(delta), beta1=float(beta1), beta2=float(beta2))


def tension_coefficients(mat: law.Material, s_ts: float, h: float, gc: float | None = None, l: float | None = None,
                         delta_factor: float = 1.0, gate: str = "i1") -> Tension:
    """The tension-only surface's numbers for the law ``mat``, the strength ``s_ts`` (Biot, Pa) and the cell size ``h``.
    The uniform onset 2 W [I1 >= 0] - c_e - 3 q (q = delta Gc / 8 l) with c_e = beta max(S_i) and
    beta = (2 W_ts - 3 q) / s_ts is zero exactly at uniaxial tension at s_ts, at any l; under all-compressive principal
    stresses it is -3 q + |beta| max(S_i) <= -3 q < 0, so compression never nucleates. delta is the 2024 paper's eq.
    (26) for the Drucker-Prager surface through the same uniaxial and hydrostatic points (s_hs = s_ts: a principal-
    stress criterion fails hydrostatic tension at s_ts), times ``delta_factor`` (a calibration against a pre-cracked
    test). Raises where beta >= 0 (l past the length at which 3 q = 2 W_ts: the surface no longer drives by stress).
    ``gate``: where the elastic energy drives the damage: "i1" where the Biot stress's trace is >= 0 (the papers' indicator),
    "s1" where its largest principal value is tensile (above 1e-3 s_ts), so a crack opened by in-plane tension under a
    large pressure (the shell over a needle's tip) can grow by Gc; uniaxial and confined compression stay excluded."""
    if mat.fibre is not None:
        raise ValueError("the strength surface reads an isotropic law (no fibre families)")
    gc = float(mat.gc if gc is None else gc)
    l = float(mat.l if l is None else l)
    _, _, w_ts = uniaxial_state(mat._replace(k_res=0.0), s_ts)
    r3 = np.sqrt(3.0)
    corr = 1.0 + 3.0 * h / (8.0 * l)
    delta = (corr ** -2 * (2.0 + 2.0 * r3) / (8.0 + 3.0 * r3) * 3.0 * gc / (16.0 * w_ts * l) + 0.4 / corr) * delta_factor
    q = delta * gc / (8.0 * l)
    beta = (2.0 * w_ts - 3.0 * q) / s_ts
    if not beta < 0.0:
        raise ValueError(f"l = {l} is past the tension-only surface's length (3 q <= 2 W_ts; beta = {beta})")
    if gate not in ("i1", "s1"):
        raise ValueError(f"gate is 'i1' or 's1', not {gate!r}")
    return Tension(s_ts=float(s_ts), delta=float(delta), beta=float(beta), gate=1.0 if gate == "s1" else 0.0)


def homogeneous_onset(mat: law.Material, st: Strength | Tension, stretches: Any) -> Any:
    """The phase field's uniform onset function at undamaged principal ``stretches`` (..., 3): 2 W [I1 >= 0] - c_e -
    3 delta Gc / (8 l); damage starts where it turns positive (the paper's eq. (20) at a uniform state)."""
    zero = jnp.zeros(())
    lam = jnp.asarray(stretches).reshape(-1, 3)

    def one(x):
        s = biot_principal(x, zero, mat)
        ce, tension = driving(s, st)
        return 2.0 * jnp.where(tension, _energy(x, zero, mat), 0.0) - ce - 3.0 * st.delta * mat.gc / (8.0 * mat.l)

    return np.asarray(jax.vmap(one)(lam)).reshape(np.shape(stretches)[:-1])


def solve_damage(history: jax.Array, ce: jax.Array, tension: jax.Array, gc: Any, l: Any, delta: Any, dx: jax.Array,
                 dy: jax.Array, grid: Grid, assembly: Assembly, I: jax.Array, J: jax.Array,
                 weights: jax.Array | None, viscosity: float, d_prev: jax.Array, iterations: int = 40) -> jax.Array:
    """The damage minimising the module's functional for ``history``, ``ce`` and ``tension`` (n_el, 4) over
    ``d_prev`` <= d <= 1, by a primal-dual active set (each pass one block-tridiagonal solve)."""
    N = jnp.asarray(N_GAUSS)
    G = jax.vmap(gauss_gradients)(dx, dy)
    wq = jnp.broadcast_to((dx * dy / 4.0)[:, None], history.shape) if weights is None else weights
    hq = jnp.where(tension, history, 0.0)
    coef = wq * (2.0 * hq + viscosity)
    ke = jnp.einsum("eg,ga,gb->eab", coef, N, N) + 0.75 * delta * gc * l * jnp.einsum("eg,egai,egbi->eab", wq, G, G)
    dprev_q = jnp.asarray(d_prev)[I, J] @ N.T
    src = wq * (2.0 * hq - ce - 3.0 * delta * gc / (8.0 * l) + viscosity * dprev_q)
    shape = (grid.nx + 1, grid.ny + 1)
    f = jnp.zeros(shape, dtype=history.dtype).at[I, J].add(jnp.einsum("eg,ga->ea", src, N))
    diag, upper = assembly.blocks(ke)
    lower = jnp.asarray(d_prev)
    scale = 1.0 / jnp.maximum(jax.vmap(jnp.diag)(diag), 1e-300)

    def solve_with(lo_set, up_set):
        fixed = lo_set | up_set
        val = jnp.where(lo_set, lower, jnp.where(up_set, 1.0, 0.0))
        r = f - blocktri.matvec(diag, upper, val)
        r = jnp.where(fixed, val, r)
        dg, ug = constrain(diag, upper, (~fixed).astype(diag.dtype))
        return blocktri.solve(dg, ug, r)

    def sets(d):
        g = blocktri.matvec(diag, upper, d) - f  # the functional's gradient; >= 0 on an active lower bound
        trial = d - scale * g
        return trial < lower, trial > 1.0

    def body(c):
        d, lo_set, up_set, k, _ = c
        d_new = solve_with(lo_set, up_set)
        lo_new, up_new = sets(d_new)
        changed = jnp.any(lo_new != lo_set) | jnp.any(up_new != up_set)
        return d_new, lo_new, up_new, k + 1, changed

    def cond(c):
        return (c[3] < iterations) & c[4]

    d0 = jnp.clip(blocktri.solve(diag, upper, f), lower, 1.0)
    lo0, up0 = sets(d0)
    d, _, _, _, _ = jax.lax.while_loop(cond, body, (d0, lo0, up0, 0, jnp.asarray(True)))
    return jnp.clip(d, lower, 1.0)


def crack_energy(d: jax.Array, gc: Any, l: Any, dx: jax.Array, dy: jax.Array, I: jax.Array, J: jax.Array,
                 weights: jax.Array | None = None) -> jax.Array:
    """Gc times the AT1 crack-area functional (3/8) (d/l + l |grad d|^2) (so that crack_energy / Gc is the area)."""
    de = d[I, J]
    dq = de @ jnp.asarray(N_GAUSS).T
    G = jax.vmap(gauss_gradients)(dx, dy)
    grad = jnp.einsum("ea,egai->egi", de, G)
    w = (dx * dy / 4.0)[:, None] if weights is None else weights
    return jnp.sum(w * gc * 0.375 * (dq / l + l * jnp.sum(grad ** 2, -1)))
