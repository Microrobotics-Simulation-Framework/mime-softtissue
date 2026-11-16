"""Quasi-static loading of a specimen with fracture, in pure JAX: each load step a staggered solve, the displacement
by Newton (``solvers.newton``) and the damage by one of three modes:

- ``"phase_field"``: AT2 (``fracture.phase_field``), staggered until the damage moves less than ``tol_d``.
- ``"local"``: the control (``fracture.deletion``): no length scale; repeated until no element is newly deleted.
- ``"elastic"``: no damage.

The kinematics is the specimen's (``kinematics.plane_strain``, in plane strain or incompressible plane stress, or
``kinematics.axisymmetric``); a specimen with a
``contact`` is pressed by a rigid indenter (``engine.contact``) whose depth is the load's ``delta``. ``viscosity``
adds eta/2 (d - d_n)^2 to the damage functional (a viscous regularisation, off by default).

A material with Prony terms (``Material.prony``) is viscoelastic: the state carries the branches' internal strains
(``State.gamma``, per Gauss point as the history), each step minimises the energy plus the branches' dissipation
potential over the step's time and then updates the internal strains in closed form (``material``). The step's time is
``dt`` when given (a hold), otherwise the load's increment over ``Solver.rate`` (the grip's speed in tension, the
insertion speed in puncture).

Every function is differentiable in forward mode (``jax.jvp`` through the loops); the reaction is the internal force
summed over the specimen's reaction nodes, or the indenter's. ``Solver.advance`` is the traceable step (a MIME node
calls it inside its own jit); ``Solver.step`` is its jitted form."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import contact as ct
from .assembly import Assembly, constrain
from . import material as law
from .fracture import deletion, phase_field
from .kinematics import axisymmetric as axi
from .kinematics.plane_strain import (centre_stretch, element_cauchy_green, element_hessian, element_plus,
                                      element_strain_energy)
from .material import Material, ViscousStep
from .solvers import bfgs, newton
from .specimen import Specimen, apply_load, morph_columns, reaction_force

MODES = ("phase_field", "local", "elastic")


class State(NamedTuple):
    u: jax.Array  # (nx+1, ny+1, 2)
    d: jax.Array  # (nx+1, ny+1) nodal (phase field) or (n_el,) element (local)
    history: jax.Array  # (n_el, 4) psi_plus's maximum so far (phase field)
    delta: jax.Array  # the load's displacement (total)
    gamma: Any = None  # (n_el, 4, n_branches, 6) the viscous branches' internal strains (None: elastic)


class StepOut(NamedTuple):
    force: jax.Array  # N, the whole specimen (thickness included)
    energy: jax.Array  # J, stored, the whole specimen
    crack_energy: jax.Array  # J, the phase field's surface term, the whole specimen
    iterations: jax.Array  # staggered iterations
    newton: jax.Array  # Newton iterations, summed over the staggered ones
    residual: jax.Array  # the last staggered change in damage (phase field) or newly deleted count (local)
    newton_capped: jax.Array  # Newton solves of this step that stopped at their cap (not converged)
    contact_energy: jax.Array  # J, the penalty's stored energy
    friction_work: jax.Array  # J, the friction dissipated over the step
    viscous_work: jax.Array  # J, the viscous damage term's dissipation over the step
    force_normal: jax.Array  # N, the indenter's force from the penalty alone (the rest is the shear: friction, adhesion)
    branch_dissipation: jax.Array  # J, the viscous branches' dissipation over the step (0 when elastic)
    dt: jax.Array  # s, the step's time (0 when elastic)


@dataclass
class Solver:
    """The step of one specimen in one ``mode``. ``mat`` and ``notch`` are passed to each call so derivatives reach
    them."""

    specimen: Specimen
    mode: str
    tol_d: float = 1e-6
    max_staggered: int = 400
    max_newton: int = 40
    viscosity: float = 0.0
    tol_p: float = 1e-3  # friction: the lagged pressure's largest change, relative to its largest value
    scheme: str = "staggered"  # the phase field's step: "staggered", or "bfgs" (monolithic, solvers.bfgs)
    newton_regularise: bool = False  # shift an indefinite tangent until Newton's step descends (solvers.newton)
    #: Friction: the lagged pressure moved this share of the way to the new one each staggered iteration (1: all the
    #: way, round 3's). The fixed point is the same; under 1 it damps an iteration that swings between two pressures.
    lag_relaxation: float = 1.0
    #: Newton's stagnation stop (m; 0 off): ``solvers.newton.minimise``'s ``stall_step``.
    newton_stall: float = 0.0
    #: The load's speed (m/s of ``delta``): a viscoelastic material's step takes the time |d delta| / rate.
    rate: float | None = None
    grid: Any = field(init=False)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode is one of {MODES}")
        sp = self.specimen
        self.grid = g = sp.grid
        self.axi = sp.kinematics == "axisymmetric"
        self.stress = sp.kinematics == "plane_stress"
        self.I, self.J = (jnp.asarray(a) for a in g.element_nodes())
        dx, dy = g.element_sizes()
        self.dx, self.dy = jnp.asarray(dx), jnp.asarray(dy)
        self.x0 = jnp.asarray(g.x[:-1][g.element_columns()[0]])
        X, Y = np.meshgrid(g.x, g.y, indexing="ij")
        self.X = jnp.asarray(np.stack([X, Y], axis=-1))
        self.free = jnp.asarray(sp.free)
        self.free_blocks = self.free.reshape(g.nx + 1, -1)
        self.vec = Assembly(g, 2)
        self.sca = Assembly(g, 1)
        self.y_frac = jnp.asarray(g.y / g.y[-1])
        self.contact = sp.contact
        self.step = jax.jit(self.advance)

    # --- pieces -------------------------------------------------------------------------------------------------
    def sizes(self, notch: Any = None) -> tuple[jax.Array, jax.Array]:
        """The elements' sizes: the grid's, or with ``notch`` the columns morphed to that notch length."""
        morph = self.specimen.morph
        if notch is None or morph is None:
            return self.dx, self.dy
        x = morph_columns(self.grid.x, morph, notch)
        return jnp.diff(x)[np.asarray(self.grid.element_columns()[0])], self.dy

    def gauss_weights(self) -> jax.Array | None:
        """The axisymmetric build's Gauss weights (n_el, 4); None for plane strain (the element's dx dy / 4)."""
        if not self.axi:
            return None
        return jax.vmap(axi.gauss_weights)(self.x0, self.dx, self.dy)

    def energy(self, u: jax.Array, dq: jax.Array, mat: Material, notch: Any = None,
               vis: ViscousStep | None = None) -> jax.Array:
        dx, dy = self.sizes(notch)
        if vis is None:
            if self.axi:
                e = jax.vmap(lambda ue, d, x0, dx, dy: axi.element_strain_energy(ue, d, x0, dx, dy, mat))(
                    u[self.I, self.J], dq, self.x0, dx, dy)
            else:
                e = jax.vmap(lambda ue, d, dx, dy: element_strain_energy(ue, d, dx, dy, mat, self.stress))(
                    u[self.I, self.J], dq, dx, dy)
            return jnp.sum(e)
        if self.axi:
            e = jax.vmap(lambda ue, d, x0, dx, dy, gm: axi.element_strain_energy(
                ue, d, x0, dx, dy, mat, ViscousStep(gm, vis.dt)))(u[self.I, self.J], dq, self.x0, dx, dy, vis.gamma)
        else:
            e = jax.vmap(lambda ue, d, dx, dy, gm: element_strain_energy(
                ue, d, dx, dy, mat, self.stress, ViscousStep(gm, vis.dt)))(u[self.I, self.J], dq, dx, dy, vis.gamma)
        return jnp.sum(e)

    def element_hessians(self, u: jax.Array, dq: jax.Array, mat: Material, notch: Any = None,
                         vis: ViscousStep | None = None) -> jax.Array:
        dx, dy = self.sizes(notch)
        if vis is None:
            if self.axi:
                return jax.vmap(lambda ue, d, x0, dx, dy: axi.element_hessian(ue, d, x0, dx, dy, mat))(
                    u[self.I, self.J], dq, self.x0, dx, dy)
            return jax.vmap(lambda ue, d, dx, dy: element_hessian(ue, d, dx, dy, mat, self.stress))(
                u[self.I, self.J], dq, dx, dy)
        if self.axi:
            return jax.vmap(lambda ue, d, x0, dx, dy, gm: axi.element_hessian(
                ue, d, x0, dx, dy, mat, ViscousStep(gm, vis.dt)))(u[self.I, self.J], dq, self.x0, dx, dy, vis.gamma)
        return jax.vmap(lambda ue, d, dx, dy, gm: element_hessian(
            ue, d, dx, dy, mat, self.stress, ViscousStep(gm, vis.dt)))(u[self.I, self.J], dq, dx, dy, vis.gamma)

    def cauchy_green(self, u: jax.Array, notch: Any = None) -> jax.Array:
        """C in Voigt order at every Gauss point: (n_el, 4, 6)."""
        dx, dy = self.sizes(notch)
        if self.axi:
            return jax.vmap(axi.element_cauchy_green)(u[self.I, self.J], self.x0, dx, dy)
        return jax.vmap(lambda ue, dx, dy: element_cauchy_green(ue, dx, dy, self.stress))(u[self.I, self.J], dx, dy)

    def weights(self, notch: Any = None) -> jax.Array:
        """The Gauss weights (n_el, 4): the element's dx dy / 4, or the ring's."""
        dx, dy = self.sizes(notch)
        wq = self.gauss_weights()
        return jnp.broadcast_to((dx * dy / 4.0)[:, None], (self.grid.n_elements, 4)) if wq is None else wq

    def contact_energy(self, u: jax.Array, depth: jax.Array, lag: ct.Lag | None, c: Any = None) -> jax.Array:
        c = self.contact if c is None else c
        e = ct.penalty_energy(c, self.X, u, depth)
        if c.sliding and lag is not None:
            e = e + ct.friction_energy(c, self.X, u, depth, lag)
        return e

    def newton(self, u: jax.Array, dq: jax.Array, mat: Material, notch: Any = None, depth: Any = None,
               lag: ct.Lag | None = None, c: Any = None,
               vis: ViscousStep | None = None) -> tuple[jax.Array, jax.Array, jax.Array]:
        c = self.contact if c is None else c

        def hessian_blocks(u):
            ke = self.element_hessians(u, dq, mat, notch, vis)
            diag, upper = self.vec.blocks(ke)
            if c is not None:
                diag = ct.add_nodal_to_blocks(diag, ct.nodal_hessian(c, self.X, u, depth, lag))
            return diag, upper

        if c is None:
            def total(v):
                return self.energy(v, dq, mat, notch, vis)
        else:
            def total(v):
                return self.energy(v, dq, mat, notch, vis) + self.contact_energy(v, depth, lag, c)

        if not self.newton_regularise:
            return newton.minimise(total, hessian_blocks, u, self.free, self.free_blocks, max_iter=self.max_newton,
                                   step_tol=1e-11 * float(self.grid.y[-1]))
        if self.newton_stall:
            return newton.minimise(total, hessian_blocks, u, self.free, self.free_blocks, max_iter=self.max_newton,
                                   step_tol=1e-11 * float(self.grid.y[-1]), regularise=True,
                                   stall_step=self.newton_stall)
        return newton.minimise(total, hessian_blocks, u, self.free, self.free_blocks, max_iter=self.max_newton,
                               step_tol=1e-11 * float(self.grid.y[-1]), regularise=True)

    def plus(self, u: jax.Array, mat: Material, notch: Any = None, vis: ViscousStep | None = None) -> jax.Array:
        if vis is not None:
            if self.axi:
                return jax.vmap(lambda ue, x0, dx, dy, gm: axi.element_plus(ue, x0, dx, dy, mat, ViscousStep(
                    gm, vis.dt)))(u[self.I, self.J], self.x0, *self.sizes(notch), vis.gamma)
            return jax.vmap(lambda ue, dx, dy, gm: element_plus(ue, dx, dy, mat, self.stress, ViscousStep(
                gm, vis.dt)))(u[self.I, self.J], *self.sizes(notch), vis.gamma)
        if self.axi:
            return jax.vmap(lambda ue, x0, dx, dy: axi.element_plus(ue, x0, dx, dy, mat))(u[self.I, self.J], self.x0,
                                                                                          *self.sizes(notch))
        return jax.vmap(lambda ue, dx, dy: element_plus(ue, dx, dy, mat, self.stress))(u[self.I, self.J],
                                                                                         *self.sizes(notch))

    def solve_damage(self, history: jax.Array, mat: Material, notch: Any = None, d_prev: Any = None) -> jax.Array:
        dx, dy = self.sizes(notch)
        return phase_field.solve_damage(history, mat.gc, mat.l, dx, dy, self.grid, self.sca, self.I, self.J,
                                        weights=self.gauss_weights(), viscosity=self.viscosity, d_prev=d_prev)

    def damage_at_gauss(self, d: jax.Array) -> jax.Array:
        if self.mode == "phase_field":
            return phase_field.damage_at_gauss(d, self.I, self.J)
        if self.mode == "local":
            return deletion.damage_at_gauss(d)
        return jnp.zeros((self.grid.n_elements, 4), dtype=d.dtype)

    def crack_energy(self, d: jax.Array, mat: Material, notch: Any = None) -> jax.Array:
        if self.mode != "phase_field":
            return jnp.zeros((), dtype=d.dtype)
        dx, dy = self.sizes(notch)
        return phase_field.crack_energy(d, mat.gc, mat.l, dx, dy, self.I, self.J, weights=self.gauss_weights())

    def centre_stretch(self, u: jax.Array, notch: Any = None) -> jax.Array:
        if self.axi:
            return jax.vmap(axi.centre_stretch)(u[self.I, self.J], self.x0, *self.sizes(notch))
        return jax.vmap(centre_stretch)(u[self.I, self.J], *self.sizes(notch))

    def reaction(self, u: jax.Array, dq: jax.Array, mat: Material, notch: Any = None,
                 vis: ViscousStep | None = None) -> jax.Array:
        """The force the load carries, per unit thickness of the model's half (with ``vis``, the step's incremental
        potential's derivative: the stress with the branches' updated strains)."""
        r = jax.grad(self.energy)(u, dq, mat, notch, vis)
        return reaction_force(self.specimen.reaction, r)

    def indenter_force(self, u: jax.Array, depth: jax.Array, lag: ct.Lag | None, c: Any = None) -> jax.Array:
        """The indenter's reaction: the contact energy's derivative in the depth at fixed displacement."""
        return jax.grad(lambda z: self.contact_energy(u, z, lag, c))(depth)

    def _pressure_scale(self, lag: ct.Lag, p_start: Any) -> Any:
        """The pressure the lag's change is judged against: the new largest (round 3's), and with the lag relaxed
        also the step's start's, so that a contact that vanishes (a relaxed lag decaying towards zero) can pass."""
        if self.lag_relaxation == 1.0:
            return jnp.max(lag.pressure)
        return jnp.maximum(jnp.max(lag.pressure), p_start)

    def _lag(self, u: jax.Array, depth: jax.Array, start: ct.Lag | None, c: Any = None) -> ct.Lag | None:
        c = self.contact if c is None else c
        if c is None or not c.sliding:
            return None
        p, n = ct.pressure_and_normal(c, self.X, u, depth)
        assert start is not None
        if self.lag_relaxation != 1.0:
            p = start.pressure + self.lag_relaxation * (p - start.pressure)
        return ct.Lag(p, n, start.u_start, start.depth_start)

    def damage_functional(self, d: jax.Array, history: jax.Array, mat: Material, d_prev: jax.Array,
                          notch: Any = None) -> jax.Array:
        """sum w [(1-d)^2 H + Gc (d^2/2l + l/2 |grad d|^2) + eta/2 (d - d_prev)^2] (the damage's energy for a
        history)."""
        dx, dy = self.sizes(notch)
        wq = self.gauss_weights()
        wq = jnp.broadcast_to((dx * dy / 4.0)[:, None], history.shape) if wq is None else wq
        dq = phase_field.damage_at_gauss(d, self.I, self.J)
        G = jax.vmap(phase_field.gauss_gradients)(dx, dy)
        grad = jnp.einsum("ea,egai->egi", d[self.I, self.J], G)
        e = (1.0 - dq) ** 2 * history + mat.gc * (dq ** 2 / (2 * mat.l) + 0.5 * mat.l * jnp.sum(grad ** 2, -1))
        if self.viscosity:
            e = e + 0.5 * self.viscosity * (dq - phase_field.damage_at_gauss(d_prev, self.I, self.J)) ** 2
        return jnp.sum(wq * e)

    def _monolithic(self, state: State, u: jax.Array, mat: Material, notch: Any, depth: Any, lag: Any, c: Any = None,
                    vis: ViscousStep | None = None):
        c = self.contact if c is None else c
        n_u = u.size
        shape_u, shape_d = u.shape, state.d.shape
        g = self.grid

        def split(z):
            return z[:n_u].reshape(shape_u), z[n_u:].reshape(shape_d)

        def total(v, dq):
            e = self.energy(v, dq, mat, notch, vis)
            return e + self.contact_energy(v, depth, lag, c) if c is not None else e

        def residual(z):
            v, d = split(z)
            ru = jax.grad(total)(v, self.damage_at_gauss(d)) * self.free
            hist = phase_field.updated_history(state.history, self.plus(v, mat, notch, vis))
            rd = jax.grad(self.damage_functional)(d, hist, mat, state.d, notch)
            return jnp.concatenate([ru.ravel(), rd.ravel()])

        def blocks(z):
            v, d = split(z)
            dq = self.damage_at_gauss(d)
            ke = self.element_hessians(v, dq, mat, notch, vis)
            du, uu = self.vec.blocks(ke)
            if c is not None:
                du = ct.add_nodal_to_blocks(du, ct.nodal_hessian(c, self.X, v, depth, lag))
            du, uu = constrain(du, uu, self.free_blocks)
            hist = phase_field.updated_history(state.history, self.plus(v, mat, notch, vis))
            # the damage functional is quadratic in d: its element matrices are assembled as the damage solve's
            dx, dy = self.sizes(notch)
            wq = self.gauss_weights()
            wq = jnp.broadcast_to((dx * dy / 4.0)[:, None], hist.shape) if wq is None else wq
            N = jnp.asarray(phase_field.N_GAUSS)
            G = jax.vmap(phase_field.gauss_gradients)(dx, dy)
            coef = wq * (2.0 * (hist + mat.gc / (2.0 * mat.l)) + self.viscosity)
            kde = jnp.einsum("eg,ga,gb->eab", coef, N, N) + mat.gc * mat.l * jnp.einsum("eg,egai,egbi->eab", wq, G, G)
            dd_, ud_ = self.sca.blocks(kde)
            return (du, uu), (dd_, ud_)

        z0 = jnp.concatenate([u.ravel(), state.d.ravel()])
        z, k, ok = bfgs.solve(residual, blocks, z0, n_u, (g.nx + 1, 2 * (g.ny + 1)), (g.nx + 1, g.ny + 1),
                              tol_u=1e-10 * float(g.y[-1]), tol_d=self.tol_d)
        v, d = split(z)
        hist = phase_field.updated_history(state.history, self.plus(v, mat, notch, vis))
        return v, d, hist, k, k, jnp.zeros((), u.dtype), (~ok).astype(jnp.int32)

    # --- the step ------------------------------------------------------------------------------------------------
    def initial(self, dtype: Any = jnp.float64, mat: Material | None = None) -> State:
        """The undeformed state; with a viscoelastic ``mat``, its branches relaxed (zero internal strains)."""
        g = self.grid
        u = jnp.zeros((g.nx + 1, g.ny + 1, 2), dtype=dtype)
        d = jnp.zeros((g.nx + 1, g.ny + 1) if self.mode != "local" else (g.n_elements,), dtype=dtype)
        gamma = None
        if mat is not None and mat.prony is not None:
            gamma = law.initial_strains((g.n_elements, 4), mat, dtype)
        return State(u, d, jnp.zeros((g.n_elements, 4), dtype=dtype), jnp.zeros((), dtype=dtype), gamma)

    def viscous_step(self, state: State, delta: Any, mat: Material, dt: Any, dtype: Any) -> ViscousStep | None:
        """The step's branches (None for an elastic material): the state's internal strains and the step's time."""
        if mat.prony is None:
            return None
        gamma = state.gamma if state.gamma is not None else law.initial_strains((self.grid.n_elements, 4), mat, dtype)
        if dt is None:
            if self.rate is None:
                raise ValueError("a viscoelastic material needs the load's rate (Solver.rate) or the step's dt")
            dt = jnp.abs(jnp.asarray(delta, dtype) - state.delta) / self.rate
        return ViscousStep(gamma, jnp.asarray(dt, dtype))

    def advance(self, state: State, delta: jax.Array, mat: Material, notch: Any = None,
                tip: Any = None, dt: Any = None) -> tuple[State, StepOut]:
        """One load step to ``delta``. ``tip``: the indenter's numbers (``contact.GEOMETRY``, possibly traced) in place
        of the specimen's, so that a derivative reaches its shape. ``dt``: the step's time (s) for a viscoelastic
        material; by default |delta - state.delta| / ``rate`` (a hold at fixed ``delta`` gives it explicitly)."""
        sp = self.specimen
        con = None if self.contact is None else self.contact.at(tip)
        u = apply_load(sp.load, state.u, delta, state.delta, self.y_frac)
        vis = self.viscous_step(state, delta, mat, dt, u.dtype)
        depth = jnp.asarray(delta, u.dtype) if con is not None else None
        if con is not None:  # a predictor that inverts an element (infinite energy) falls back to the last state
            e_pred = self.energy(u, self.damage_at_gauss(state.d), mat, notch, vis)
            u = jnp.where(jnp.isfinite(e_pred), u, state.u)
        lag, p_start = None, None
        if con is not None and con.sliding:
            p0, n0 = ct.pressure_and_normal(con, self.X, state.u, state.delta)
            p_start = jnp.max(p0)
            lag = ct.Lag(p0, n0, state.u, jnp.asarray(state.delta, u.dtype))
        friction = lag is not None

        if self.mode == "phase_field" and self.scheme == "bfgs":
            u, d, hist, k, nk, err, capped = self._monolithic(state, u, mat, notch, depth, lag, con, vis)
            d_used = d
            dq = self.damage_at_gauss(d)
        elif self.mode == "phase_field":
            def body(c):
                u, d, hist, k, nk, _, _, capped, lag, _ = c
                dq = self.damage_at_gauss(d)
                u, n, ok = self.newton(u, dq, mat, notch, depth, lag, con, vis)
                hist = phase_field.updated_history(state.history, self.plus(u, mat, notch, vis))
                d_new = self.solve_damage(hist, mat, notch, state.d)
                dp = jnp.zeros((), u.dtype)
                if friction:
                    new_lag = self._lag(u, depth, lag, con)
                    dp = (jnp.max(jnp.abs(new_lag.pressure - lag.pressure)) / self.lag_relaxation
                          / (self._pressure_scale(new_lag, p_start) + 1e-300))
                    lag = new_lag
                return u, d_new, hist, k + 1, nk + n, jnp.max(jnp.abs(d_new - d)), d, capped + (~ok), lag, dp

            def cond(c):
                return (c[3] < self.max_staggered) & ((c[5] > self.tol_d) | (c[9] > self.tol_p))

            init = (u, state.d, state.history, 0, 0, jnp.asarray(jnp.inf, u.dtype), state.d, 0, lag,
                    jnp.zeros((), u.dtype))
            u, d, hist, k, nk, err, d_used, capped, lag, _ = jax.lax.while_loop(cond, body, init)
            dq = self.damage_at_gauss(d_used)
        elif self.mode == "local":
            def body(c):
                u, d, k, nk, _, capped, lag = c
                u, n, ok = self.newton(u, self.damage_at_gauss(d), mat, notch, depth, lag, con, vis)
                stretch = self.centre_stretch(u, notch)
                new = deletion.newly_deleted(stretch, d, mat.stretch_limit)
                if friction:
                    lag = self._lag(u, depth, lag, con)
                return u, jnp.where(new, 1.0, d), k + 1, nk + n, jnp.sum(new).astype(u.dtype), capped + (~ok), lag

            def cond(c):
                return (c[2] < self.max_staggered) & (c[4] > 0)

            u, d, k, nk, err, capped, lag = jax.lax.while_loop(
                cond, body, (u, state.d, 0, 0, jnp.ones((), u.dtype), 0, lag))
            hist = state.history
            dq = self.damage_at_gauss(d)  # equals the damage the last solve used (no new deletion)
        else:
            dq = self.damage_at_gauss(state.d)
            if friction:
                def body(c):
                    u, k, nk, _, capped, lag = c
                    u, n, ok = self.newton(u, dq, mat, notch, depth, lag, con, vis)
                    new_lag = self._lag(u, depth, lag, con)
                    dp = (jnp.max(jnp.abs(new_lag.pressure - lag.pressure)) / self.lag_relaxation
                          / (self._pressure_scale(new_lag, p_start) + 1e-300))
                    return u, k + 1, nk + n, dp, capped + (~ok), new_lag

                def cond(c):
                    return (c[1] < self.max_staggered) & (c[3] > self.tol_p)

                u, k, nk, err, capped, lag = jax.lax.while_loop(
                    cond, body, (u, 0, 0, jnp.asarray(jnp.inf, u.dtype), 0, lag))
            else:
                u, nk, ok = self.newton(u, dq, mat, notch, depth, lag, con, vis)
                k, err, capped = 1, jnp.zeros((), u.dtype), (~ok).astype(jnp.int32)
            d, hist = state.d, state.history

        zero = jnp.zeros((), u.dtype)
        if con is None:
            force = self.reaction(u, dq, mat, notch, vis) * sp.thickness * sp.force_factor
            e_c, w_f, f_n = zero, zero, zero
        else:
            force = self.indenter_force(u, depth, lag, con) * sp.thickness * sp.force_factor
            f_n = self.indenter_force(u, depth, None, con) * sp.thickness * sp.force_factor
            e_c = ct.penalty_energy(con, self.X, u, depth) * sp.thickness * sp.force_factor
            w_f = (ct.friction_energy(con, self.X, u, depth, lag) * sp.thickness * sp.force_factor
                   if friction else zero)
        w_v = zero
        if self.viscosity and self.mode == "phase_field":
            dd = phase_field.damage_at_gauss(d, self.I, self.J) - phase_field.damage_at_gauss(state.d, self.I, self.J)
            wq = self.gauss_weights()
            wq = jnp.broadcast_to((self.dx * self.dy / 4.0)[:, None], dd.shape) if wq is None else wq
            w_v = self.viscosity * jnp.sum(wq * dd * dd) * sp.thickness * sp.energy_factor
        gamma, w_b, stored, step_dt = None, zero, None, zero
        if vis is not None:
            c_end = self.cauchy_green(u, notch)
            gamma = law.updated_strains(c_end, vis.gamma, mat, vis.dt)
            per_point = law.branch_dissipation(self.cauchy_green(state.u, notch), c_end, vis.gamma, gamma, mat)
            g = law.degradation(dq, mat.k_res)
            w_b = jnp.sum(self.weights(notch) * g * per_point) * sp.thickness * sp.energy_factor
            stored = ViscousStep(gamma, zero)
            step_dt = vis.dt
        out = StepOut(force=force, energy=sp.energy_factor * self.energy(u, dq, mat, notch, stored) * sp.thickness,
                      crack_energy=sp.energy_factor * self.crack_energy(d, mat, notch) * sp.thickness,
                      iterations=jnp.asarray(k), newton=jnp.asarray(nk), residual=err,
                      newton_capped=jnp.asarray(capped), contact_energy=e_c, friction_work=w_f, viscous_work=w_v,
                      force_normal=f_n, branch_dissipation=w_b, dt=step_dt)
        return State(u, d, hist, jnp.asarray(delta, u.dtype), gamma), out
