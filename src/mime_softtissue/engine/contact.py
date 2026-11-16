"""A rigid indenter as a signed distance with a nodal penalty, frictionless or with regularised Coulomb friction.

The indenter is data (``Indenter``), its lowest point at ``top - depth``: a blunt punch (a cylinder of radius ``radius``
whose bottom edge is rounded by ``fillet``), a hollow needle (a tube from ``inner`` to ``radius`` with a fully rounded
end), a cone (half-angle ``half_angle`` up to a shaft of ``radius``, its apex rounded by ``fillet``) or a headed tip (a
head of ``head_radius`` and ``head_length``, its edges rounded by ``fillet``, on a neck of ``radius``: the
axisymmetric stand-in for a barb). In the (R, Z) half-plane its signed distance ``phi`` is negative inside; its numbers
may be traced (``Contact.at``), and a caller may give its own ``distance`` (a registered shape read through MIME).
The penalty energy is

    E_c = sum over nodes of w_n (k / 2) <-phi(X_n + u_n)>^2,

``w_n`` a node's ring area (``axisymmetric.nodal_ring_area``), ``k`` a stiffness in Pa/m. Friction is the step's
pseudo-potential with the normal pressure and the normal lagged (taken from the last iterate):

    D = sum over nodes of (mu_f p_n + tau_a chi_n) w_n (sqrt(|s_t|^2 + eps^2) - eps),
    s_t = (I - n n^T)(u_n - u_n^start + depth step e_Z)

the slip of the node against the indenter over the step; ``tau_a`` an adhesive shear on the nodes in contact at the
step's start (``chi_n``, lagged with the pressure; the interface law tau = tau_a + mu p, a placeholder). Both are differentiable; the punch force is the derivative
of E_c + D in the depth at the solved displacement."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


KINDS = ("punch", "tube", "cone", "headed")
#: The numbers of an indenter that may be traced (a study's or a test's derivative reaches them).
GEOMETRY = ("radius", "fillet", "inner", "half_angle", "head_radius", "head_length")


@dataclass(frozen=True)
class Indenter:
    kind: str  # one of KINDS
    radius: Any  # the punch's radius, the tube's outer radius, the cone's shaft radius, the headed tip's neck radius
    fillet: Any  # the punch's edge radius, the cone's apex radius, the head's edge radius; a tube's is ignored
    top: float  # the block's top face (Z): the tip is at top - depth
    inner: Any = 0.0  # the tube's inner radius
    half_angle: Any = 0.0  # the cone's half-angle, radians
    head_radius: Any = 0.0  # the headed tip's head
    head_length: Any = 0.0

    def geometry(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in GEOMETRY}


@dataclass(frozen=True)
class Contact:
    indenter: Any  # an ``Indenter``, or whatever a caller's own ``distance`` reads (the registered tip's ``ShapeTip``)
    stiffness: float  # k, Pa/m
    weights: np.ndarray  # (nx+1, ny+1) nodal ring areas
    active: np.ndarray  # (nx+1, ny+1) 1 where a node may touch the indenter
    friction: float = 0.0
    slip: float = 1e-7  # eps, m
    adhesion: Any = 0.0  # tau_a, Pa: an adhesive shear on the nodes in contact (a placeholder law)
    distance: Callable[..., Any] | None = None  # (indenter, R, Z, depth) -> phi; None: signed_distance
    #: The step lags a pressure whatever the friction's value (a caller whose friction is traced says so here).
    always_sliding: bool = False

    @property
    def sliding(self) -> bool:
        """Whether the contact resists slip (friction or adhesion), so that the step lags a pressure."""
        return self.always_sliding or _nonzero(self.friction) or _nonzero(self.adhesion)

    def at(self, geometry: Mapping[str, Any] | None) -> "Contact":
        """The same contact with the numbers in ``geometry`` (traced values) in place of its own: the interface's
        (``friction``, ``adhesion``, ``stiffness``) and the indenter's (any other key)."""
        if not geometry:
            return self
        own = {k: v for k, v in geometry.items() if k in ("friction", "adhesion", "stiffness")}
        tip = {k: v for k, v in geometry.items() if k not in own}
        return replace(self, indenter=replace(self.indenter, **tip) if tip else self.indenter, **own)

    def phi(self, R: jax.Array, Z: jax.Array, depth: jax.Array) -> jax.Array:
        if self.distance is not None:
            return self.distance(self.indenter, R, Z, depth)
        return signed_distance(self.indenter, R, Z, depth)


class Lag(NamedTuple):
    """The friction's lagged quantities and the step's start: pressure (nx+1, ny+1), normal (nx+1, ny+1, 2), the
    displacement at the step's start, and the depth at the step's start."""
    pressure: jax.Array
    normal: jax.Array
    u_start: jax.Array
    depth_start: jax.Array


def _nonzero(v: Any) -> bool:
    """Whether a number given as a Python value is not zero; a traced value counts as not zero."""
    return not isinstance(v, (int, float)) or v != 0.0


def _norm(a: jax.Array, b: jax.Array) -> jax.Array:
    """sqrt(a^2 + b^2) with a finite derivative at zero."""
    s = a * a + b * b
    safe = jnp.where(s > 0, s, 1.0)
    return jnp.where(s > 0, jnp.sqrt(safe), 0.0)


def _cone(R: jax.Array, z: jax.Array, half_angle: Any) -> jax.Array:
    """The signed distance from a sharp cone, its apex at the origin, opening up the Z axis at ``half_angle``."""
    c, s = jnp.cos(half_angle), jnp.sin(half_angle)
    along = jnp.abs(R) * s + z * c  # the projection on the side's direction (sin a, cos a)
    return jnp.where(along > 0, jnp.abs(R) * c - z * s, _norm(R, z))


def _rounded_box(R: jax.Array, Z: jax.Array, half_r: Any, z0: Any, half_z: Any, rounding: Any) -> jax.Array:
    """The signed distance from a box |R| <= half_r, |Z - z0| <= half_z, its corners rounded by ``rounding``."""
    qr = jnp.abs(R) - (half_r - rounding)
    qz = jnp.abs(Z - z0) - (half_z - rounding)
    outside = _norm(jnp.maximum(qr, 0.0), jnp.maximum(qz, 0.0))
    return outside + jnp.minimum(jnp.maximum(qr, qz), 0.0) - rounding


def indenter_step(depth: jax.Array, depth_start: jax.Array) -> jax.Array:
    """How far the indenter moved down over the step (the friction's slip is the node's motion relative to it). A
    module function looked up at trace time, so that a control's plant can replace it."""
    return depth - depth_start


def signed_distance(ind: Indenter, R: jax.Array, Z: jax.Array, depth: jax.Array) -> jax.Array:
    tip = ind.top - depth
    if ind.kind == "cone":
        apex = tip + ind.fillet
        return jnp.maximum(_cone(R, Z - apex, ind.half_angle) - ind.fillet, jnp.abs(R) - ind.radius)
    if ind.kind == "headed":
        head = _rounded_box(R, Z, ind.head_radius, tip + 0.5 * ind.head_length, 0.5 * ind.head_length, ind.fillet)
        neck = jnp.maximum(jnp.abs(R) - ind.radius, tip + 0.5 * ind.head_length - Z)
        return jnp.minimum(head, neck)
    if ind.kind == "punch":
        b = ind.radius - ind.fillet
        zb = tip + ind.fillet
        qr = jnp.abs(R) - b
        qz = zb - Z
        outside = _norm(jnp.maximum(qr, 0.0), jnp.maximum(qz, 0.0))
        return outside + jnp.minimum(jnp.maximum(qr, qz), 0.0) - ind.fillet
    if ind.kind == "tube":
        rho = 0.5 * (ind.radius - ind.inner)
        rc = 0.5 * (ind.radius + ind.inner)
        zc = tip + rho
        dz = jnp.minimum(Z - zc, 0.0)
        return _norm(R - rc, dz) - rho
    raise ValueError(f"indenter kind is one of {KINDS}")


def _positions(X: jax.Array, u: jax.Array) -> tuple[jax.Array, jax.Array]:
    x = X + u
    return x[..., 0], x[..., 1]


def penalty_energy(c: Contact, X: jax.Array, u: jax.Array, depth: jax.Array) -> jax.Array:
    phi = c.phi(*_positions(X, u), depth)
    gap = jnp.minimum(phi, 0.0)
    return 0.5 * c.stiffness * jnp.sum(jnp.asarray(c.active * c.weights) * gap * gap)


def shear_strength(c: Contact, pressure: jax.Array) -> jax.Array:
    """The interface's shear strength at each node, tau = mu p + tau_a [p > 0], from the lagged pressure."""
    tau = c.friction * pressure
    if _nonzero(c.adhesion):
        tau = tau + c.adhesion * (pressure > 0).astype(pressure.dtype)
    return tau


def friction_energy(c: Contact, X: jax.Array, u: jax.Array, depth: jax.Array, lag: Lag) -> jax.Array:
    if not c.sliding:
        return jnp.zeros((), dtype=u.dtype)
    rel = u - lag.u_start
    rel = rel.at[..., 1].add(indenter_step(depth, lag.depth_start))  # the indenter moved by -(depth step) e_Z
    n = lag.normal
    sn = jnp.sum(rel * n, axis=-1, keepdims=True)
    st = rel - sn * n
    eps = c.slip
    mag = jnp.sqrt(jnp.sum(st * st, axis=-1) + eps * eps) - eps
    if not _nonzero(c.adhesion):  # round 3's arithmetic, in its order (its records reproduce bit for bit)
        return c.friction * jnp.sum(jnp.asarray(c.active * c.weights) * lag.pressure * mag)
    return jnp.sum(jnp.asarray(c.active * c.weights) * shear_strength(c, lag.pressure) * mag)


def pressure_and_normal(c: Contact, X: jax.Array, u: jax.Array, depth: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The penalty pressure k <-phi> and the outward normal grad phi at each node (for the friction's lag)."""
    def phi_of(xy):
        return c.phi(xy[0], xy[1], depth)
    x = X + u
    phi = jax.vmap(jax.vmap(phi_of))(x)
    grad = jax.vmap(jax.vmap(jax.grad(phi_of)))(x)
    nrm = jnp.sqrt(jnp.sum(grad * grad, axis=-1, keepdims=True))
    n = grad / jnp.where(nrm > 0, nrm, 1.0)
    return c.stiffness * jnp.maximum(-phi, 0.0) * jnp.asarray(c.active), n


def nodal_hessian(c: Contact, X: jax.Array, u: jax.Array, depth: jax.Array, lag: Lag | None) -> jax.Array:
    """The contact energy's 2 x 2 Hessian at each node (it couples no two nodes): (nx+1, ny+1, 2, 2)."""
    w = jnp.asarray(c.active * c.weights)

    def node_energy(un, Xn, wn, pn, nn, usn):
        phi = c.phi(Xn[0] + un[0], Xn[1] + un[1], depth)
        gap = jnp.minimum(phi, 0.0)
        e = 0.5 * c.stiffness * wn * gap * gap
        if c.sliding and lag is not None:
            rel = (un - usn).at[1].add(indenter_step(depth, lag.depth_start))
            st = rel - jnp.sum(rel * nn) * nn
            if not _nonzero(c.adhesion):  # round 3's order
                e = e + c.friction * wn * pn * (jnp.sqrt(jnp.sum(st * st) + c.slip ** 2) - c.slip)
            else:
                e = e + wn * shear_strength(c, pn) * (jnp.sqrt(jnp.sum(st * st) + c.slip ** 2) - c.slip)
        return e

    h = jax.hessian(node_energy)
    if lag is None:  # frictionless: the lag is not read
        z = jnp.zeros(u.shape[:-1], u.dtype)
        return jax.vmap(jax.vmap(h))(u, X, w, z, jnp.zeros_like(u), jnp.zeros_like(u))
    return jax.vmap(jax.vmap(h))(u, X, w, lag.pressure, lag.normal, lag.u_start)


def add_nodal_to_blocks(diag: jax.Array, Hn: jax.Array) -> jax.Array:
    """Adds per-node 2 x 2 blocks to the block-tridiagonal diagonal (column blocks of 2 (ny+1))."""
    nb, m, _ = diag.shape
    ny1 = m // 2
    eye = jnp.eye(ny1, dtype=diag.dtype)
    big = jnp.einsum("jk,cjab->cjakb", eye, Hn).reshape(nb, m, m)
    return diag + big
