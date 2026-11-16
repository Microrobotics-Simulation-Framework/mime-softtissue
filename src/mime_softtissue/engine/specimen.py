"""A specimen is data: a grid, which displacements are prescribed, how the grip or blade is moved, which nodes carry
the reaction, and an optional notch morph. The strip and the blade block are two factories that fill it in, not
engine classes; a new specimen is a new factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from .grid import Grid, graded_rows

Predictor = Literal["spread", "proportional", "prescribed_only"]


@dataclass(frozen=True)
class Load:
    """The prescribed motion: nodes ``(i, j)`` (index arrays) take the y displacement ``sense * delta`` and the x
    displacement 0. ``predictor`` is the first guess of a load step: ``"spread"`` adds the grip's increment linearly
    over the height; ``"proportional"`` scales the previous field by the depth's ratio; ``"prescribed_only"`` moves
    only the prescribed nodes."""

    nodes: tuple[np.ndarray, np.ndarray]
    sense: float
    predictor: Predictor


@dataclass(frozen=True)
class Reaction:
    """The force the load carries: ``sign`` times the sum of the internal force's y component over ``nodes``."""

    nodes: tuple[np.ndarray, np.ndarray]
    sign: float


@dataclass(frozen=True)
class Morph:
    """The notch's tip node ``node`` of a ``width``-wide specimen: the node columns are mapped piecewise linearly so
    that the tip lies at a traced notch length (the topology fixed, the geometry continuous in the notch)."""

    node: int
    width: float


@dataclass(frozen=True)
class Specimen:
    """``force_factor`` scales the reaction of a half model to the whole specimen's force (symmetry);
    ``energy_factor`` the stored and crack energies likewise. ``kinematics``: "plane_strain" or "axisymmetric" (x is
    then the radius, the energy is the whole ring's, and thickness and both factors are 1). ``contact``: a rigid
    indenter moved by the load's ``delta`` (its depth) instead of prescribed nodes; the force is then its reaction."""

    grid: Grid
    free: np.ndarray  # (nx+1, ny+1, 2): 1 where a displacement is solved for, 0 where prescribed
    thickness: float
    force_factor: float
    load: Load
    reaction: Reaction
    morph: Morph | None = None
    kinematics: str = "plane_strain"
    energy_factor: float = 2.0
    contact: Any = None


def morph_columns(x: np.ndarray, morph: Morph, notch: Any) -> Any:
    """The node columns ``x`` with the tip node moved to ``notch``."""
    k = morph.node
    xa = float(x[k])
    left = jnp.asarray(x[:k + 1]) * (notch / xa)
    right = notch + jnp.asarray(x[k + 1:] - xa) * ((morph.width - notch) / (morph.width - xa))
    return jnp.concatenate([left, right])


def apply_load(load: Load, u: jax.Array, delta: jax.Array, previous: jax.Array, y_frac: jax.Array) -> jax.Array:
    """The first guess of the load step to ``delta`` from the field ``u`` of the step to ``previous``."""
    i, j = load.nodes
    if load.predictor == "spread":
        inc = load.sense * (delta - previous)
        u = u.at[:, :, 1].add(inc * y_frac[None, :])
    elif load.predictor == "proportional":
        u = u * jnp.where(previous > 0, delta / jnp.where(previous > 0, previous, 1.0), 1.0)
    return u.at[i, j, 1].set(load.sense * delta).at[i, j, 0].set(0.0)


def reaction_force(reaction: Reaction, r: jax.Array) -> jax.Array:
    """The reaction, per unit thickness of the model, from the internal force ``r``."""
    i, j = reaction.nodes
    return reaction.sign * jnp.sum(r[i, j, 1])


def _row(count: int, row: int) -> tuple[np.ndarray, np.ndarray]:
    return np.arange(count), np.full(count, row)


def edge_notched_strip(*, width: float, half_height: float, thickness: float, notch: float, h: float,
                       fine: float = 2.4e-3, ratio: float = 1.25, cap: float = 8.0) -> Specimen:
    """A single-edge-notched strip, the upper half by symmetry: ``width`` across, ``half_height`` from the crack
    plane to the grip, a notch of length ``notch`` from the left edge on the crack plane (traction free there,
    symmetric on the ligament), the grip clamped and pulled. ``h``: the grid spacing along the crack plane and over
    ``fine``. The tip node is the nearest to the notch; a notch off a node by up to a quarter cell is placed
    exactly by the morph."""
    n = width / h
    if abs(n - round(n)) > 1e-6:
        raise ValueError(f"the width {width} is not a whole number of cells {h}")
    grid = Grid(np.linspace(0.0, width, int(round(n)) + 1), graded_rows(half_height, h, fine, ratio, cap))
    cells = notch / h
    if abs(cells - round(cells)) > 0.25:
        raise ValueError(f"the notch {notch} is more than a quarter cell off a node of spacing {h}")
    tip = int(round(cells))
    free = np.ones((grid.nx + 1, grid.ny + 1, 2))
    free[:, -1, :] = 0.0  # the grip
    free[tip:, 0, 1] = 0.0  # the ligament's symmetry
    top = _row(grid.nx + 1, grid.ny)
    return Specimen(grid=grid, free=free, thickness=thickness, force_factor=1.0,
                    load=Load(top, sense=0.5, predictor="spread"), reaction=Reaction(top, sign=1.0),
                    morph=Morph(tip, width))


def blade_block(*, half_width: float, height: float, thickness: float, punch: float, h: float, fine_x: float,
                fine_y: float, ratio: float = 1.25, cap: float = 8.0, predictor: Predictor = "proportional",
                reaction_skips_edge: bool = False) -> Specimen:
    """A block's right half (symmetry at x = 0), ``half_width`` across and ``height`` tall, its bottom fixed, pressed
    at the top by a rigid flat-ended blade of half-width ``punch`` with stick: the blade's nodes move with it
    (prescribed boundary motion, no contact law). ``h``: the spacing within ``fine_x`` of the axis and ``fine_y`` of
    the top; graded beyond. The load's ``delta`` is the blade's depth. ``reaction_skips_edge`` leaves the blade's
    edge node out of the reaction (a control's planted error)."""
    x = graded_rows(half_width, h, fine_x, ratio, cap)
    y = height - graded_rows(height, h, fine_y, ratio, cap)[::-1]
    x[0], x[-1], y[0], y[-1] = 0.0, half_width, 0.0, height
    grid = Grid(x, y)
    n = punch / h
    if abs(n - round(n)) > 1e-6 or punch > fine_x:
        raise ValueError(f"the blade's edge {punch} is not a node of spacing {h} in the fine region")
    edge = int(round(n))
    free = np.ones((grid.nx + 1, grid.ny + 1, 2))
    free[0, :, 0] = 0.0  # the symmetry
    free[:, 0, :] = 0.0  # the bottom
    free[:edge + 1, -1, :] = 0.0  # the blade
    blade = _row(edge + 1, grid.ny)
    held = (blade[0][:edge], blade[1][:edge]) if reaction_skips_edge else blade
    return Specimen(grid=grid, free=free, thickness=thickness, force_factor=2.0,
                    load=Load(blade, sense=-1.0, predictor=predictor), reaction=Reaction(held, sign=-1.0))


def _axi_grid(radius: float, height: float, h: float, fine_r: float, fine_z: float, ratio: float, cap: float) -> Grid:
    x = graded_rows(radius, h, fine_r, ratio, cap)
    y = height - graded_rows(height, h, fine_z, ratio, cap)[::-1]
    x[0], x[-1], y[0], y[-1] = 0.0, radius, 0.0, height
    return Grid(x, y)


def stretched_cylinder(*, radius: float, height: float, nr: int, nz: int, confined: bool = False) -> Specimen:
    """A solid cylinder (axisymmetric) between frictionless ends: the bottom's u_Z and the top's u_Z prescribed (the
    load's ``delta`` is the top's displacement), the radial displacement free at both ends and zero on the axis.
    ``confined``: the side held radially (a frictionless rigid wall), so the field is u_R = 0 everywhere."""
    grid = Grid(np.linspace(0.0, radius, nr + 1), np.linspace(0.0, height, nz + 1))
    free = np.ones((grid.nx + 1, grid.ny + 1, 2))
    free[0, :, 0] = 0.0  # the axis
    if confined:
        free[-1, :, 0] = 0.0  # the wall
    free[:, 0, 1] = 0.0  # the bottom (u_Z = 0, u_R free)
    free[:, -1, 1] = 0.0  # the top (u_Z prescribed, u_R free)
    top = _row(grid.nx + 1, grid.ny)
    return Specimen(grid=grid, free=free, thickness=1.0, force_factor=1.0,
                    load=Load(top, sense=1.0, predictor="spread"), reaction=Reaction(top, sign=1.0),
                    kinematics="axisymmetric", energy_factor=1.0)


def axisymmetric_block(*, radius: float, height: float, h: float, fine_r: float, fine_z: float,
                       ratio: float = 1.25, cap: float = 8.0, stuck_punch: float | None = None,
                       contact: Any = None, predictor: Predictor = "prescribed_only",
                       x: np.ndarray | None = None, y: np.ndarray | None = None) -> Specimen:
    """A cylindrical block (axisymmetric), its bottom fixed, its side free, pressed on its top face either by a
    rigid flat punch of radius ``stuck_punch`` whose nodes move with it (stick, prescribed nodes, as the 2-D blade)
    or by a ``contact`` indenter (``engine.contact``; built by :func:`with_indenter`). ``h``: the spacing within
    ``fine_r`` of the axis and ``fine_z`` of the top; graded beyond. ``x``, ``y``: node coordinates given outright
    instead (``h`` is then the spacing the punch's edge sits on)."""
    grid = _axi_grid(radius, height, h, fine_r, fine_z, ratio, cap) if x is None else Grid(np.asarray(x), np.asarray(y))
    free = np.ones((grid.nx + 1, grid.ny + 1, 2))
    free[0, :, 0] = 0.0  # the axis
    free[:, 0, :] = 0.0  # the bottom
    if stuck_punch is not None:
        n = stuck_punch / h
        if abs(n - round(n)) > 1e-6 or stuck_punch > fine_r:
            raise ValueError(f"the punch's edge {stuck_punch} is not a node of spacing {h} in the fine region")
        edge = int(round(n))
        free[:edge + 1, -1, :] = 0.0
        nodes = _row(edge + 1, grid.ny)
        return Specimen(grid=grid, free=free, thickness=1.0, force_factor=1.0,
                        load=Load(nodes, sense=-1.0, predictor="proportional"), reaction=Reaction(nodes, sign=-1.0),
                        kinematics="axisymmetric", energy_factor=1.0)
    empty = (np.zeros(0, int), np.zeros(0, int))
    return Specimen(grid=grid, free=free, thickness=1.0, force_factor=1.0,
                    load=Load(empty, sense=-1.0, predictor=predictor), reaction=Reaction(empty, sign=1.0),
                    kinematics="axisymmetric", energy_factor=1.0, contact=contact)


def with_indenter(block: Specimen, indenter: Any, *, stiffness: float, friction: float = 0.0, slip: float = 1e-7,
                  reach: float | None = None) -> Specimen:
    """The block pressed by a rigid ``indenter`` (``engine.contact.Indenter``) through a nodal penalty. ``reach``:
    nodes within this radius of the axis may touch (default: all)."""
    from dataclasses import replace

    from .contact import Contact
    from .kinematics.axisymmetric import nodal_ring_area
    g = block.grid
    w = np.broadcast_to(nodal_ring_area(g.x)[:, None], (g.nx + 1, g.ny + 1)).copy()
    active = np.ones_like(w) if reach is None else (g.x[:, None] <= reach) * np.ones_like(w)
    c = Contact(indenter=indenter, stiffness=stiffness, weights=w, active=active, friction=friction, slip=slip)
    return replace(block, contact=c)


def tube_on_rod(*, inner: float, outer: float, length: float, h: float, interference: float, stiffness: float,
                friction: float, slip: float, adhesion: float = 0.0, weight_plant: bool = False) -> Specimen:
    """An elastic tube (axisymmetric, ``inner`` to ``outer`` radius, ``length`` long) whose ends are held axially and
    free radially (plane strain), on a rigid rod of radius ``inner + interference`` that runs through it and slides
    along its axis (the load's ``delta``): the closed-form check of the friction law's pull-out force. The contact is
    the inner face's, each node weighted by its share of that face, 2 pi R dZ; ``weight_plant`` drops the 2 pi R (a
    control's planted error)."""
    from .contact import Contact, Indenter
    n_r, n_z = int(round((outer - inner) / h)), int(round(length / h))
    grid = Grid(np.linspace(inner, outer, n_r + 1), np.linspace(0.0, length, n_z + 1))
    free = np.ones((grid.nx + 1, grid.ny + 1, 2))
    free[:, 0, 1] = 0.0
    free[:, -1, 1] = 0.0
    dz = np.diff(grid.y)
    share = np.concatenate([[0.0], dz / 2]) + np.concatenate([dz / 2, [0.0]])
    w = np.zeros((grid.nx + 1, grid.ny + 1))
    w[0, :] = share * (1.0 if weight_plant else 2.0 * np.pi * inner)
    active = np.zeros_like(w)
    active[0, :] = 1.0
    # the rod: a punch whose end lies far below the tube, so that only its side touches
    rod = Indenter("punch", radius=inner + interference, fillet=1e-3 * h, top=0.0)
    c = Contact(indenter=rod, stiffness=stiffness, weights=w, active=active, friction=friction, slip=slip,
                adhesion=adhesion)
    empty = (np.zeros(0, int), np.zeros(0, int))
    return Specimen(grid=grid, free=free, thickness=1.0, force_factor=1.0, load=Load(empty, -1.0, "prescribed_only"),
                    reaction=Reaction(empty, 1.0), kinematics="axisymmetric", energy_factor=1.0, contact=c)
