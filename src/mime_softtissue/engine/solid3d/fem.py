"""Trilinear hexes on a rectilinear grid: nodal fields ``(nx+1, ny+1, nz+1, ncomp)``, Gauss-point arrays
``(2, 2, 2, nx, ny, nz)`` (the first three indices pick the Gauss point's side along x, y, z), written with array slicing
(no scatter). The solid's energy is the plugin's law (``engine.material.energy_parts``) at each Gauss point's invariants,
so a 3-D run reads the same material as the 2-D and axisymmetric builds; the damage is AT2 with a history field.

From spike 3D-PUNCTURE's ``fem3d.py`` (its plants and energy variants left out: the split here is the law's own)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..material import Material, energy_parts

GQ = float(1.0 / np.sqrt(3.0))  # a Python float, so a float32 field is not promoted
SIDES = ((0.5 * (1 + GQ), 0.5 * (1 - GQ)), (0.5 * (1 - GQ), 0.5 * (1 + GQ)))  # (w_left, w_right) at side 0, 1


@dataclass(frozen=True)
class Grid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.x) - 1, len(self.y) - 1, len(self.z) - 1)

    @property
    def nodes(self) -> tuple[int, int, int]:
        return (len(self.x), len(self.y), len(self.z))

    def sizes(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        return (jnp.asarray(np.diff(self.x))[:, None, None], jnp.asarray(np.diff(self.y))[None, :, None],
                jnp.asarray(np.diff(self.z))[None, None, :])

    def coarse_axes(self, min_cells: int) -> tuple[bool, bool, bool]:
        """Coarsen an axis while it keeps at least ``min_cells`` cells; an odd count pairs all cells but the last."""
        a, b, c = (n >= 2 * min_cells for n in self.shape)
        return a, b, c

    def coarse(self, axes: tuple[bool, bool, bool]) -> "Grid":
        x, y, z = (a[coarse_index(len(a) - 1)] if f else a for a, f in zip((self.x, self.y, self.z), axes))
        return Grid(x, y, z)


def coarse_index(n: int) -> np.ndarray:
    """The fine nodes kept on the coarse grid of an axis with ``n`` cells: every other node, and the last one when
    ``n`` is odd (its last coarse cell is one fine cell)."""
    idx = np.arange(0, n + 1, 2)
    return idx if n % 2 == 0 else np.append(idx, n)


def _diff(f: jax.Array, axis: int) -> jax.Array:
    n = f.shape[axis]
    return jax.lax.slice_in_dim(f, 1, n, axis=axis) - jax.lax.slice_in_dim(f, 0, n - 1, axis=axis)


def _sides(f: jax.Array, axis: int) -> jax.Array:
    """Linear interpolation along ``axis`` at both Gauss sides at once: a new leading axis of size 2."""
    n = f.shape[axis]
    a = jax.lax.slice_in_dim(f, 0, n - 1, axis=axis)
    b = jax.lax.slice_in_dim(f, 1, n, axis=axis)
    return jnp.stack([SIDES[0][0] * a + SIDES[0][1] * b, SIDES[1][0] * a + SIDES[1][1] * b])


def gauss_values(f: jax.Array) -> jax.Array:
    """A nodal scalar field (nx+1, ny+1, nz+1) at the Gauss points: (2, 2, 2, nx, ny, nz)."""
    t = _sides(_sides(_sides(f, 0), 2), 4)  # (sz, sy, sx, nx, ny, nz)
    return jnp.transpose(t, (2, 1, 0, 3, 4, 5))


def gauss_gradients(f: jax.Array, grid: Grid) -> jax.Array:
    """The gradient of a nodal field (nx+1, ny+1, nz+1, ncomp) at all Gauss points: (2, 2, 2, nx, ny, nz, ncomp, 3)."""
    dx, dy, dz = (a.astype(f.dtype) for a in grid.sizes())
    ex = f.ndim - 3

    def scale(s: jax.Array) -> jax.Array:
        return s.reshape(s.shape + (1,) * ex)

    Dx = _diff(f, 0) / scale(dx)
    Dy = _diff(f, 1) / scale(dy)
    Dz = _diff(f, 2) / scale(dz)
    gx = jnp.swapaxes(_sides(_sides(Dx, 1), 3), 0, 1)[None]          # (1, sy, sz, ...)
    gy = jnp.swapaxes(_sides(_sides(Dy, 0), 3), 0, 1)[:, None]       # (sx, 1, sz, ...)
    gz = jnp.swapaxes(_sides(_sides(Dz, 0), 2), 0, 1)[:, :, None]    # (sx, sy, 1, ...)
    full = (2, 2, 2) + gx.shape[3:]
    return jnp.stack([jnp.broadcast_to(gx, full), jnp.broadcast_to(gy, full), jnp.broadcast_to(gz, full)], axis=-1)


def gauss_weight(grid: Grid, dtype: Any = None) -> jax.Array:
    dx, dy, dz = grid.sizes()
    w = dx * dy * dz / 8.0
    return w if dtype is None else w.astype(dtype)


def invariants(F: jax.Array) -> tuple[jax.Array, jax.Array]:
    """(J, I1) of deformation gradients ``F`` (..., 3, 3)."""
    J = (F[..., 0, 0] * (F[..., 1, 1] * F[..., 2, 2] - F[..., 1, 2] * F[..., 2, 1])
         - F[..., 0, 1] * (F[..., 1, 0] * F[..., 2, 2] - F[..., 1, 2] * F[..., 2, 0])
         + F[..., 0, 2] * (F[..., 1, 0] * F[..., 2, 1] - F[..., 1, 1] * F[..., 2, 0]))
    return J, jnp.sum(F * F, axis=(-2, -1))


def plus_minus(u: jax.Array, grid: Grid, mat: Material) -> tuple[jax.Array, jax.Array]:
    """The law's (psi_plus, psi_minus) at every Gauss point of the displacement ``u``."""
    F = jnp.eye(3, dtype=u.dtype) + gauss_gradients(u, grid)
    J, I1 = invariants(F)
    return energy_parts(J, I1, (), mat)


def solid_energy(u: jax.Array, g: jax.Array, grid: Grid, mat: Material) -> jax.Array:
    """sum over Gauss points of w (g psi_plus + psi_minus); ``g`` (2, 2, 2, nx, ny, nz) the degradation."""
    plus, minus = plus_minus(u, grid, mat)
    return jnp.sum(gauss_weight(grid, u.dtype) * (g * plus + minus))


def degradation(d: jax.Array, k_res: Any) -> jax.Array:
    """g = (1 - d)^2 + k_res at the Gauss points from nodal damage (nx+1, ny+1, nz+1)."""
    return (1.0 - gauss_values(d)) ** 2 + k_res


def damage_energy(d: jax.Array, H: jax.Array, grid: Grid, gc: Any, l: Any) -> jax.Array:
    """AT2 with the history field: sum w [(1-d)^2 H + Gc (d^2/2l + l/2 |grad d|^2)], quadratic in nodal ``d``."""
    w = gauss_weight(grid, d.dtype)
    dq = gauss_values(d)
    G = gauss_gradients(d[..., None], grid)[..., 0, :]
    return jnp.sum(w * ((1.0 - dq) ** 2 * H + gc * dq ** 2 / (2.0 * l) + gc * 0.5 * l * jnp.sum(G * G, axis=-1)))


def crack_energy(d: jax.Array, grid: Grid, gc: Any, l: Any) -> jax.Array:
    return damage_energy(d, jnp.zeros((2, 2, 2) + grid.shape, d.dtype), grid, gc, l)
