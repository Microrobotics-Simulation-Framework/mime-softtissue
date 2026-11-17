"""The 3-D kinematics option: trilinear hexahedra (Q1, 2 x 2 x 2 Gauss points) on a rectilinear grid, written with array
slicing (no scatter), feeding the plugin's material law (``material.energy_parts``: neo-Hookean with Yeoh's stiffening and
the tension/compression split) through the 3-D invariants J = det F and I1 = tr(F^T F), and the AT2 phase field.

Nodal fields have shape (nx+1, ny+1, nz+1, ncomp). Gauss-point arrays have shape (2, 2, 2, nx, ny, nz): the first three
indices pick the Gauss point's side (-1/sqrt3 or +1/sqrt3 in the element's reference coordinate) along x, y, z.

Near incompressibility: with ``volumetric="mean"`` the volumetric term is evaluated once per element at the element's mean
dilatation Jbar (a Q1/P0-type mean-dilatation element, which does not lock as kappa/mu grows), the damage's split switched
on Jbar; ``"gauss"`` evaluates it at every Gauss point (the 3-D spikes' arithmetic). The isochoric part is always pointwise.

Carried over from the 3-D spikes (3D-PUNCTURE, 3D-COST), with their float32 care: Gauss constants are Python floats, so a
float32 field is not promoted. No MIME import."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .. import material as law
from ..material import Material

GQ = float(1.0 / np.sqrt(3.0))
SIDES = ((0.5 * (1 + GQ), 0.5 * (1 - GQ)), (0.5 * (1 - GQ), 0.5 * (1 + GQ)))  # (w_left, w_right) at side 0, 1
VOLUMETRIC = ("mean", "gauss")


# ---------------------------------------------------------------------------------------------------------- grid ---
@dataclass(frozen=True)
class Grid3:
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
        return tuple(n >= 2 * min_cells for n in self.shape)  # type: ignore[return-value]

    def coarse(self, axes: tuple[bool, bool, bool]) -> "Grid3":
        return Grid3(*[a[coarse_index(len(a) - 1)] if f else a for a, f in zip((self.x, self.y, self.z), axes)])


def coarse_index(n: int) -> np.ndarray:
    """The fine nodes kept on the coarse grid of an axis with ``n`` cells: every other node, and the last one when
    ``n`` is odd (its last coarse cell is one fine cell)."""
    idx = np.arange(0, n + 1, 2)
    return idx if n % 2 == 0 else np.append(idx, n)


def uniform_grid(n: tuple[int, int, int], lengths: tuple[float, float, float]) -> Grid3:
    return Grid3(*[np.linspace(0.0, L, k + 1) for k, L in zip(n, lengths)])


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


def gauss_gradients(f: jax.Array, grid: Grid3) -> jax.Array:
    """The gradient of a nodal field (nx+1, ny+1, nz+1, ncomp) at all Gauss points, vectorised:
    (2, 2, 2, nx, ny, nz, ncomp, 3)."""
    dx, dy, dz = (a.astype(f.dtype) for a in grid.sizes())  # a float32 field stays float32
    ex = f.ndim - 3

    def scale(s):
        return s.reshape(s.shape + (1,) * ex)

    Dx = _diff(f, 0) / scale(dx)
    Dy = _diff(f, 1) / scale(dy)
    Dz = _diff(f, 2) / scale(dz)
    gx = jnp.swapaxes(_sides(_sides(Dx, 1), 3), 0, 1)[None]          # (1, sy, sz, ...)
    gy = jnp.swapaxes(_sides(_sides(Dy, 0), 3), 0, 1)[:, None]       # (sx, 1, sz, ...)
    gz = jnp.swapaxes(_sides(_sides(Dz, 0), 2), 0, 1)[:, :, None]    # (sx, sy, 1, ...)
    full = (2, 2, 2) + gx.shape[3:]
    G = jnp.stack([jnp.broadcast_to(gx, full), jnp.broadcast_to(gy, full), jnp.broadcast_to(gz, full)], axis=-1)
    return G


def gauss_weight(grid: Grid3, dtype=None) -> jax.Array:
    dx, dy, dz = grid.sizes()
    w = dx * dy * dz / 8.0
    return w if dtype is None else w.astype(dtype)



# -------------------------------------------------------------------------------------------- the solid's energy ---
def invariants(F: jax.Array) -> tuple[jax.Array, jax.Array]:
    """J = det F and I1 = tr(F^T F) at deformation gradients F (..., 3, 3)."""
    J = (F[..., 0, 0] * (F[..., 1, 1] * F[..., 2, 2] - F[..., 1, 2] * F[..., 2, 1])
         - F[..., 0, 1] * (F[..., 1, 0] * F[..., 2, 2] - F[..., 1, 2] * F[..., 2, 0])
         + F[..., 0, 2] * (F[..., 1, 0] * F[..., 2, 1] - F[..., 1, 1] * F[..., 2, 0]))
    return J, jnp.sum(F * F, axis=(-2, -1))


def deformation(u: jax.Array, grid: Grid3) -> jax.Array:
    """F at every Gauss point: (2, 2, 2, nx, ny, nz, 3, 3)."""
    return jnp.eye(3, dtype=u.dtype) + gauss_gradients(u, grid)


def _isochoric(mat: Material) -> Material:
    return mat._replace(kappa=0.0)


def _volumetric(mat: Material) -> Material:
    return mat._replace(mu=0.0, c2=0.0, c3=0.0, fibre=None)


def element_mean(a: jax.Array) -> jax.Array:
    """The mean over an element's eight Gauss points: for J, the element's deformed volume over its reference volume
    (exact: J of a trilinear map is at most quadratic in each reference coordinate)."""
    return jnp.mean(a, axis=(0, 1, 2))


def energy_density_parts(u: jax.Array, grid: Grid3, mat: Material, volumetric: str = "mean"):
    """(plus, minus, plus_v, minus_v): with ``"mean"``, the isochoric (plus, minus = 0) at the Gauss points and the
    volumetric (plus_v, minus_v) per element at Jbar; with ``"gauss"``, the law's whole (plus, minus) at the Gauss points
    and zero volumetric arrays. Infinite where J (or Jbar) <= 0."""
    J, I1 = invariants(deformation(u, grid))
    if volumetric == "gauss":
        plus, minus = law.energy_parts(J, I1, (), mat)
        z = jnp.zeros(grid.shape, u.dtype)
        return plus, minus, z, z
    iso, _ = law.energy_parts(J, I1, (), _isochoric(mat))
    Jb = element_mean(J)
    vp, vm = law.energy_parts(Jb, jnp.full_like(Jb, 3.0), (), _volumetric(mat))
    return iso, jnp.zeros_like(iso), vp, vm


def solid_energy(u: jax.Array, g: jax.Array, grid: Grid3, mat: Material, volumetric: str = "mean") -> jax.Array:
    """sum over Gauss points w (g psi_plus + psi_minus) [+ per element V_e (gbar psi_v_plus(Jbar) + psi_v_minus(Jbar))],
    ``g`` (2, 2, 2, nx, ny, nz) the degradation at the Gauss points."""
    w = gauss_weight(grid, u.dtype)
    plus, minus, vp, vm = energy_density_parts(u, grid, mat, volumetric)
    e = jnp.sum(w * (g * plus + minus))
    if volumetric == "gauss":
        return e
    return e + jnp.sum(8.0 * w * (element_mean(g) * vp + vm))


def tension_energy(u: jax.Array, grid: Grid3, mat: Material, volumetric: str = "mean") -> jax.Array:
    """The history field's source at the Gauss points: psi_plus (with ``"mean"``, the element's volumetric tension part
    added at each of its Gauss points)."""
    plus, _, vp, _ = energy_density_parts(u, grid, mat, volumetric)
    return plus if volumetric == "gauss" else plus + vp[None, None, None]


# ------------------------------------------------------------------------------------------------- the damage ---
def degradation(d: jax.Array, k_res: Any) -> jax.Array:
    """g = (1 - d)^2 + k_res at the Gauss points from nodal damage (nx+1, ny+1, nz+1)."""
    return (1.0 - gauss_values(d)) ** 2 + k_res


def damage_energy(d: jax.Array, H: jax.Array, grid: Grid3, gc: Any, l: Any) -> jax.Array:
    """AT2 with the history field: sum w [(1-d)^2 H + Gc (d^2/2l + l/2 |grad d|^2)], quadratic in nodal ``d``."""
    w = gauss_weight(grid, d.dtype)
    dq = gauss_values(d)
    G = gauss_gradients(d[..., None], grid)[..., 0, :]
    return jnp.sum(w * ((1.0 - dq) ** 2 * H + gc * dq ** 2 / (2.0 * l) + gc * 0.5 * l * jnp.sum(G * G, axis=-1)))


def crack_energy(d: jax.Array, grid: Grid3, gc: Any, l: Any) -> jax.Array:
    return damage_energy(d, jnp.zeros((2, 2, 2) + grid.shape, d.dtype), grid, gc, l)

