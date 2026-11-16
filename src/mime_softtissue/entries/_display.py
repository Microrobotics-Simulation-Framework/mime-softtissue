"""The block's section shown as a surface, for display alone: its outer boundary (the top face from the axis to the rim,
then the side down to the bottom) revolved through ``SWEEP`` about the axis in ``SEGMENTS`` steps, and the section's
grid on the two cut faces of the sweep, so that the damage inside the block shows. Each vertex is a node of the grid at
an angle; the points are the nodes' deformed positions turned to that angle, and the damage the node's."""

from __future__ import annotations

import math

import numpy as np

#: Facets of the sweep, and the sweep: three quarters of a turn, the quarter left open to show the section.
SEGMENTS = 36
SWEEP = 1.5 * math.pi


def revolved(nx: int, ny: int, segments: int = SEGMENTS, sweep: float = SWEEP) -> tuple[np.ndarray, np.ndarray,
                                                                                          np.ndarray, np.ndarray]:
    """Each vertex's node ``(i, j)`` and angle, and the triangles: the pole (the top's axis node, once), the boundary's
    other nodes at ``segments + 1`` angles, then each cut face's grid (``(nx + 1)(ny + 1)`` nodes at its angle)."""
    profile = [(i, ny) for i in range(1, nx + 1)] + [(nx, j) for j in range(ny - 1, -1, -1)]
    angles = np.linspace(0.0, sweep, segments + 1)
    nodes_i, nodes_j, theta = [0], [ny], [0.0]
    ring = np.zeros((len(profile), segments + 1), dtype=np.int64)
    for a, angle in enumerate(angles):
        for k, (i, j) in enumerate(profile):
            ring[k, a] = len(nodes_i)
            nodes_i.append(i)
            nodes_j.append(j)
            theta.append(float(angle))
    faces: list[tuple[int, int, int]] = []
    for a in range(segments):  # the fan about the pole, then the quads down the profile, counter-clockwise from outside
        faces.append((0, int(ring[0, a]), int(ring[0, a + 1])))
        for k in range(len(profile) - 1):
            p, q, r, s = int(ring[k, a]), int(ring[k + 1, a]), int(ring[k + 1, a + 1]), int(ring[k, a + 1])
            faces += [(p, q, r), (p, r, s)]
    for angle, flip in ((0.0, True), (sweep, False)):  # the cut faces, each facing out of the sweep
        start = len(nodes_i)
        grid = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1) + start
        for i in range(nx + 1):
            for j in range(ny + 1):
                nodes_i.append(i)
                nodes_j.append(j)
                theta.append(float(angle))
        a_, b_, c_, d_ = grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]
        tri = np.concatenate([np.stack([a_, b_, c_], -1).reshape(-1, 3), np.stack([a_, c_, d_], -1).reshape(-1, 3)])
        faces += [tuple(int(v) for v in (row[::-1] if flip else row)) for row in tri]
    return (np.asarray(nodes_i, dtype=np.int64), np.asarray(nodes_j, dtype=np.int64), np.asarray(theta),
            np.asarray(faces, dtype=np.int64))


def placed(x: np.ndarray, y: np.ndarray, u: object, i: np.ndarray, j: np.ndarray, theta: np.ndarray, xp: object = np) -> object:
    """The vertices' points: node ``(i, j)``'s radius and height, displaced by ``u`` (``None``: at rest), turned to its
    angle; ``xp`` the array module (NumPy at build, JAX in the step)."""
    m: object = xp
    r = m.asarray(x)[i]  # type: ignore[attr-defined]
    z = m.asarray(y)[j]  # type: ignore[attr-defined]
    if u is not None:
        r = r + u[i, j, 0]  # type: ignore[index]
        z = z + u[i, j, 1]  # type: ignore[index]
    return m.stack([r * m.cos(theta), r * m.sin(theta), z], axis=1)  # type: ignore[attr-defined]


def box_boundary(nx: int, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    """A 3-D block's boundary for display, ``nx ny nz`` cells: its top and its four sides (the bottom is held and not
    shown), each face its own grid of nodes. The faces at ``i = 0`` and ``j = 0`` are the planes of symmetry where the
    block is a quarter or a half (where a crack inside the block shows), its outer sides otherwise. Each vertex's node
    ``(i, j, k)`` (``(V, 3)``) and the triangles, counter-clockwise seen from outside the block."""
    nodes: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    start = 0

    def face(fixed_axis: int, index: int, outward: bool) -> None:
        nonlocal start
        sizes = [nx + 1, ny + 1, nz + 1]
        a_axis, b_axis = [axis for axis in range(3) if axis != fixed_axis]
        na, nb = sizes[a_axis], sizes[b_axis]
        A, B = np.meshgrid(np.arange(na), np.arange(nb), indexing="ij")
        ijk = np.zeros((na * nb, 3), dtype=np.int64)
        ijk[:, fixed_axis] = index
        ijk[:, a_axis] = A.ravel()
        ijk[:, b_axis] = B.ravel()
        grid = np.arange(na * nb).reshape(na, nb) + start
        p, q, r, s = grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]
        tri = np.concatenate([np.stack([p, q, r], -1).reshape(-1, 3), np.stack([p, r, s], -1).reshape(-1, 3)])
        # (a, b, fixed) right-handed: counter-clockwise in (a, b) faces +fixed; x: (y, z), y: (x, z) is left-handed
        positive = (fixed_axis != 1)
        faces.append(tri if positive == outward else tri[:, ::-1])
        nodes.append(ijk)
        start += na * nb

    face(2, nz, True)  # the top
    face(0, nx, True)  # the outer side x = half-width
    face(1, ny, True)  # the outer side y = half-width
    face(0, 0, False)  # the side at i = 0 (x = 0: a symmetry plane in a quarter)
    face(1, 0, False)  # the side at j = 0 (y = 0: a symmetry plane in a quarter or a half)
    return np.concatenate(nodes), np.concatenate(faces).astype(np.int64)
