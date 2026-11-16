"""A rectilinear grid of quadrilateral cells, numbered column by column, and the row spacing that grades from a fine
band to coarse cells. No MIME import."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics.plane_strain import COLUMN, ROW


@dataclass(frozen=True)
class Grid:
    """Node coordinates ``x`` (nx+1) and ``y`` (ny+1); elements numbered column by column."""

    x: np.ndarray
    y: np.ndarray

    @property
    def nx(self) -> int:
        return len(self.x) - 1

    @property
    def ny(self) -> int:
        return len(self.y) - 1

    @property
    def n_elements(self) -> int:
        return self.nx * self.ny

    def element_columns(self) -> tuple[np.ndarray, np.ndarray]:
        i, j = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
        return i.ravel(), j.ravel()

    def element_nodes(self) -> tuple[np.ndarray, np.ndarray]:
        i, j = self.element_columns()
        return i[:, None] + COLUMN[None, :], j[:, None] + ROW[None, :]

    def element_sizes(self) -> tuple[np.ndarray, np.ndarray]:
        i, j = self.element_columns()
        return np.diff(self.x)[i], np.diff(self.y)[j]


def graded_rows(height: float, h: float, fine: float, ratio: float = 1.25, cap: float = 8.0) -> np.ndarray:
    """Row coordinates from 0 to ``height``: uniform ``h`` up to ``fine``, then cells growing by ``ratio`` up to
    ``cap`` times ``h``, then uniform; the last stretch rescaled to land on ``height``."""
    rows = list(np.arange(0.0, fine + 0.5 * h, h))
    size = h
    while rows[-1] < height - 1e-12:
        size = min(size * ratio, cap * h)
        rows.append(rows[-1] + size)
    rows_a = np.asarray(rows)
    start = int(np.searchsorted(rows_a, fine + 0.5 * h))
    tail = rows_a[start - 1:]
    scaled = rows_a[start - 1] + (tail - tail[0]) * (height - tail[0]) / (tail[-1] - tail[0])
    return np.concatenate([rows_a[:start - 1], scaled])
