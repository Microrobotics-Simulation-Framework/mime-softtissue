"""The control without a length scale: an element's damage is set to 1 for good once the largest principal stretch at
its centre passes ``stretch_limit`` (element deletion on a fixed grid), repeated until no element is newly deleted.
Its toughness depends on the grid, which is what the control shows."""

from __future__ import annotations

import jax.numpy as jnp


def damage_at_gauss(d: jnp.ndarray) -> jnp.ndarray:
    """An element's damage at its four Gauss points (the same at each)."""
    return jnp.broadcast_to(d[:, None], (d.shape[0], 4))


def newly_deleted(stretch: jnp.ndarray, d: jnp.ndarray, limit) -> jnp.ndarray:
    return (stretch >= limit) & (d < 0.5)
