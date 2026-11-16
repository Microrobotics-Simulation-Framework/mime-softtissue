"""A specimen's load path: the steps of a load, one after another, as numpy results (``run``) or as one traced array
for ``jax.jvp`` (``forces``)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .material import Material
from .solver import Solver, StepOut


def run(solver: Solver, mat: Material, deltas: Any, *, notch: Any = None, keep_fields: bool = False,
        stop_below: float | None = None, verbose: bool = False, after_peak: int = 10 ** 6,
        keep_u: bool = False) -> dict[str, Any]:
    """The load path over the displacements ``deltas`` (total), step by step. ``stop_below``: stop once the force
    has fallen below that share of its peak, or ``after_peak`` steps after the largest force so far (primal runs
    only)."""
    state = solver.initial()
    rows: list[StepOut] = []
    fields: list[np.ndarray] = []
    u_fields: list[np.ndarray] = []
    peak, at = 0.0, 0
    for n, delta in enumerate(deltas):
        state, out = solver.step(state, jnp.asarray(delta), mat, notch)
        rows.append(jax.device_get(out))
        if keep_fields:
            fields.append(np.asarray(state.d))
        if keep_u:
            u_fields.append(np.asarray(state.u))
        f = float(out.force)
        if f > peak:
            peak, at = f, n
        if verbose:
            print(f"step {n:4d} delta {float(delta)*1e3:8.4f} mm force {f*1e3:9.4f} mN it {int(out.iterations):4d} "
                  f"newton {int(out.newton):4d} res {float(out.residual):.2e}", flush=True)
        if (stop_below is not None and n > 5 and f < stop_below * peak) or n - at >= after_peak:
            break
    result = {key: np.asarray([getattr(r, key) for r in rows]) for key in StepOut._fields}
    result["delta"] = np.asarray(deltas[:len(rows)])
    if keep_fields:
        result["damage"] = np.stack(fields)
    if keep_u:
        result["u_fields"] = np.stack(u_fields)
    result["final_u"] = np.asarray(state.u)
    return result


def forces(solver: Solver, mat: Material, deltas: Any, notch: Any = None) -> jax.Array:
    """The force at every step as one traced array (for ``jax.jvp``): no early stop."""
    state = solver.initial()
    out = []
    for delta in deltas:
        state, o = solver.step(state, jnp.asarray(delta), mat, notch)
        out.append(o.force)
    return jnp.stack(out)
