"""Properties of the solver's state through a load path (Hypothesis): the history field never decreases, also when the
grip is unloaded, and the damage stays in [0, 1]. A coarse strip (h = 0.75 mm); the draws are loads a
strip sees (up to 8 mm of grip displacement, the peak being near 7 mm at the benchmark's h)."""
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings, strategies as st

import reproduction_cases as rc
from mime_softtissue.engine.solver import Solver

settings.register_profile("solver", deadline=None)
settings.load_profile("solver")
_SOLVER = None


def solver_and_material():
    global _SOLVER
    if _SOLVER is None:
        _SOLVER = Solver(rc.strip(10.5e-3, 0.75e-3), "phase_field", tol_d=1e-5), rc.strip_material(0.75e-3)
    return _SOLVER


def check_history_and_damage_through_a_load_path(up, down):
    """Load to ``up`` in two steps, then unload to ``down * up``: the history never decreases, the damage is in [0, 1]."""
    solver, mat = solver_and_material()
    state = solver.initial()
    previous = state
    for delta in (up / 2, up, down * up):
        state, _ = solver.step(state, jnp.asarray(delta), mat)
        assert bool(jnp.all(state.history >= previous.history)), f"history fell at delta {delta}"
        d = np.asarray(state.d)
        assert d.min() >= 0.0 and d.max() <= 1.0
        # Not asserted: that the nodal damage never decreases. Irreversibility is the history field's; the damage solved
        # from it is not monotone in it pointwise (measured: a nodal decrease of 2e-3 in a loading step, up = 7.8 mm).
        previous = state


@settings(max_examples=100)
@given(st.floats(2e-3, 8e-3, allow_nan=False), st.floats(0.2, 0.9, allow_nan=False))
def check_history_through_unloading(up, down):
    check_history_and_damage_through_a_load_path(up, down)


def test_history_and_damage_through_a_load_path():
    check_history_through_unloading()


def reset():
    """A solver's step is jitted when it is built: a plant (``tests/plants.py``) takes effect only in a new one."""
    global _SOLVER
    _SOLVER = None


check_history_through_unloading.reset = reset
