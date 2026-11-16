"""The controls: each planted error must be caught, every run. Two kinds.

- Property plants: a planted defect in the material law, the element, the damage solve or the history, run against the
  Hypothesis property it breaks, ``REPS`` times (a Hypothesis run is random; a control that fails most runs is not a
  control). Never derandomised, never fewer examples.
- Reproduction plants: a planted defect run against the reference comparison (``reproduction_cases``): a wrong Gauss
  weight at four sizes, the shear modulus off, the grid's Gc correction left out, the history not kept, the blade's edge
  node left out of the reaction. The comparison's line is stated in ``PREDICTIONS.md``: equal bits, otherwise 1e-9
  relative."""
import json
from pathlib import Path

import numpy as np
import pytest

import plants
import reproduction_cases as rc
import engine.test_properties as props
import engine.test_solver_properties as solver_props

REF = Path(__file__).resolve().parent / "reference"
REPS = 3  # local measurement used 10; CI repeats fewer to stay inside its budget (never fewer examples a run)

PROPERTY_PLANTS = [
    ("rotation invariance", props.check_rotation_invariance, plants.I1_from_diagonal_only),
    ("zero at the identity", props.check_zero_and_stationary_at_identity, plants.energy_offset_at_identity),
    ("stationary at the identity", props.check_zero_and_stationary_at_identity, plants.volumetric_term_not_stationary),
    ("symmetric Hessian", props.check_hessian_symmetric, plants.hessian_row_scaled),
    ("assembled stiffness matches the gradient (row scaled)", props.check_assembled_stiffness_matches_the_gradient,
     plants.hessian_row_scaled),
    ("assembled stiffness matches the gradient (blocks transposed)", props.check_assembled_stiffness_matches_the_gradient,
     plants.assembly_upper_block_transposed),
    ("damage in [0, 1]", props.check_damage_in_unit_interval, plants.damage_not_clipped),
    ("history never decreases", props.check_history_function_never_decreases, plants.history_not_kept),
    ("history never decreases through unloading (solver)", solver_props.check_history_through_unloading,
     plants.history_not_kept),
]


def caught(check, plant) -> bool:
    reset = getattr(check, "reset", lambda: None)
    reset()
    try:
        with plant():
            try:
                check()
            except Exception:  # a failed property, or a defect that stops the code: both are a catch
                return True
        return False
    finally:
        reset()


@pytest.mark.parametrize("name,check,plant", PROPERTY_PLANTS, ids=[p[0] for p in PROPERTY_PLANTS])
def test_property_plant_is_caught_every_run(name, check, plant):
    results = [caught(check, plant) for _ in range(REPS)]
    assert all(results), f"{name}: caught {sum(results)} of {REPS}"


def test_property_checks_pass_on_the_honest_engine():
    for _, check, _ in PROPERTY_PLANTS:
        check()


# --- reproduction plants -------------------------------------------------------------------------------------------

def detector() -> tuple[bool, float]:
    """(equal bits, worst relative departure) of the cheap comparison: the blade case's 12 steps and the strip's first
    20 steps, against the reference."""
    ref5, ref1 = np.load(REF / "R5.npz"), np.load(REF / "R1.npz")
    worst, bits = 0.0, True
    arrays, _ = rc.R5()
    for key in ("force", "energy", "crack_energy", "final_u", "final_d"):
        a, b = arrays[key], ref5[key]
        bits &= bool(np.array_equal(a, b))
        worst = max(worst, float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)))
    r = rc.run_strip_prefix(20)
    for key in ("force", "energy", "crack_energy"):
        a, b = r[key], ref1[key][:20]
        bits &= bool(np.array_equal(a, b))
        worst = max(worst, float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)))
    return bits, worst


def test_the_honest_engine_is_within_the_line_of_the_reference():
    bits, worst = detector()
    assert bits or worst <= 1e-9, worst


REPRODUCTION_PLANTS = [
    ("Gauss weight 1e-3", lambda: plants.gauss_weight(1e-3)),
    ("Gauss weight 1e-6", lambda: plants.gauss_weight(1e-6)),
    ("Gauss weight 1e-9", lambda: plants.gauss_weight(1e-9)),
    ("Gauss weight 1e-12", lambda: plants.gauss_weight(1e-12)),
    ("shear modulus x (1 + 1e-6)", lambda: plants.shear_modulus_scaled(1 + 1e-6)),
    ("Gc grid correction left out", plants.gc_correction_left_out),
    ("history not kept", plants.history_not_kept),
]


@pytest.mark.parametrize("name,plant", REPRODUCTION_PLANTS, ids=[p[0] for p in REPRODUCTION_PLANTS])
def test_reproduction_plant_leaves_the_line(name, plant):
    """Caught = the comparison reports a departure past the line (1e-9) ... except the plants below it, which only the
    bit-equality line sees; their resolution is what ``REFACTOR_PLUGIN.md`` reports."""
    with plant():
        bits, worst = detector()
    below_line = name in ("Gauss weight 1e-9", "Gauss weight 1e-12")
    if below_line:
        assert not bits, "an error below the 1e-9 line must still change bits"
    else:
        assert worst > 1e-9, f"{name}: worst relative departure {worst:.2e}, not caught by the 1e-9 line"


def test_blade_edge_node_left_out_of_the_reaction_is_caught():
    arrays, _ = rc.R5(specimen_kw={"reaction_skips_edge": True})
    ref = np.load(REF / "R5.npz")["force"]
    assert float(np.max(np.abs(arrays["force"] - ref) / np.abs(ref))) > 1e-2
