"""The edge-notched strip as a verification benchmark under this plugin's own id prefix (SOFTTISSUE-VER-nnn), so it lands
in the same evidence tables as MIME's.

**What it is and is not.** It bounds the phase-field model's self-consistency: the toughness an experimenter would infer
from the model's own force and stored energy (Irwin's G = -(1/t) dU/da at fixed grip displacement, the exact nonlinear
form of Tutwiler et al. 2020, Eq. 1) against the toughness that went in. It is **not a comparison against measured
forces** (the paper's force curves were not read) and states no accuracy against fibrin. The grip spacing is assumed,
the Poisson ratio is a placeholder."""
import numpy as np
import pytest
from maddening.core.compliance.validation import BenchmarkType, verification_benchmark

import reproduction_cases as rc
from mime_softtissue.engine.readouts.toughness import implied

GC = rc.GC


@pytest.mark.slow
@verification_benchmark(
    benchmark_id="SOFTTISSUE-VER-001",
    description="The toughness implied by the phase-field strip's own force and stored energy (Irwin's exact nonlinear "
                "G = -(1/t) dU/da at the peak's grip displacement) recovers the input Gc, notch 10.5 mm, h = 0.3 mm",
    node_type="NotchedStrip",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria="|G_implied / Gc - 1| <= 0.20 at notch 10.5 mm, l = 0.3 mm, h = l (prototype: 0.05 for h = 0.3)",
    references=("Tutwiler2020",),
)
def test_implied_toughness_recovers_the_input_toughness():
    """Acceptance criterion: |G_implied / Gc - 1| <= 0.20 [the line stated before the prototype's runs: within 20 % at
    the 10.5 mm notch]. An acceptance test of the model's self-consistency, not an accuracy claim against measurement."""
    arrays, _ = rc.R1()
    _, delta_star, _ = rc.peak_parabola(arrays["delta"], arrays["force"])
    lo, hi = rc.elastic(10.2), rc.elastic(10.8)
    g = implied(lo, hi, 10.2e-3, 10.8e-3, delta_star, rc.THICKNESS)["exact"]
    assert abs(g / GC - 1.0) <= 0.20, f"G_implied/Gc = {g / GC:.3f}"
