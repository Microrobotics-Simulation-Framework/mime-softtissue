"""The tip-shape property's planted errors are caught every run: each plant (``test_tip_shapes.PLANTS``) against 100
draws of the property's own ranges, ten runs of fresh draws; caught where any draw fails. The honest shapes pass the
same draws (the false-alarm arm)."""
import numpy as np
import pytest

from engine.test_tip_shapes import PLANTS, cone, headed, property_holds

RUNS, DRAWS = 10, 100


def failing_draws(seed: int) -> int:
    rng = np.random.default_rng(seed)
    s, t, depth = rng.uniform(0, 1, DRAWS), rng.uniform(1e-6, 0.04e-3, DRAWS), rng.uniform(0, 1.5e-3, DRAWS)
    return sum(bool(property_holds(cone(), headed(), *x)) for x in zip(s, t, depth))


@pytest.mark.parametrize("plant", sorted(PLANTS))
def test_each_plant_is_caught_every_run(plant):
    with PLANTS[plant]():
        caught = [failing_draws(seed) > 0 for seed in range(RUNS)]
    assert all(caught), f"{plant}: caught {sum(caught)} of {RUNS} runs"


def test_the_honest_shapes_pass_the_same_draws():
    assert [failing_draws(seed) for seed in range(RUNS)] == [0] * RUNS
