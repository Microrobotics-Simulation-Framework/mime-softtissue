"""Test configuration, and the fixture every adapter test shares.

Pin to one core and the CPU when you run these: `JAX_PLATFORMS=cpu taskset -c 0 pytest`. An unpinned JAX run on a shared
machine oversubscribes the cores and was measured five times slower for the same answer.

This plugin's reference numbers (tests/reference) are float64 bits, so the suite runs in float64 by default
(`JAX_ENABLE_X64=1`, set here unless the environment says otherwise); the scaffold's advice to run in both dtypes does not
apply to the reproduction, whose references exist in one.
"""

import os
from collections.abc import Iterator

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from mime.experiment import Experiment  # noqa: E402 - after the environment is set, before JAX starts

import plants  # noqa: E402 - tests/plants.py
from tutwiler_strip import strip_experiment  # noqa: E402 - examples/tutwiler_strip.py


def coarse_strip() -> Experiment:
    """The strip at h = 0.75 mm, standing in for the benchmark's 0.3 mm: the adapter checks are about the entries' contracts,
    not the physics, and a node battery steps the solve hundreds of times."""
    return strip_experiment(h_mm=0.75, notch_mm=10.5)


@pytest.fixture
def experiment() -> Experiment:
    return coarse_strip()


@pytest.fixture(autouse=True)
def planted_from_the_environment() -> Iterator[None]:
    """Where ``MIME_PLANT_ERROR`` is set, put the planted error (tests/plants.py) into the engine for the test. This is how a
    control is run from outside: `MIME_PLANT_ERROR=1 pytest tests/engine/test_properties.py::test_rotation_invariance` must
    FAIL, on every run, and `.github/probe/detection_rate.py` of MIME counts how often (`--plant-var MIME_PLANT_ERROR`)."""
    if os.environ.get(plants.PLANT_VARIABLE):
        with plants.PLANTED_FROM_THE_ENVIRONMENT():
            yield
    else:
        yield
