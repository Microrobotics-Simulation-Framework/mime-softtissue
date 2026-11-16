"""The pure-shear study's converged runs, as computed on CI (the study's report holds how): the calibrated constants,
each notch group's predicted curve and readouts, and one run's frames for the animation. Small files in ``results/``."""

from __future__ import annotations

import io
import json
from importlib import resources
from typing import Any

import numpy as np


def load() -> dict[str, Any]:
    """``calibrated`` (c1, c2 Pa; gc J/m^2; l_mm; h_mm; the calibration group: the constants the curves were run at),
    ``calibrated_raw`` (the constants re-derived on the raw data, with their uncertainty and fit card) and ``groups``
    (each group's model curve, peak force, its displacement and the crack-start displacement)."""
    return json.loads(resources.files(__package__).joinpath("results/pure_shear.json").read_text())


def frames() -> dict[str, Any]:
    """One run's saved fields (displacement in mm and damage at every second step, the grid's nodes in mm), for the
    animation."""
    raw = resources.files(__package__).joinpath("results/pure_shear_frames.npz").read_bytes()
    with np.load(io.BytesIO(raw)) as f:  # the damage is stored in float16 (display only)
        return {key: f[key].astype(np.float32) if f[key].dtype == np.float16 else f[key] for key in f.files}
