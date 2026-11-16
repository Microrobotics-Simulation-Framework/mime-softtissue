"""The studies' converged runs, as computed on CI, in small files in ``results/``: the pure-shear study's calibrated
constants, each notch group's predicted curve and readouts, and one run's frames (``load``, ``frames``); the puncture
tutorial's curves and run folders (``puncture``, ``puncture_run``); and the published puncture data it is compared with
(``fregonese_fig5``)."""

from __future__ import annotations

import io
import json
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np


def load() -> dict[str, Any]:
    """``calibrated`` (c1, c2 Pa; gc J/m^2; l_mm; h_mm; the calibration group) and ``groups`` (each group's model
    curve, peak force, its displacement and the crack-start displacement)."""
    return json.loads(resources.files(__package__).joinpath("results/pure_shear.json").read_text())


def frames() -> dict[str, Any]:
    """One run's saved fields (displacement in mm and damage at every second step, the grid's nodes in mm), for the
    animation."""
    raw = resources.files(__package__).joinpath("results/pure_shear_frames.npz").read_bytes()
    with np.load(io.BytesIO(raw)) as f:  # the damage is stored in float16 (display only)
        return {key: f[key].astype(np.float32) if f[key].dtype == np.float16 else f[key] for key in f.files}


def puncture() -> dict[str, Any]:
    """The puncture tutorial's converged runs: ``about``, and ``runs`` (each by name: its settings, depth in mm, force in
    N and crack area in mm^2 at every step, and where it ran) and ``reference`` (spike 3D-COST's rows the 3-D punch is
    checked against)."""
    return json.loads(resources.files(__package__).joinpath("results/puncture.json").read_text())


def puncture_run(name: str) -> Path:
    """A converged run's folder (its stage and recording, for ``mime.viz.display``)."""
    return Path(str(resources.files(__package__).joinpath("results/puncture_runs", name)))


def fregonese_fig5() -> dict[str, Any]:
    """Fregonese and Bacca's Fig. 5 (Fakhouri et al.'s gel, "Exp. 1", and their theory for Ogden alpha = 2 to 9),
    digitised: ``exp1`` points with their error bars, ``theory`` curves, and how they were read."""
    return json.loads(resources.files(__package__).joinpath("data/fregonese2021_fig5.json").read_text())
