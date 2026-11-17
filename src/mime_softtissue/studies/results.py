"""The studies' shipped results, each read through its record (``mime.assets``: named, pinned by its sha256, its licence
and credit stated in ``records/records.json``, so a changed byte is refused): the pure-shear study's calibrated constants
and curves (``load``) and one run's frames (``frames``); the puncture tutorial's run index (``puncture``), each MIME run's
result as MIME's own ``RunResult`` (``puncture_result``), a run's stage and recording for the view (``puncture_run``) and
the 3-D engine's own runs (``studies.puncture3d_results``); and the published puncture data it is compared with
(``fregonese_fig5``) and the saved experiment one's own starts from (``saved``)."""

from __future__ import annotations

import io
import json
import shutil
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from mime.assets import reference, resolve
from mime.results import RunResult

#: The pure-shear study's records: its calibrated constants and curves, and its group-4 run's frames.
PURE_SHEAR, PURE_SHEAR_FRAMES = "softtissue.pure_shear_results", "softtissue.pure_shear_tear_g4"
#: The puncture tutorial's run index.
PUNCTURE_RUNS = "softtissue.puncture_runs"


def record_path(name: str) -> Path:
    """The local file of the record ``name``, through ``mime.assets.resolve`` (its sha256 checked, its licence's gate
    applied)."""
    return Path(resolve(reference(name)))


def record_json(name: str) -> dict[str, Any]:
    """The record ``name``, a JSON file, read."""
    return json.loads(record_path(name).read_text(encoding="utf-8"))


def load() -> dict[str, Any]:
    """``calibrated`` (c1, c2 Pa; gc J/m^2; l_mm; h_mm; the calibration group) and ``groups`` (each group's model
    curve, peak force, its displacement and the crack-start displacement)."""
    return record_json(PURE_SHEAR)


def frames(name: str = PURE_SHEAR_FRAMES) -> dict[str, Any]:
    """One run's saved fields (displacement in mm and damage at every second step, the grid's nodes in mm), for the
    animation, from the record ``name``."""
    raw = record_path(name).read_bytes()
    with np.load(io.BytesIO(raw)) as f:  # the damage is stored in float16 (display only)
        return {key: f[key].astype(np.float32) if f[key].dtype == np.float16 else f[key] for key in f.files}


def puncture() -> dict[str, Any]:
    """The puncture tutorial's run index: ``about``; ``runs`` (each MIME run by name: the settings it ran at, where it
    ran, and the record of its result, read by ``puncture_result``); ``engine_runs`` (the 3-D engine's own runs outside
    MIME, read by ``studies.puncture3d_results``); ``sweeps`` (the needle sweep's members, a ``Study`` run in memory:
    each one's depth in mm and force in N, kept here because no study folder was written) and ``reference`` (spike
    3D-COST's rows the 3-D punch is checked against, the same solver outside MIME)."""
    return record_json(PUNCTURE_RUNS)


def puncture_result(name: str) -> RunResult:
    """A MIME run's result (``RunResult.read`` of its record): its observations' series with their units and flags,
    ``result["softtissue.tip_force:clot"].force``, ``.depth``, and ``result["softtissue.crack_size:clot"].area``."""
    return RunResult.read(record_path(puncture()["runs"][name]["result"]))


def puncture_run(name: str) -> Path:
    """A run's folder for ``mime.viz.display``, made from its records (``softtissue.puncture_run.<name>.stage``,
    ``.recording`` and ``.run``, each resolved and checked) in a new temporary folder."""
    folder = Path(tempfile.mkdtemp(prefix=f"softtissue_{name}_")) / name
    (folder / "experiment/scene").mkdir(parents=True)
    shutil.copyfile(record_path(f"softtissue.puncture_run.{name}.stage"), folder / "experiment/scene/scene.usda")
    shutil.copyfile(record_path(f"softtissue.puncture_run.{name}.recording"), folder / "recording.usdc")
    shutil.copyfile(record_path(f"softtissue.puncture_run.{name}.run"), folder / "run.json")
    return folder


def fregonese_fig5() -> dict[str, Any]:
    """Fregonese and Bacca's Fig. 5 (Fakhouri et al.'s gel, "Exp. 1", and their theory for Ogden alpha = 2 to 9),
    digitised: ``exp1`` points with their error bars, ``theory`` curves, and how they were read."""
    return json.loads(resources.files(__package__).joinpath("data/fregonese2021_fig5.json").read_text())


def saved(name: str) -> Path:
    """A saved experiment shipped with the plugin (``studies/saved/<name>``), to base one's own on:
    ``Experiment.based_on(saved("clot_puncture"), name="my_clot")``."""
    return Path(str(resources.files(__package__).joinpath("saved", name)))
