"""``softtissue.replayed_tear`` replays the saved frames: step k holds frame k, and its surface is the declared one.

A replay computes nothing, so the checks are that what a run holds is the saved frames (not a neighbour), that the run
saves and reloads, and that the surface it declares is the replay's own."""

from pathlib import Path

import numpy as np
from mime.experiment import Experiment
from mime.scene import Scene, Z
from mime.testing import round_trip_failures
from mime.things import Body, State
from mime.units import m, m_per_s2, s

import mime_softtissue.entries as st
from mime_softtissue.entries.replayed_tear import replay

COARSE = dict(every_column=16, every_frame=6)  # 3 kept frames of a few hundred vertices: the contract, not the picture


def replayed() -> Experiment:
    entry = st.ReplayedTear(**COARSE)
    experiment = Experiment("replayed_tear", scene=Scene.new(up=Z, units="mm"), gravity=m_per_s2(0.0),
                            dt=s(entry.step_seconds))
    experiment.add_body(Body("sheet", dynamics=entry, initial=State(position=m((0.0, 0.0, 0.0)))))
    return experiment


def test_step_k_holds_frame_k(tmp_path: Path):
    _, faces, points, damage, seconds = replay(16, 6, 3.6)
    assert len(points) > 3 and damage.max() > 0.9 and damage.min() == 0.0, "the run must contain a tear to be a test"
    steps = len(points) - 1
    replayed().build().run_to(tmp_path / "run", s(steps * seconds), history=True)
    held = np.load(tmp_path / "run" / "outputs.npz")
    assert np.array_equal(held["history/sheet/frame"], np.arange(1, steps + 1))
    np.testing.assert_allclose(held["history/sheet/points"], points[1:], rtol=0, atol=1e-12)
    np.testing.assert_allclose(held["history/sheet/damage"], damage[1:], rtol=0, atol=1e-12)
    assert faces.max() == points.shape[1] - 1


def test_the_declared_surface_is_the_replays():
    entry = st.ReplayedTear(**COARSE)
    mesh, = entry.surface_meshes({"to_mm": 3.6e-3})
    rest, faces, points, *_ = replay(16, 6, 3.6)
    assert mesh.display == "extruded" and [f.name for f in mesh.fields] == ["damage"]
    assert np.array_equal(mesh.rest, rest) and np.array_equal(mesh.faces, faces)
    assert mesh.problems() == []
    assert np.abs(points[0] - rest).max() < 5e-5, "frame 0 is the sheet after the first 0.05 mm of the pull"


def test_the_saved_round_trip(tmp_path: Path):
    assert round_trip_failures(replayed(), tmp_path / "saved") == []
