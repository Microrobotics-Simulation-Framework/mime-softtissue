"""The shipped 3-D puncture runs (``studies.puncture3d_results``): each loads whole, and the findings the TIP3D report
quotes from them stay true of the files shipped (replacing a file in a way that changes a finding fails here)."""

from __future__ import annotations

import numpy as np

from mime_softtissue.studies import puncture3d_results as R

GRIDS = ("0.25", "0.177", "0.125")  # h in mm, each finer by sqrt(2)


def _force_at(name: str, depth: float) -> float:
    run = R.load(name)
    return float(np.interp(depth, run.rows["depth_m"], run.rows["force_N"]))


def _largest_dip(force: np.ndarray) -> float:
    """The largest fall below the force reached so far, as a share of it."""
    reached = np.maximum.accumulate(force)
    return float(np.max((reached - force) / np.where(reached > 0, reached, 1.0)))


def test_every_shipped_run_loads_with_its_rows_counts_and_surface_frames() -> None:
    assert R.names()
    for name in R.names():
        run = R.load(name)
        assert len(run.rows["depth_m"]) > 0 and np.all(np.isfinite(run.rows["force_N"])), name
        assert len(run.surface.points) == len(run.surface.damage) == len(run.surface.depth) >= 2, name
        assert run.counts["status"] in ("depth reached", "wall time"), name


def test_deletions_first_peak_halves_with_each_refinement_while_the_phase_fields_force_settles() -> None:
    deletion = [R.load(f"clot_cone_deletion_h{h}").puncture() for h in GRIDS]
    assert all(found for _, _, found in deletion)
    peaks = [f for f, _, _ in deletion]
    assert peaks[1] / peaks[0] < 0.6 and peaks[2] / peaks[1] < 0.6
    phase_field = [_force_at(f"clot_cone_h{h}", 1.5e-3) for h in GRIDS]
    assert abs(phase_field[2] - phase_field[1]) < abs(phase_field[1] - phase_field[0])
    assert abs(phase_field[2] / phase_field[1] - 1.0) < 0.02


def test_the_half_domain_carries_the_full_domains_solution_and_the_quarter_does_not() -> None:
    for depth in (1.0e-3, 2.0e-3):
        full = _force_at("symmetry_cone_full_h0.177", depth)
        assert abs(_force_at("symmetry_cone_half_h0.177", depth) / full - 1.0) < 1e-5
    assert _force_at("symmetry_cone_quarter_h0.177", 1.0e-3) / _force_at("symmetry_cone_full_h0.177", 1.0e-3) > 1.2


def test_the_sharp_cone_wedges_where_the_sphere_tipped_needle_punctures() -> None:
    cone = R.load("symmetry_cone_full_h0.177")
    assert cone.rows["d_max"][-1] == 1.0 and cone.rows["depth_m"][-1] >= 2.0e-3 - 1e-9
    assert _largest_dip(cone.rows["force_N"]) < 0.02
    sphere = R.load("clot_sphere_R0.25")
    force, _, found = sphere.puncture()
    assert found and _largest_dip(sphere.rows["force_N"]) > 0.4
