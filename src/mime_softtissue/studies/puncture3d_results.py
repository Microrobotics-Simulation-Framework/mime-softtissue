"""The 3-D engine's own puncture runs (``engine.puncture3d``, outside MIME: element deletion and adaptive load steps have
no MIME build; TIP3D's report holds how they ran), as ``engine.puncture3d.PunctureRun`` objects, each read from its two
records (``softtissue.puncture3d.<name>.rows``, the rows and the block's boundary surface at its saved frames, and
``.meta``, its counts, settings and case), which the run index lists (``results.puncture()["engine_runs"]``)."""

from __future__ import annotations

from ..engine.puncture3d import PunctureRun, load_run
from .results import puncture, record_path


def names() -> list[str]:
    """The runs shipped with the package."""
    return sorted(puncture()["engine_runs"])


def load(name: str) -> PunctureRun:
    """One shipped run by name (``names()``)."""
    held = puncture()["engine_runs"][name]
    return load_run(str(record_path(held["rows"])), str(record_path(held["meta"])))
