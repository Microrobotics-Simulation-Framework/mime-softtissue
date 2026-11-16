"""The 3-D puncture runs computed on CI runners (TIP3D's report holds how), as ``engine.puncture3d.PunctureRun`` objects:
small files in ``results/puncture3d/`` (``<name>.npz`` and ``<name>.json``), each with its rows, counts, settings and the
block's boundary surface at its saved frames."""

from __future__ import annotations

from importlib import resources

from ..engine.puncture3d import PunctureRun, load_run

_FOLDER = "results/puncture3d"


def names() -> list[str]:
    """The runs shipped with the package."""
    folder = resources.files(__package__).joinpath(_FOLDER)
    return sorted(p.name[:-4] for p in folder.iterdir() if p.name.endswith(".npz"))


def load(name: str) -> PunctureRun:
    """One shipped run by name (``names()``)."""
    folder = resources.files(__package__).joinpath(_FOLDER)
    with resources.as_file(folder.joinpath(name + ".npz")) as npz, resources.as_file(folder.joinpath(name + ".json")):
        return load_run(str(npz)[:-4])
