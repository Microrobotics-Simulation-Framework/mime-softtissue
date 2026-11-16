"""The `mime.entries` entry point (pyproject.toml): MIME imports this module when it lists what is installed (`mime
registry`) or loads a saved experiment that names the namespace `softtissue`, so importing it registers every entry, kind
by kind. A script imports an entry from its kind, as it does MIME's: `from mime_softtissue.dynamics import GelBlock`.
The arithmetic lives in `mime_softtissue.engine`, which imports no `mime`."""

from . import dynamics, interactions, observations, scene

__all__ = ["dynamics", "interactions", "observations", "scene"]
