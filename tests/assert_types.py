"""Types an editor completes from, held by `assert_type`: pyright (strict) fails CI if one changes under you.

Autocomplete in an editor is only as good as the types it reads. MIME ships `py.typed` and this plugin's entries are
frozen dataclasses, so their settings complete; what a string names (`ctx.scalar("force")`, a record's name) cannot. These
lines hold what does complete. Add a line for each public entry and law you write. This file is never run by pytest (it
holds no test), only read by pyright: `pyright` in this folder, with the environment active or `--pythonpath
"$(which python)"` (CI does the second).
"""

from typing import assert_type

from mime.registry import Dynamics, Observation

from mime_softtissue.entries import GripForce, NotchedStrip

assert_type(NotchedStrip.name, str)
assert_type(NotchedStrip.validity, tuple[str, ...])
assert_type(GripForce.name, str)
assert issubclass(NotchedStrip, Dynamics)
assert issubclass(GripForce, Observation)
