"""Write the strip benchmark as a saved MIME experiment, `benchmarks/tutwiler_strip/`, with its `mime.toml`.

    python benchmarks/build_strip.py       # about a minute: the saved form runs the experiment's first steps
    mime check benchmarks/tutwiler_strip   # what is still to write in mime.toml (the creator's ORCID, ...)
    mime verify benchmarks/tutwiler_strip  # loads it, rebuilds it, and reruns its first steps bit for bit

`benchmarks/mime.toml` is the record's manifest (what is shared and its metadata, see `mime push`); it is copied beside the
saved form, which is the folder `mime check`, `mime pack` and `mime push` read.
"""

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "examples"))
from tutwiler_strip import strip_experiment  # noqa: E402 - examples/tutwiler_strip.py

FOLDER = HERE / "tutwiler_strip"

if __name__ == "__main__":
    if FOLDER.exists():
        shutil.rmtree(FOLDER)
    experiment = strip_experiment()
    # The licence the saved files are shared under: it is bound to them when the experiment is saved.
    experiment.licence("CC-BY-4.0")
    experiment.save(FOLDER)
    shutil.copy(HERE / "mime.toml", FOLDER / "mime.toml")
    print("saved", sorted(path.name for path in FOLDER.iterdir()))
