"""No hidden data: every file the plugin ships as data is a declared record, and every record's bytes are the ones it
names.

A result the notebooks show is read through MIME's records (named, pinned by its sha256, its licence and credit stated in
`records/records.json`), never from a file in the package that nothing declares. These read the manifest and the package's
files directly (no fetch, no cache) and fail on a data file no record names, on a record whose bytes differ from its
sha256 or size, and on package data outside the records that is not on the known interim list. Each has a control that
must fail: a planted unnamed file, a planted changed byte.

What they cannot see: whether a record's licence and credit are true (a person states them), and data a module builds
at import from a constant."""

import hashlib
import json
from pathlib import Path

import numpy as np

import mime_softtissue.records
from mime.results import RunResult
from mime_softtissue.studies import results

RECORDS = Path(mime_softtissue.records.__file__).parent
PACKAGE = RECORDS.parent
SOURCE = "package:softtissue/"

#: Package data outside the records, each an interim with its home (PLUGIN-CLEANUP's report lists them): the digitised
#: published data the studies compare with (records once the raw-data branch's rewrite of them lands) and the saved
#: experiment one's own starts from (MIME's own saved form, read and checked by `Experiment.based_on`; its published
#: record later). Held as the exact set: a new file outside the records fails here; one that becomes a record leaves it.
NOT_YET_RECORDS = {
    "studies/data/fregonese2021_fig5.json",
    "studies/data/lohr2025_digitised.json",
    "studies/data/rausch_visco_na0wxy.json",
    "studies/saved/clot_puncture/experiment.mime.yaml",
    "studies/saved/clot_puncture/mime.lock",
    "studies/saved/clot_puncture/reference.npz",
    "studies/saved/clot_puncture/scene/scene.usda",
}


def declared() -> dict[str, dict]:
    return {r["name"]: r for r in json.loads((RECORDS / "records.json").read_text(encoding="utf-8"))["assets"]}


def packaged(records: dict[str, dict]) -> dict[str, str]:
    """Each package file a record names (relative to the records' folder) and the record."""
    return {location[len(SOURCE):]: name for name, r in records.items() for location in r["locations"]
            if location.startswith(SOURCE)}


def unnamed(root: Path, records: dict[str, dict]) -> list[str]:
    """The data files under ``root``'s ``data/`` that no record names."""
    named = packaged(records)
    return sorted(p.relative_to(root).as_posix() for p in (root / "data").rglob("*")
                  if p.is_file() and p.relative_to(root).as_posix() not in named)


def mismatched(root: Path, records: dict[str, dict]) -> list[str]:
    """The records whose package file's sha256 or size is not the record's."""
    found = []
    for path, name in packaged(records).items():
        body = (root / path).read_bytes()
        if hashlib.sha256(body).hexdigest() != records[name]["sha256"] or len(body) != records[name]["size"]:
            found.append(name)
    return found


def test_every_shipped_data_file_is_a_record():
    records = declared()
    assert len(packaged(records)) > 30, "the manifest names the shipped results: the test would pass on nothing"
    assert unnamed(RECORDS, records) == []


def test_every_record_is_the_bytes_it_names():
    assert mismatched(RECORDS, declared()) == []


def test_the_data_outside_the_records_is_exactly_the_known_interim_set():
    found = {p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob("*")
             if p.is_file() and p.suffix not in (".py", ".pyc") and "__pycache__" not in p.parts
             and not p.is_relative_to(RECORDS)}
    assert found == NOT_YET_RECORDS


def test_a_planted_unnamed_file_and_a_changed_byte_are_found(tmp_path: Path):
    """The controls: a copy of the records' folder with one file added that no record names, and one byte changed in
    a named file, must each be found, every time."""
    records = declared()
    path, name = sorted(packaged(records).items())[0]
    copy = tmp_path / "records"
    (copy / Path(path).parent).mkdir(parents=True)
    (copy / path).write_bytes((RECORDS / path).read_bytes())
    only = {name: records[name]}
    assert unnamed(copy, only) == [] and mismatched(copy, only) == []
    (copy / "data" / "stray.npz").write_bytes(b"not declared")
    assert unnamed(copy, only) == ["data/stray.npz"]
    body = bytearray((copy / path).read_bytes())
    body[len(body) // 2] ^= 1
    (copy / path).write_bytes(bytes(body))
    assert mismatched(copy, only) == [name]


def test_a_mime_runs_record_reads_back_as_its_result():
    """A shipped run is MIME's own result: read through its record, its series carry their units, and its last
    force is the one the run index states as its peak's neighbour (the run's own numbers, not a copy)."""
    index = results.puncture()
    assert index["runs"], "the index lists the runs"
    for name, run in index["runs"].items():
        assert run["result"] == f"softtissue.puncture_run.{name}.result"
    result = results.puncture_result("cone-clot-h0.125")
    assert isinstance(result, RunResult)
    force = np.asarray(result["softtissue.tip_force:clot"].force.values.to("N"))
    assert len(force) == index["runs"]["cone-clot-h0.125"]["steps"]
    assert float(force.max()) == index["runs"]["cone-clot-h0.125"]["peak_N"]
