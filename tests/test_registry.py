"""The registry walk and the saved round trip: what MIME asks of every entry, asked of this plugin's.

MIME lists what is installed by importing every package that declares a `mime.entries` entry point, in a fresh process
(`installed_entries`), and a saved experiment is loaded the same way. So these fail if the entry point is wrong, if an
entry does not register on import, or if an entry's settings do not survive a save: the three things a user hits first.
"""

import hashlib
import json
from pathlib import Path

from mime.assets import reference
from mime.experiment import Experiment
from mime.testing import installed_entries, round_trip_failures

RECORDS = Path(__file__).resolve().parent.parent / "src" / "mime_softtissue" / "records"
ENTRIES = {("dynamics", "softtissue.notched_strip"), ("observation", "softtissue.grip_force"),
           ("dynamics", "softtissue.gel_block"), ("interaction", "softtissue.tip_contact"),
           ("interaction", "softtissue.servo"), ("observation", "softtissue.tip_force"), ("geometry", "softtissue.headed_tip"),
           ("dynamics", "softtissue.replayed_tear")}


def test_the_registry_lists_the_entries():
    listed = {(kind, name) for kind, name, _ in installed_entries()}
    assert ENTRIES <= listed


def test_the_saved_round_trip(experiment: Experiment, tmp_path: Path):
    assert round_trip_failures(experiment, tmp_path / "strip") == []


def test_every_record_is_listed_and_names_its_bytes():
    """Each record is found through the `mime.assets` entry point; a record that ships its bytes has the sha256 and size of the
    file it names (the lab's placeholder names only a URL, and never ships)."""
    listed = json.loads((RECORDS / "records.json").read_text())["assets"]
    assert {a["name"] for a in listed} == {"softtissue_fibrin_plasma", "softtissue_gel_placeholder",
                                           "softtissue_clot_lab_placeholder", "softtissue.lohr2025_pure_shear",
                                           "softtissue.lohr2025_critical_displacements"}
    for entry in listed:
        record = reference(entry["name"])
        assert record.name == entry["name"]
        shipped = [loc for loc in entry["locations"] if loc.startswith("package:")]
        for loc in shipped:
            data = (RECORDS / "data" / loc.rsplit("/", 1)[1]).read_bytes()
            assert record.sha256 == hashlib.sha256(data).hexdigest() and record.size == len(data), entry["name"]


def test_the_entry_declares_the_record_it_reads(experiment: Experiment, tmp_path: Path):
    """P1 (an entry declares the assets it reads): the lock lists the record, with no observation naming it."""
    experiment.build().save(tmp_path / "saved")
    lock = json.loads((tmp_path / "saved" / "mime.lock").read_text())
    assert [asset["name"] for asset in lock["assets"]] == ["softtissue_fibrin_plasma"]
