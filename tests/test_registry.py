"""The registry walk and the saved round trip: what MIME asks of every entry, asked of this plugin's.

MIME lists what is installed by importing every package that declares a `mime.entries` entry point, in a fresh process
(`installed_entries`), and a saved experiment is loaded the same way. So these fail if the entry point is wrong, if an
entry does not register on import, or if an entry's settings do not survive a save: the three things a user hits first.
"""

import json
from pathlib import Path

from mime.assets import reference
from mime.experiment import Experiment
from mime.testing import installed_entries, round_trip_failures

RECORDS = Path(__file__).resolve().parent.parent / "src" / "mime_softtissue" / "records"
ENTRIES = {("dynamics", "softtissue.notched_strip"), ("observation", "softtissue.grip_force"),
           ("dynamics", "softtissue.gel_block"), ("interaction", "softtissue.tip_contact"),
           ("interaction", "softtissue.servo"), ("observation", "softtissue.tip_force"), ("geometry", "softtissue.headed_tip"),
           ("geometry", "softtissue.punch_tip"), ("geometry", "softtissue.cone_tip"),
           ("observation", "softtissue.crack_size"),
           ("dynamics", "softtissue.replayed_tear")}


def test_the_registry_lists_the_entries():
    listed = {(kind, name) for kind, name, _ in installed_entries()}
    assert ENTRIES <= listed


def test_the_saved_round_trip(experiment: Experiment, tmp_path: Path):
    assert round_trip_failures(experiment, tmp_path / "strip") == []


def test_every_record_is_listed():
    """Each record is found through the `mime.assets` entry point; the materials are exactly the known set (the rest are
    the shipped results, `RESULTS`). That each shipped record's bytes are the ones it names, and that every shipped data
    file is a record, is tests/test_records.py's, with its controls."""
    listed = json.loads((RECORDS / "records.json").read_text())["assets"]
    materials = {a["name"] for a in listed if not a["name"].startswith(RESULTS)}
    assert materials == {"softtissue_fibrin_plasma", "softtissue_gel_placeholder", "softtissue_clot_lab_placeholder",
                         "softtissue.whole_blood_clot", "softtissue.puncture_gel", "softtissue.puncture_gel_gent"}
    for entry in listed:
        assert reference(entry["name"]).name == entry["name"]


#: The shipped results' record names begin with one of these: the MIME runs, their index and the display's run, the
#: 3-D engine's runs, and the pure-shear and viscoelastic studies' results.
RESULTS = ("softtissue.puncture_run.", "softtissue.puncture_runs", "softtissue.puncture3d.", "softtissue.pure_shear_",
           "softtissue.visco_")


def test_the_entry_declares_the_record_it_reads(experiment: Experiment, tmp_path: Path):
    """P1 (an entry declares the assets it reads): the lock lists the record, with no observation naming it."""
    experiment.build().save(tmp_path / "saved")
    lock = json.loads((tmp_path / "saved" / "mime.lock").read_text())
    assert [asset["name"] for asset in lock["assets"]] == ["softtissue_fibrin_plasma"]
