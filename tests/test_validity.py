"""Every entry says where it is not valid, and no line is empty.

Validity lines are listed in every build report that uses the entry, and on its card. An entry with none claims to be valid
everywhere, which no law is; an empty line claims it and says nothing. Add each of your entries to ENTRIES.
"""

import pytest

import mime_softtissue.entries as st

ENTRIES = [st.NotchedStrip, st.GripForce, st.GelBlock, st.TipContact, st.Servo, st.TipForce, st.HeadedTip]
#: Found by this test at the port [measured]: round 4's tip entries (the contact law, the servo, the tip-force observation and
#: the headed-tip kind) declare NO validity line. Their limits are in the report (one-step lag, axisymmetric only, a coaxial
#: tip), not on the entries. Held here as the exact set, so that writing the lines (the set shrinks: delete the name) or a new
#: entry that forgets them (a name missing from this set fails) is each seen. Not a claim that they are valid everywhere.
WITHOUT_VALIDITY = {"softtissue.tip_contact", "softtissue.servo", "softtissue.tip_force", "softtissue.headed_tip"}
#: Words an entry's lines must say, where it has a claim a reader must not miss.
MUST_SAY = {st.NotchedStrip: ("plane strain", "quasi-static"), st.GripForce: ("peak",)}


def test_the_entries_without_validity_lines_are_exactly_the_known_set():
    assert {entry.name for entry in ENTRIES if not entry.validity} == WITHOUT_VALIDITY


@pytest.mark.parametrize("entry", [e for e in ENTRIES if e.name not in WITHOUT_VALIDITY], ids=lambda entry: entry.name)
def test_every_other_entry_declares_validity_lines_and_none_is_empty(entry):
    assert entry.validity, f"{entry.name} declares no validity line"
    assert all(line.strip() for line in entry.validity), f"{entry.name} declares an empty validity line"


@pytest.mark.parametrize("entry", list(MUST_SAY), ids=lambda entry: entry.name)
def test_the_validity_lines_say_what_a_reader_must_not_miss(entry):
    text = " ".join(entry.validity).lower()
    for word in MUST_SAY[entry]:
        assert word in text
