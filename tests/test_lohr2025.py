"""The raw Lohr 2025 data as the pure-shear study reads them: each file is the published one, the notch lengths taken
from Fig. 5A sit on the right files, and the study's target is built from them."""

import numpy as np
import pytest

from mime_softtissue.studies import lohr2025


def assignment_problems(table: dict[str, tuple[float, float, int]]) -> list[str]:
    """What is wrong with a files-to-Fig.-5A table: a file whose recomputed Griffith A is off its marker by more than the
    common offset allows, or whose group differs from the one the match over all 12 (names ignored) gives it."""
    data = lohr2025.curves()
    dc = lohr2025.critical_displacements()
    raw = {n: lohr2025.griffith_a(dc[n], data) for n in table}
    diff = np.array([raw[n] - table[n][1] for n in table])
    problems = [f"{n}: {d:+.4f} J/m^2 from the common offset" for n, d in zip(table, diff - np.median(diff))
                if abs(d) > 0.002]
    by_raw = sorted(table, key=lambda n: raw[n])
    by_marker = sorted(table.values(), key=lambda v: v[1])
    problems += [f"{n}: group {table[n][2]}, the match gives {v[2]}" for n, v in zip(by_raw, by_marker) if v[2] != table[n][2]]
    return problems


def test_every_file_is_the_published_one():
    data = lohr2025.curves()
    assert sorted(data) == sorted([f"Unnotched_{i}" for i in (1, 2, 3)] + [f"Notched_{i}" for i in range(1, 13)])
    assert sorted(lohr2025.critical_displacements()) == sorted(f"Notched_{i}" for i in range(1, 13))


def test_a_changed_byte_is_caught(monkeypatch: pytest.MonkeyPatch):
    record = lohr2025._record(lohr2025.CURVES)
    f = record["files"]["Notched_4.csv"]
    f["text"] = f["text"].replace('"0.1000","0.0', '"0.1000","1.0', 1)
    monkeypatch.setattr(lohr2025, "_record", lambda path: record)
    with pytest.raises(ValueError, match="Notched_4.csv"):
        lohr2025.curves()


def test_the_rate_is_what_the_files_say():
    """0.2 mm/s in every file (the paper's text says 0.1)."""
    for x, _ in lohr2025.curves().values():
        t = np.arange(len(x))[: len(x) - 1] * 0.1
        mid = slice(10, len(x) - 10)
        assert np.polyfit(t[mid], x[mid], 1)[0] == pytest.approx(0.2, rel=1e-3)


def test_the_notch_lengths_sit_on_their_files():
    assert assignment_problems(lohr2025.FIG5A) == []


def test_a_swapped_notch_is_caught():
    swapped = dict(lohr2025.FIG5A)
    swapped["Notched_1"], swapped["Notched_12"] = lohr2025.FIG5A["Notched_12"], lohr2025.FIG5A["Notched_1"]
    assert assignment_problems(swapped) != []
    within = dict(lohr2025.FIG5A)
    within["Notched_7"], within["Notched_8"] = lohr2025.FIG5A["Notched_8"], lohr2025.FIG5A["Notched_7"]
    assert assignment_problems(within) != []


def test_the_target():
    t = lohr2025.target()
    assert [t["groups"][g]["n"] for g in "12345"] == [3, 2, 1, 3, 3]
    assert [t["groups"][g]["notch_mean_mm"] for g in "12345"] == pytest.approx([3.708, 7.335, 9.709, 11.9, 16.513], abs=1e-3)
    assert t["group_curves"]["4"]["peak_N"] == pytest.approx(0.06976, abs=2e-5)
    assert t["unnotched"]["peak_N"] == pytest.approx(0.17102, abs=2e-5)
