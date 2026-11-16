"""The refactor reproduces the prototype: every case of ``tests/reference`` (``R1`` to ``R6``, produced by the untouched
prototype code; see ``reference/README.md``) against its line: equal bits, otherwise 1e-9 relative (1e-7 for the derivative).
Minutes of CPU, so ``slow``: the `slow` CI lane runs it; a physics assertion behind a marker the default lane skips is
documentation, so the cheap form of the same comparison (``test_plants.py::detector``) is in the default lane."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import compare_reference as cr  # noqa: E402


@pytest.mark.slow
@pytest.mark.parametrize("case", ["R1", "R2", "R3", "R4", "R5", "R6"])
def test_case_reproduces_the_reference(case):
    out = cr.run_case(case)
    assert out["within"], f"{case}: worst relative departure {out['worst_rel']:.2e} (line {out['line']:g})"
