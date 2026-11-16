"""Run reference cases with the engine as it is and compare every array and scalar against ``tests/reference``.

    python scripts/compare_reference.py R1 [R2 ...]        # prints a table; exit 1 if any case leaves its line

The line (stated before the refactor, ``PREDICTIONS.md``): equal bits, otherwise a relative 1e-9 on forces and
energies and 1e-7 on the derivative. Each array reports whether it is bit-equal and its largest relative departure."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))
sys.path.insert(0, str(HERE.parent / "examples"))
import reproduction_cases as rc  # noqa: E402

REF = HERE.parent / "tests" / "reference"
LINE = {"R6": 1e-7}


def compare(name: str, arrays: dict, scalars: dict) -> dict:
    ref = np.load(REF / f"{name}.npz")
    meta = json.loads((REF / f"{name}.json").read_text())
    line = LINE.get(name, 1e-9)
    rows, worst, all_bits = [], 0.0, True
    for key in ref.files:
        a, b = np.asarray(arrays[key]), ref[key]
        if a.shape != b.shape:
            rows.append((key, "SHAPE", a.shape, b.shape))
            worst, all_bits = np.inf, False
            continue
        equal = bool(np.array_equal(a, b))
        scale = max(float(np.max(np.abs(b))), 1e-300)
        rel = float(np.max(np.abs(a - b)) / scale)
        rows.append((key, "bits" if equal else f"{rel:.2e}", a.shape, ""))
        worst, all_bits = max(worst, rel), all_bits and equal
    for key, val in scalars.items():
        if key in meta and isinstance(val, float):
            rel = abs(val - meta[key]) / max(abs(meta[key]), 1e-300)
            rows.append((key, "bits" if val == meta[key] else f"{rel:.2e}", "", ""))
            worst, all_bits = max(worst, rel), all_bits and val == meta[key]
    return dict(case=name, bits=all_bits, worst_rel=worst, line=line, within=bool(all_bits or worst <= line), rows=rows)


def run_case(name: str) -> dict:
    if name == "R2":
        r1 = np.load(REF / "R1.npz")
        arrays, scalars = rc.R2({k: r1[k] for k in ("delta", "force")})
    else:
        arrays, scalars = getattr(rc, name)()
    return compare(name, arrays, scalars)


if __name__ == "__main__":
    bad = 0
    for case in sys.argv[1:]:
        out = run_case(case)
        print(f"{case}: {'EQUAL BITS' if out['bits'] else 'worst relative ' + format(out['worst_rel'], '.2e')} "
              f"(line {out['line']:g}) -> {'within' if out['within'] else 'OUTSIDE'}")
        for row in out["rows"]:
            print("   ", *row)
        bad += not out["within"]
    sys.exit(1 if bad else 0)
