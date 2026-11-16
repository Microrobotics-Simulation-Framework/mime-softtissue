"""Measure the detection rate of every planted error (tests/plants.py). Usage: measure_plants.py property [reps] |
reproduction. Prints one line per plant: caught n of m."""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "tests"), str(HERE.parent / "tests" / "engine"), str(HERE.parent / "examples")]
import test_plants as tp  # noqa: E402

which = sys.argv[1]
if which == "property":
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    for name, check, plant in tp.PROPERTY_PLANTS:
        t0 = time.time()
        n = sum(tp.caught(check, plant) for _ in range(reps))
        print(f"{name}: caught {n} of {reps} runs ({time.time() - t0:.0f} s)", flush=True)
else:
    bits, worst = tp.detector()
    print(f"honest: equal bits {bits}, worst relative {worst:.2e}", flush=True)
    for name, plant in tp.REPRODUCTION_PLANTS:
        with plant():
            bits, worst = tp.detector()
        print(f"{name}: equal bits {bits}, worst relative departure {worst:.2e}; caught by 1e-9 line: {worst > 1e-9}; "
              f"caught by bits: {not bits}", flush=True)
    arrays, _ = tp.rc.R5(specimen_kw={"reaction_skips_edge": True})
    ref = np.load(tp.REF / "R5.npz")["force"]
    print(f"blade edge node out of the reaction: max relative force departure "
          f"{float(np.max(np.abs(arrays['force'] - ref) / np.abs(ref))):.2e}")
