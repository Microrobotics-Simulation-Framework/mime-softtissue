"""Detection rates of the axisymmetric build's planted errors against each check (``tests/engine/test_axisymmetric``):
the uniform-stretch closed form (deterministic, once) and the three Hypothesis properties (random, ``--reps`` runs).
``python scripts/axi_plant_rates.py --out FILE [--reps 10]``."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import engine.test_axisymmetric as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--only", default="", help="comma-separated check names (default: all)")
a = ap.parse_args()


def caught(fn, plant):
    with plant():
        try:
            fn()
        except Exception as e:  # a failed check, or a defect that stops the code: both a catch
            return True, type(e).__name__
    return False, ""


checks = {
    "uniform stretch 1.3": (lambda: T.test_uniform_stretch_equals_the_closed_form(1.3), 1),
    "uniform stretch 0.7": (lambda: T.test_uniform_stretch_equals_the_closed_form(0.7), 1),
    "confined compression 1.3": (lambda: T.test_confined_compression_equals_the_closed_form(1.3), 1),
    "confined compression 0.7": (lambda: T.test_confined_compression_equals_the_closed_form(0.7), 1),
    "homogeneous energy": (T.check_homogeneous_energy, a.reps),
    "axial translation": (T.check_axial_translation_is_free, a.reps),
    "hessian symmetric": (T.check_hessian_symmetric, a.reps),
}
if a.only:
    checks = {k: v for k, v in checks.items() if k in a.only.split(",")}
res = {"honest": {}, "plants": {}}
t0 = time.time()
for name, (fn, reps) in checks.items():
    ok = []
    for _ in range(reps):
        try:
            fn()
            ok.append(True)
        except Exception:
            ok.append(False)
    res["honest"][name] = f"passed {sum(ok)} of {reps}"
    for pname, plant in T.AXI_PLANTS.items():
        hits = [caught(fn, plant)[0] for _ in range(reps)]
        res["plants"][f"{pname} | {name}"] = f"caught {sum(hits)} of {reps}"
    print(name, res["honest"][name], {k: v for k, v in res["plants"].items() if k.endswith(name)}, flush=True)
res["wall_s"] = time.time() - t0
Path(a.out).write_text(json.dumps(res, indent=1))
