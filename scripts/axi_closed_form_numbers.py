"""The axisymmetric closed-form checks' departures, honest and under each plant (numbers for the report).
``python scripts/axi_closed_form_numbers.py OUT.json``"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import engine.test_axisymmetric as T  # noqa: E402

out = {}
for name, fn in (("uniform", T.uniform_stretch_error), ("confined", T.confined_compression_error)):
    for st in (1.3, 0.7):
        err, info = fn(st)
        out[f"{name} {st} honest"] = {"worst": err, **{k: float(v) for k, v in info.items()}}
        for pname, plant in T.AXI_PLANTS.items():
            with plant():
                err, info = fn(st)
            out[f"{name} {st} {pname}"] = {"worst": float(err), **{k: float(v) for k, v in info.items()}}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(f"{k:40s} {v['worst']:.3e}")
