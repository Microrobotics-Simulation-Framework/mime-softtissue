"""A rigid tip pushed into a soft block (gel or clot) through MIME, axisymmetric or in 3-D
(``mime_softtissue.studies.puncture.puncture_experiment``, the experiment the notebook
``notebooks/tutorial_puncture_3d.ipynb`` builds cell by cell): the runs behind its converged curves, each written to a
run folder with ``puncture.json`` (depth, force and crack area at every step).

    python examples/puncture.py OUT --tip cone --material softtissue.whole_blood_clot --depth 3 --h 0.125
    python examples/puncture.py OUT --tip punch --material softtissue_gel_placeholder --dimension 3d --depth 0.44 --h 0.1"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from mime.units import s
from mime_softtissue.studies.puncture import CRACK, FORCE, puncture_experiment, tips


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--tip", choices=sorted(tips()), default="cone")
    ap.add_argument("--material", default="softtissue.whole_blood_clot")
    ap.add_argument("--dimension", default="axisymmetric")
    ap.add_argument("--h", type=float, default=0.1, help="mm")
    ap.add_argument("--fine", type=float, default=1.0, help="mm")
    ap.add_argument("--radius", type=float, default=5.0, help="mm")
    ap.add_argument("--height", type=float, default=5.0, help="mm")
    ap.add_argument("--step", type=float, default=0.02, help="mm")
    ap.add_argument("--depth", type=float, default=1.0, help="mm")
    ap.add_argument("--every", type=int, default=1, help="a stage frame every this many steps")
    a = ap.parse_args()
    exp, steps = puncture_experiment(tips()[a.tip], a.material, dimension=a.dimension, radius_mm=a.radius,
                                     height_mm=a.height, h_mm=a.h, fine_mm=a.fine, step_mm=a.step, depth_mm=a.depth)
    t0 = time.perf_counter()
    built = exp.build()
    dt = a.step / 0.1  # the default speed, 0.1 mm/s
    steps = a.every * math.ceil(steps / a.every)  # a run is whole chunks of the stage's frames; the servo keeps pushing
    written = built.run_to(a.out, s(steps * dt), history=True, stage_every=s(a.every * dt))
    wall = time.perf_counter() - t0
    run = written.result
    force = [float(v) for v in run.history[FORCE]["force"]]
    depth = [float(v) for v in run.history[FORCE]["depth"]]
    area = [float(v) for v in run.history[CRACK]["area"]]
    peak = max(range(len(force)), key=lambda k: force[k])
    summary = dict(tip=a.tip, material=a.material, dimension=a.dimension, h_mm=a.h, fine_mm=a.fine, step_mm=a.step,
                   steps=steps, wall_s=wall, depth_mm=[d * 1e3 for d in depth], force_N=force, crack_area_mm2=[
                       x * 1e6 for x in area], peak_N=force[peak], peak_depth_mm=depth[peak] * 1e3)
    Path(a.out).joinpath("puncture.json").write_text(json.dumps(summary, indent=1))
    print(f"peak {force[peak] * 1e3:.3f} mN at {depth[peak] * 1e3:.3f} mm; {steps} steps in {wall:.0f} s "
          f"(build and compile included)", flush=True)
    if not all(math.isfinite(f) for f in force):
        raise SystemExit("a force is not finite")


if __name__ == "__main__":
    main()
