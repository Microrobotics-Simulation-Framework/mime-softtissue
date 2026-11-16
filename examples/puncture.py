"""A rigid tip pushed into a soft block (gel or clot) through MIME, axisymmetric or in 3-D
(``mime_softtissue.studies.puncture.puncture_experiment``, the experiment the notebook
``notebooks/tutorial_puncture_3d.ipynb`` builds cell by cell): the runs behind its converged curves, each written to a
run folder with ``puncture.json`` (depth, force and crack area at every step). The run keeps the observations' series
(``record=``) and the stage's frames, not every node's fields (a 3-D block's history costs minutes to save).

    python examples/puncture.py OUT --tip cone --material softtissue.whole_blood_clot --depth 3 --h 0.125
    python examples/puncture.py OUT --tip punch --material softtissue_gel_placeholder --dimension 3d --depth 0.44 --h 0.1"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from mime.units import N, m, m2, mm, mm_per_s
from mime_softtissue.studies.puncture import CRACK, FORCE, puncture_experiment, tips


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--tip", choices=sorted(tips()), default="cone")
    ap.add_argument("--material", default="softtissue.whole_blood_clot")
    ap.add_argument("--dimension", default="axisymmetric")
    ap.add_argument("--h", type=float, default=0.1, help="mm")
    ap.add_argument("--fine", type=float, default=1.0, help="mm")
    ap.add_argument("--fine-across", type=float, default=0.8, help="mm (3-D)")
    ap.add_argument("--radius", type=float, default=5.0, help="mm")
    ap.add_argument("--height", type=float, default=5.0, help="mm")
    ap.add_argument("--step", type=float, default=0.02, help="mm")
    ap.add_argument("--depth", type=float, default=1.0, help="mm")
    ap.add_argument("--speed", type=float, default=0.1, help="mm/s")
    ap.add_argument("--every", type=int, default=1, help="a stage frame every this many steps")
    ap.add_argument("--solver", default="tip3d", help="3-D: tip3d or all2")
    ap.add_argument("--symmetry", default="quarter", help="3-D: quarter, half or full")
    ap.add_argument("--offset", type=float, default=0.0, help="mm: the tip off the axis in x (3-D half or full)")
    ap.add_argument("--wall", type=float, default=0.0, help="minutes: stop at a frame after this long and write the "
                                                               "steps reached (0: no limit)")
    a = ap.parse_args()
    p = puncture_experiment(material=a.material, tip=tips()[a.tip], speed=mm_per_s(a.speed), depth=mm(a.depth),
                            dimension=a.dimension, resolution=mm(a.h), step=mm(a.step), block_radius=mm(a.radius),
                            block_height=mm(a.height), fine=mm(a.fine), frames_every=a.every, solver=a.solver,
                            symmetry=a.symmetry, offset=mm(a.offset), fine_across=mm(a.fine_across))
    t0 = time.perf_counter()

    def stop(_progress: object) -> str | None:
        return "wall time" if a.wall and time.perf_counter() - t0 > 60.0 * a.wall else None

    written = p.experiment.build().run_to(a.out, p.duration, record=[f"{FORCE}.force", f"{FORCE}.depth",
                                                                     f"{CRACK}.area"], stage_every=p.stage_every,
                                          stop=stop)
    wall = time.perf_counter() - t0
    run = written.result
    force = [float(v) for v in run[FORCE].force.values.to(N)]
    depth = [float(v) * 1e3 for v in run[FORCE].depth.values.to(m)]
    area = [float(v) * 1e6 for v in run[CRACK].area.values.to(m2)]
    peak = max(range(len(force)), key=lambda k: force[k])
    summary = dict(tip=a.tip, material=a.material, dimension=a.dimension, solver=a.solver, symmetry=a.symmetry,
                   offset_mm=a.offset, h_mm=a.h, fine_mm=a.fine, fine_across_mm=a.fine_across, step_mm=a.step, speed_mm_s=a.speed, steps=p.steps,
                   wall_s=wall, depth_mm=depth, force_N=force, crack_area_mm2=area,
                   peak_N=force[peak], peak_depth_mm=depth[peak], stopped=written.stopped is not None)
    Path(a.out).joinpath("puncture.json").write_text(json.dumps(summary, indent=1))
    print(f"peak {force[peak] * 1e3:.3f} mN at {depth[peak]:.3f} mm; {p.steps} steps in {wall:.0f} s "
          f"(build and compile included)", flush=True)
    if not all(math.isfinite(f) for f in force):
        raise SystemExit("a force is not finite")


if __name__ == "__main__":
    main()
