"""Fakhouri's puncture experiment through MIME's Study: sphere-tipped needles of several radii R pushed into the soft gel
of Fakhouri, Hutchens and Crosby (2015) (the record ``softtissue.puncture_gel``), the peak force Fc of each read and put
on Fregonese and Bacca's master curve, Fc / (mu R^2) against mu R / Gc (their Fig. 5, digitised in
``studies/data/fregonese2021_fig5.json``).

One experiment (``puncture.puncture_experiment``), one study: the needle's radius is the axis (its ``radius`` and its
``fillet``, equal for a spherical tip, moved together with ``along``). Everything else is held: the gel, the block, the
grid, the step. The study runs into ``OUT``, a MIME study folder (``mime.results.StudyResult.read(OUT)``: each member's
depth and force with their units, the base experiment saved, resumable if stopped).
``python examples/needle_sweep.py OUT --dimension axisymmetric --h 0.1 --depth 8``"""
from __future__ import annotations

import argparse
import time

import numpy as np

from mime_softtissue.scene import PunchTip
from mime.study import Study, along
from mime.units import mm, mN
from mime_softtissue.studies.puncture import FORCE, puncture_experiment

RADII_MM = (0.5, 1.0, 2.0, 4.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--dimension", default="axisymmetric")
    ap.add_argument("--h", type=float, default=0.1, help="mm")
    ap.add_argument("--fine", type=float, default=5.0, help="mm")
    ap.add_argument("--block", type=float, default=25.0, help="mm: the block's radius and height")
    ap.add_argument("--step", type=float, default=0.05, help="mm")
    ap.add_argument("--depth", type=float, default=8.0, help="mm")
    ap.add_argument("--radii", default=",".join(str(r) for r in RADII_MM), help="mm")
    a = ap.parse_args()
    radii = [float(r) for r in a.radii.split(",")]
    needle = PunchTip(radius=mm(radii[0]), fillet=mm(radii[0]), length=mm(a.block))
    p = puncture_experiment(material="softtissue.puncture_gel", tip=needle, dimension=a.dimension, resolution=mm(a.h),
                            step=mm(a.step), depth=mm(a.depth), block_radius=mm(a.block), block_height=mm(a.block),
                            fine=mm(a.fine), name="needle_sweep")
    study = Study(p.experiment).sweep(along({"tip.parts.tip.radius": [mm(r) for r in radii],
                                             "tip.parts.tip.fillet": [mm(r) for r in radii]}),
                                      duration=p.duration, record=[f"{FORCE}.force", f"{FORCE}.depth"])
    print(study.plan(), flush=True)
    t0 = time.perf_counter()
    result = study.run_to(a.out)
    for k, r in enumerate(radii):
        member = result.member(k)
        force = np.asarray(member[FORCE].force.values.to(mN)).reshape(-1)
        depth = np.asarray(member[FORCE].depth.values.to(mm)).reshape(-1)
        peak = int(np.argmax(force))
        print(f"R {r} mm: peak {force[peak]:.3f} mN at {depth[peak]:.3f} mm", flush=True)
    print(f"{time.perf_counter() - t0:.0f} s; the study folder: {a.out}", flush=True)


if __name__ == "__main__":
    main()
