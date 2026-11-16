"""A rigid tip pushed into a soft block (gel or clot) through MIME, axisymmetric or in 3-D: the experiment the
notebook ``notebooks/tutorial_puncture_3d.ipynb`` builds cell by cell, as one function, and the runs behind its
converged curves.

``puncture_experiment(tip, material, ...)`` composes the plugin's entries and nothing else: the block
(``softtissue.gel_block``, its ``dimension`` axisymmetric or 3d, its material a record), the tip (a MIME rigid body whose
one part is ``softtissue.punch_tip``, ``softtissue.cone_tip`` or ``softtissue.headed_tip``), the servo driving it down
at a steady speed (``softtissue.servo``), the contact law joining them (``softtissue.tip_contact``, the penalty 100 E/h)
and two observations (``softtissue.tip_force``, ``softtissue.crack_size``).

    python examples/puncture.py OUT --tip cone --material softtissue.whole_blood_clot --depth 3 --h 0.125
    python examples/puncture.py OUT --tip punch --material softtissue_gel_placeholder --dimension 3d --depth 0.44 --h 0.1"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import mime_softtissue.entries as st
from mime.dynamics import RigidInertial
from mime.experiment import Experiment
from mime.scene import Scene, Z
from mime.things import Body, Part, State
from mime.units import deg, kg, kg_m2, m_per_s, m_per_s2, mm, rad_per_s, s
from mime_softtissue.entries._material_record import read_material

FORCE, CRACK = "softtissue.tip_force:clot", "softtissue.crack_size:clot"
SHAPES = {"punch": "softtissue.punch_tip", "cone": "softtissue.cone_tip", "headed": "softtissue.headed_tip"}


def tips() -> dict[str, Any]:
    """The tips the tutorial uses: S2's blunt punch, a sharp cone, and a sphere-tipped needle (Fakhouri's, R = 1 mm)."""
    return {"punch": st.PunchTip(radius=mm(0.5), fillet=mm(0.05), length=mm(4.0)),
            "cone": st.ConeTip(radius=mm(0.5), tip_radius=mm(0.02), half_angle=deg(15.0), length=mm(6.0)),
            "needle": st.PunchTip(radius=mm(1.0), fillet=mm(1.0), length=mm(6.0))}


def puncture_experiment(tip: Any, material: str, *, dimension: str = "axisymmetric", radius_mm: float = 5.0,
                        height_mm: float = 5.0, h_mm: float = 0.1, fine_mm: float = 1.0, step_mm: float = 0.02,
                        depth_mm: float = 1.0, speed_mm_s: float = 0.1,
                        name: str = "puncture") -> tuple[Experiment, int]:
    """The experiment and its number of base steps: the tip, its lowest point on the block's top face at the start,
    pushed down ``step_mm`` a base step (at ``speed_mm_s``) to ``depth_mm``. The block solves at the tip's pose of the
    step before (the law reads the pose, then the block reads the law), so the steps are one more than the depth's."""
    dt = step_mm / speed_mm_s
    exp = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=s(dt))
    exp.add_body(Body("clot", dynamics=st.GelBlock(radius=mm(radius_mm), height=mm(height_mm), material=material,
                                                   h=mm(h_mm), fine=mm(fine_mm), dimension=dimension,
                                                   tip_shape=SHAPES["cone" if isinstance(tip, st.ConeTip) else
                                                                    "punch" if isinstance(tip, st.PunchTip) else
                                                                    "headed"]),
                      initial=State(position=mm((0.0, 0.0, 0.0)))))
    exp.add_body(Body("tip", dynamics=RigidInertial(), parts={"tip": Part(tip)},
                      initial=State(position=mm((0.0, 0.0, height_mm)), orientation=(1.0, 0.0, 0.0, 0.0),
                                    velocity=m_per_s((0.0, 0.0, -speed_mm_s * 1e-3)),
                                    angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    exp.declare("tip.mass", kg(1.2e-5), reason="a placeholder: a steel tip of a few millimetres")
    exp.declare("tip.inertia", kg_m2((1e-11, 1e-11, 1e-12)), reason="a placeholder: the same tip's order")
    steps = int(round(depth_mm / step_mm)) + 1
    exp.attach(st.Servo(speed=m_per_s(speed_mm_s * 1e-3), insert_steps=10 * steps), body="tip")  # down all run
    young = read_material(material)["young_modulus"]
    exp.attach(st.TipContact(friction=0.0, adhesion=0.0, stiffness=100.0 * young / (h_mm * 1e-3)), tip="tip",
               solid="clot")
    exp.observe(st.TipForce(solid="clot"))
    exp.observe(st.CrackSize(solid="clot"))
    return exp, steps


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
