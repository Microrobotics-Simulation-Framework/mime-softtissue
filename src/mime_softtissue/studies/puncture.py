"""The puncture experiment as one function (``puncture_experiment``): the tutorial's notebook builds it cell by cell, and
``examples/puncture.py`` and ``examples/needle_sweep.py`` run it for the converged curves. It composes the plugin's
entries and nothing else: the block (``softtissue.gel_block``, its ``dimension`` axisymmetric or 3d, its material a
record), the tip (a MIME rigid body whose one part is ``softtissue.punch_tip``, ``softtissue.cone_tip`` or
``softtissue.headed_tip``), the servo driving it down at a steady speed (``softtissue.servo``), the contact law joining
them (``softtissue.tip_contact``, the penalty 100 E/h) and two observations (``softtissue.tip_force``,
``softtissue.crack_size``)."""
from __future__ import annotations

from typing import Any

from mime.dynamics import RigidInertial
from mime.experiment import Experiment
from mime.scene import Scene, Z
from mime.things import Body, Part, State
from mime.units import deg, kg, kg_m2, m_per_s, m_per_s2, mm, rad_per_s, s

from .. import entries as st
from ..entries._material_record import read_material

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
