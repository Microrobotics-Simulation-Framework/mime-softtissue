"""The puncture experiment as one function, the template to adapt: ``puncture_experiment(material=..., tip=...,
speed=..., dimension=..., resolution=...)``. Its arguments are the swap points (a material record, a tip and its
dimensions, the insertion speed, axisymmetric or 3-D, the grid); everything else is held. The tutorial's notebook builds
the same experiment cell by cell; ``examples/puncture.py`` and ``examples/needle_sweep.py`` run it for the converged
curves.

It composes the plugin's entries and nothing else: the block (``softtissue.gel_block``, its ``dimension`` axisymmetric
or 3d, its material a record), the tip (a MIME rigid body whose one part is ``softtissue.punch_tip``,
``softtissue.cone_tip`` or ``softtissue.headed_tip``), the servo pushing it down at a steady speed (``softtissue.servo``),
the contact law joining them (``softtissue.tip_contact``: frictionless, the penalty 100 E/h) and two observations
(``softtissue.tip_force``, ``softtissue.crack_size``)."""
from __future__ import annotations

import math
from typing import Any, NamedTuple

from mime.dynamics import RigidInertial
from mime.experiment import Experiment
from mime.scene import Scene, Z
from mime.things import Body, Part, State
from mime.units import Quantity, deg, kg, kg_m2, m, m_per_s, m_per_s2, mm, mm_per_s, rad_per_s, s

from .._material_record import read_material
from .._quantities import scalar
from ..dynamics import GelBlock
from ..interactions import Servo, TipContact
from ..observations import CrackSize, TipForce
from ..scene import ConeTip, HeadedTip, PunchTip

FORCE, CRACK = "softtissue.tip_force:clot", "softtissue.crack_size:clot"
SHAPES = {PunchTip: "softtissue.punch_tip", ConeTip: "softtissue.cone_tip", HeadedTip: "softtissue.headed_tip"}


class Puncture(NamedTuple):
    """The experiment, the run's ``duration`` (the tip at ``depth``, rounded up to whole frames) and ``stage_every``
    (a frame every ``frames_every`` steps): ``p.experiment.build().run_to(folder, p.duration, stage_every=p.stage_every)``."""

    experiment: Experiment
    duration: Quantity
    stage_every: Quantity
    steps: int


def tips() -> dict[str, Any]:
    """The tutorial's tips: S2's blunt punch (1 mm across), a sharp 20-degree cone whose apex is rounded by 0.05 mm on a
    2 mm needle (TIP3D's cone: its conical part reaches the shaft 2.6 mm deep), and a sphere-tipped needle (Fakhouri's,
    R = 1 mm)."""
    return {"punch": PunchTip(radius=mm(0.5), fillet=mm(0.05), length=mm(4.0)),
            "cone": ConeTip(radius=mm(1.0), tip_radius=mm(0.05), half_angle=deg(20.0), length=mm(6.0)),
            "needle": PunchTip(radius=mm(1.0), fillet=mm(1.0), length=mm(6.0))}


def puncture_experiment(*, material: str, tip: Any, speed: Quantity = mm_per_s(0.1), depth: Quantity = mm(3.0),
                        dimension: str = "axisymmetric", resolution: Quantity = mm(0.125),
                        step: Quantity = mm(0.02), block_radius: Quantity = mm(5.0), block_height: Quantity = mm(5.0),
                        fine: Quantity = mm(1.0), frames_every: int = 1, solver: str = "tip3d",
                        symmetry: str = "quarter", offset: Quantity = mm(0.0), fine_across: Quantity = mm(0.8),
                        name: str = "puncture") -> Puncture:
    """A rigid ``tip`` pushed at ``speed`` into a block of the record ``material`` to ``depth``.

    - ``material``: a record's name (``"softtissue.whole_blood_clot"``, or your own fitted record);
    - ``tip``: a tip kind with its dimensions (``ConeTip(...)``, ``PunchTip(...)``, ``HeadedTip(...)``);
    - ``speed``: the insertion speed (the model is quasi-static: it sets the clock only);
    - ``dimension``: ``"axisymmetric"`` (fast; ring cracks only) or ``"3d"``, solved on its ``symmetry`` (``"quarter"``,
      ``"half"`` or ``"full"``; a flat crack through the axis needs a half or the whole block), the tip ``offset`` from
      the axis in x (an imperfection; a half wants about 0.2 ``resolution``), its fine cells within ``fine_across`` of
      the axis;
    - ``resolution``: the cell size near the tip (at most half the record's length scale), within ``fine`` of the axis
      and the top; ``step`` the depth a base step.

    The block is a cylinder (axisymmetric) or a square (3-D), ``block_radius`` from the axis and ``block_height`` tall,
    its bottom held; the tip starts with its lowest point on the top face."""
    if type(tip) not in SHAPES:
        raise TypeError(f"the tip is one of {[k.__name__ for k in SHAPES]}, not {type(tip).__name__}")
    step_m, speed_m_s = scalar(step, m), scalar(speed, m_per_s)
    dt = step_m / speed_m_s
    h_m = scalar(resolution, m)
    exp = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=s(dt))
    exp.add_body(Body("clot", dynamics=GelBlock(radius=block_radius, height=block_height, material=material,
                                                   h=resolution, fine=fine, dimension=dimension,
                                                   tip_shape=SHAPES[type(tip)], solver=solver, symmetry=symmetry,
                                                   fine_across=fine_across),
                      initial=State(position=mm((0.0, 0.0, 0.0)))))
    exp.add_body(Body("tip", dynamics=RigidInertial(), parts={"tip": Part(tip)},
                      initial=State(position=mm((scalar(offset, mm), 0.0, scalar(block_height, mm))),
                                    orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, -speed_m_s)),
                                    angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    exp.declare("tip.mass", kg(1.2e-5), reason="a steel tip of a few millimetres (quasi-static: its mass does not matter)")
    exp.declare("tip.inertia", kg_m2((1e-11, 1e-11, 1e-12)), reason="the same tip's order")
    # one step more than the depth needs (the block solves at the tip's pose of the step before), in whole frames
    steps = frames_every * math.ceil((round(scalar(depth, mm) / scalar(step, mm)) + 1) / frames_every)
    exp.attach(Servo(speed=m_per_s(speed_m_s), insert_steps=steps), body="tip")
    young = read_material(material)["young_modulus"]
    exp.attach(TipContact(friction=0.0, adhesion=0.0, stiffness=100.0 * young / h_m), tip="tip", solid="clot")
    exp.observe(TipForce(solid="clot"))
    exp.observe(CrackSize(solid="clot"))
    return Puncture(exp, s(steps * dt), s(frames_every * dt), steps)
