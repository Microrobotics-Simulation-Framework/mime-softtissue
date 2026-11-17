"""Puncture and retrieval through MIME: a rigid barb-like tip (a MIME ``RigidInertial`` body whose one part is the
plugin's ``softtissue.headed_tip`` kind) moved by the plugin's servo at 10 mm/min [placeholder] into the axisymmetric
gel block (``softtissue.gel_block``) to a set depth and back, joined by ``softtissue.tip_contact``; the force read by
``softtissue.tip_force``. Every number is a placeholder."""
from __future__ import annotations

from mime_softtissue.dynamics import GelBlock
from mime_softtissue.interactions import Servo, TipContact
from mime_softtissue.observations import TipForce
from mime_softtissue.scene import HeadedTip
from mime.dynamics import RigidInertial
from mime.experiment import Experiment
from mime.scene import Scene, Z
from mime.things import Body, Part, State
from mime.units import kg, kg_m2, m_per_s, m_per_s2, mm, rad_per_s, s

SPEED = 10e-3 / 60.0  # m/s, 10 mm/min [placeholder: the hollow-needle coring tests' speed]
STEP = 0.02e-3  # m a base step
DT = STEP / SPEED
OBS = "tip_force"


def tip_experiment(*, head_radius_mm: float = 0.6, friction: float = 0.3, set_depth_mm: float = 0.6,
                   end_depth_mm: float = -0.1, h_mm: float = 0.05,
                   name: str = "tip_retrieval") -> tuple[Experiment, int]:
    """The experiment and its number of base steps (insertion to the set depth, withdrawal to the end depth)."""
    exp = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=s(DT))
    exp.add_body(Body("clot", dynamics=GelBlock(radius=mm(5.0), height=mm(5.0),
                                                   material="softtissue_gel_placeholder", h=mm(h_mm)),
                      initial=State(position=mm((0.0, 0.0, 0.0)))))
    tip = HeadedTip(neck_radius=mm(0.4), head_radius=mm(head_radius_mm), head_length=mm(0.2), fillet=mm(0.05),
                       length=mm(3.0))
    exp.add_body(Body("tip", dynamics=RigidInertial(), parts={"tip": Part(tip)},
                      initial=State(position=mm((0.0, 0.0, 5.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                                    velocity=m_per_s((0.0, 0.0, -SPEED)),
                                    angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    exp.declare("tip.mass", kg(1.2e-5), reason="a placeholder: a 3 mm steel tip of 0.4 mm radius")
    exp.declare("tip.inertia", kg_m2((1e-11, 1e-11, 1e-12)), reason="a placeholder: the same tip's order")
    insert = int(round(set_depth_mm * 1e-3 / STEP))
    exp.attach(Servo(speed=m_per_s(SPEED), insert_steps=insert), body="tip")
    exp.attach(TipContact(friction=friction, adhesion=0.0), tip="tip", solid="clot")
    exp.observe(TipForce(solid="clot"))
    steps = insert + int(round((set_depth_mm - end_depth_mm) * 1e-3 / STEP))
    return exp, steps
