"""Writes the tutorial's base experiment as a saved MIME experiment, the folder a user's own starts from with
``Experiment.based_on``: a 20-degree cone (its apex rounded by 0.05 mm, on a 2 mm needle) pushed 3 mm into the
calibrated whole-blood clot, axisymmetric, cells of 0.125 mm, built step by step as the notebook ``notebooks/tutorial_puncture_3d.ipynb`` builds it.

    python examples/saved/build_clot_puncture.py   # a minute or two: the saved form runs the experiment's first steps

The folder is written inside the package (``mime_softtissue/studies/saved/clot_puncture``) so that an installed plugin
carries it (``mime_softtissue.studies.results.saved("clot_puncture")``); its published record will take its place."""
import json
import shutil
from pathlib import Path

import numpy as np

import mime_softtissue.entries as st
from mime import Body, Experiment, State
from mime.assets import reference, resolve
from mime.dynamics import RigidInertial
from mime.scene import Scene, Z
from mime.things import Part
from mime.units import deg, kg, kg_m2, m_per_s, m_per_s2, mm, rad_per_s, s

FOLDER = Path(__file__).resolve().parents[2] / "src/mime_softtissue/studies/saved/clot_puncture"

cone_half_angle, needle_radius, apex_radius, push_to, grid = 20.0, 1.0, 0.05, 3.0, 0.125
experiment = Experiment("clot_puncture", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=s(0.2))
speed = 0.1e-3
clot_record = "softtissue.whole_blood_clot"
experiment.add_body(Body(
    "clot",
    dynamics=st.GelBlock(radius=mm(5.0), height=mm(5.0), material=clot_record, h=mm(grid), fine=mm(1.0),
                         dimension="axisymmetric", tip_shape="softtissue.cone_tip"),
    initial=State(position=mm((0.0, 0.0, 0.0))),
))
needle = st.ConeTip(radius=mm(needle_radius), tip_radius=mm(apex_radius), half_angle=deg(cone_half_angle),
                    length=mm(6.0))
experiment.add_body(Body(
    "tip",
    dynamics=RigidInertial(),
    parts={"tip": Part(needle)},
    initial=State(position=mm((0.0, 0.0, 5.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                  velocity=m_per_s((0.0, 0.0, -speed)), angular_velocity=rad_per_s((0.0, 0.0, 0.0))),
))
experiment.declare("tip.mass", kg(1.2e-5), reason="a steel needle of a few millimetres (its mass does not matter here)")
experiment.declare("tip.inertia", kg_m2((1e-11, 1e-11, 1e-12)), reason="the same needle's order")
steps = 6 * int(np.ceil((push_to / 0.02 + 1) / 6))
experiment.attach(st.Servo(speed=m_per_s(speed), insert_steps=steps), body="tip")
with open(resolve(reference(clot_record))) as f:
    young = json.load(f)["values"]["young_modulus"]  # the record's (Pa): the contact's penalty is 100 E / h
experiment.attach(st.TipContact(friction=0.0, adhesion=0.0, stiffness=100.0 * young / (grid * 1e-3)),
                  tip="tip", solid="clot")
experiment.observe(st.TipForce(solid="clot"))
experiment.observe(st.CrackSize(solid="clot"))

if __name__ == "__main__":
    if FOLDER.exists():
        shutil.rmtree(FOLDER)
    FOLDER.parent.mkdir(parents=True, exist_ok=True)
    experiment.licence("CC-BY-4.0")
    experiment.save(FOLDER)
    print("saved", FOLDER, sorted(p.name for p in FOLDER.iterdir()),
          sum(p.stat().st_size for p in FOLDER.rglob("*") if p.is_file()), "bytes")
