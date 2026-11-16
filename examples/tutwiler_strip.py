"""The edge-notched fibrin strip of Tutwiler et al. (Sci. Adv. 2020) as a MIME experiment, built from this plugin's
entries: one body (the strip) whose dynamics is ``softtissue.notched_strip``, its notch the body's initial state, pulled
at the paper's 3 mm/min on the base step's clock, with the ``softtissue.grip_force`` observation.

Run it:  python examples/tutwiler_strip.py        (about 1 to 2 minutes on a laptop CPU for the 40-step pull)

Everything here is the model, not a measurement: plane strain, an assumed 20 mm grip spacing (the paper's text read
does not give it). See README.md."""

import mime_softtissue.entries as st
from mime.experiment import Experiment
from mime.scene import Scene, Z
from mime.things import Body, State
from mime.units import m_per_s, m_per_s2, mm, s

RATE = 3e-3 / 60.0  # m/s, 3 mm/min [read: Tutwiler 2020]
OBS = "softtissue.grip_force:strip"


def strip_experiment(*, notch_mm=10.5, h_mm=0.3, material="softtissue_fibrin_plasma", mode="phase_field",
                     name="tutwiler_sen", dt=1.0, thickness_mm=6.5):
    exp = Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=s(dt))
    exp.add_body(Body("strip", dynamics=st.NotchedStrip(
        width=mm(30.0), height=mm(20.0), thickness=mm(thickness_mm), grip_rate=m_per_s(RATE),
        material=material, h=mm(h_mm), mode=mode), initial=State(position=mm((0.0, 0.0, 0.0)), notch=mm(notch_mm))))
    exp.observe(st.GripForce(specimen="strip"))
    return exp


if __name__ == "__main__":
    import numpy as np

    run = strip_experiment().build().run(s(40.0), history=True)
    f = np.asarray(run.history[OBS]["force"])
    print(f"force after {len(f)} steps: {f[-1] * 1e3:.3f} mN; peak so far {np.asarray(run.final[OBS]['peak_force']) * 1e3:.3f} mN")
