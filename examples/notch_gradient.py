"""The derivative of the strip's peak grip force with respect to its initial notch length, through MIME's Study.gradient
(forward mode), at a coarse grid so that it runs in about a minute: ``python examples/notch_gradient.py``.

The notch is the body's initial state, so its axis is ``strip.initial.notch``. A central difference of two MIME runs is
printed beside it: an independent check that is not the model's claim of anything."""
import numpy as np

from mime.study import Study, along
from mime.units import mm, s
from tutwiler_strip import OBS, strip_experiment

H_MM, NOTCH_MM, DT, DURATION, EPS = 0.75, 10.5, 3.0, 60.0, 1e-3  # 0.15 mm a step at 3 mm/min, 20 steps


def peak(notch_mm: float) -> float:
    run = strip_experiment(notch_mm=notch_mm, h_mm=H_MM, dt=DT).build().run(s(DURATION), history=True)
    return float(np.asarray(run.final[OBS]["peak_force"]))


if __name__ == "__main__":
    study = Study(strip_experiment(notch_mm=NOTCH_MM, h_mm=H_MM, dt=DT)).sweep(
        along({"strip.initial.notch": [mm(NOTCH_MM)]}), duration=s(DURATION))
    member = study.gradient(lambda final: final[OBS]["peak_force"], wrt=["strip.initial.notch"], mode="forward").member(0)
    g = float(np.asarray(member.gradient["strip.initial.notch"]).reshape(-1)[0])
    central = (peak(NOTCH_MM * (1 + EPS)) - peak(NOTCH_MM * (1 - EPS))) / (2 * EPS * NOTCH_MM * 1e-3)
    print(f"d(peak force)/d(notch) = {g:.5f} N/m forward; {central:.5f} N/m central difference; "
          f"relative difference {abs(g - central) / abs(central):.2e}")
