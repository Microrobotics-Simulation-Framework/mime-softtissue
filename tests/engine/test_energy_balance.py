"""The solver's own energy account: the work the load does equals the stored energy plus the crack surface energy,
step by step (W - U - Gamma). A coarse strip (l = 1.2 mm, h = l / 2), run to just past its peak.

The line, stated before the check: |W - U - Gamma| at most 2 % of W at every step up to the peak [the prototype's
tolerance for the blade case]; W is the trapezoid of the force over the grip displacement."""
import numpy as np
import pytest

from mime_softtissue.engine.loadpath import run
from mime_softtissue.engine.solver import Solver
from mime_softtissue.engine.specimen import edge_notched_strip
import reproduction_cases as rc

LINE = 0.02


def balance(r):
    d = np.concatenate([[0.0], r["delta"]])
    f = np.concatenate([[0.0], r["force"]])
    work = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(d))])[1:]
    return work, work - r["energy"] - r["crack_energy"]


@pytest.fixture(scope="module")
def coarse_run():
    sp = edge_notched_strip(width=rc.WIDTH, half_height=rc.HALF_HEIGHT, thickness=rc.THICKNESS, notch=10.8e-3, h=0.6e-3,
                            fine=2.4e-3)
    s = Solver(sp, "phase_field", tol_d=1e-5)
    mat = rc.strip_material(0.6e-3, l=1.2e-3)
    return run(s, mat, np.linspace(0, 12e-3, 81)[1:], stop_below=0.5, after_peak=2)


def test_balance_holds_to_the_line_up_to_the_peak(coarse_run):
    work, resid = balance(coarse_run)
    peak = int(np.argmax(coarse_run["force"]))
    assert peak > 10
    assert np.max(np.abs(resid[:peak + 1]) / work[:peak + 1]) <= LINE


def test_balance_resolves_a_planted_error(coarse_run):
    """A planted error of 3 % in the crack energy's book (above the line) must be seen: the control can fail."""
    r = dict(coarse_run)
    r["energy"] = coarse_run["energy"] * 1.03
    work, resid = balance(r)
    peak = int(np.argmax(r["force"]))
    assert np.max(np.abs(resid[:peak + 1]) / work[:peak + 1]) > LINE
