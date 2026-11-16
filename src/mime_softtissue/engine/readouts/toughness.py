"""The toughness a specimen's response implies, from its force and stored energy alone (the same instrument for any
fracture model, so a model's input toughness can be checked against what an experimenter would infer from it).

Exact form (nonlinear elasticity, Rice's J as the energy release rate of a body with a sharp notch):

    G(Delta*, a) = -(1/t) dU(Delta*, a)/da  at fixed grip displacement Delta*

with ``U`` the stored energy of the undamaged specimen (an ``"elastic"`` run) and ``Delta*`` the grip displacement at
the fracturing run's peak force; ``dU/da`` by a central difference over two elastic runs at a -+ da.

Linear form (Tutwiler et al. 2020, Eq. 1): ``G = -(1/2t) Delta*^2 d(F/Delta)/da`` with ``F/Delta`` the specimen's
initial stiffness."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline


def energy_at(run: dict[str, Any], delta: float) -> float:
    """The stored energy of an elastic run at ``delta`` (a cubic spline through the steps, from the origin)."""
    d = np.concatenate([[0.0], np.asarray(run["delta"])])
    e = np.concatenate([[0.0], np.asarray(run["energy"])])
    if delta > d[-1] * (1 + 1e-12):
        raise ValueError(f"the elastic run stops at {d[-1]}, before {delta}")
    return float(CubicSpline(d, e)(delta))


def initial_stiffness(run: dict[str, Any]) -> float:
    return float(np.asarray(run["force"])[0] / np.asarray(run["delta"])[0])


def implied(elastic_minus: dict[str, Any], elastic_plus: dict[str, Any], a_minus: float, a_plus: float,
            delta_star: float, thickness: float) -> dict[str, float]:
    du = energy_at(elastic_plus, delta_star) - energy_at(elastic_minus, delta_star)
    exact = -du / (a_plus - a_minus) / thickness
    dk = initial_stiffness(elastic_plus) - initial_stiffness(elastic_minus)
    linear = -0.5 / thickness * delta_star ** 2 * dk / (a_plus - a_minus)
    return {"exact": exact, "linear": linear}


def peak(run: dict[str, Any]) -> tuple[int, float, float]:
    """The step, force and grip displacement of the largest force."""
    k = int(np.argmax(run["force"]))
    return k, float(run["force"][k]), float(run["delta"][k])
