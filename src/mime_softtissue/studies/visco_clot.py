"""Prony terms for whole-blood clot from the Rausch lab's relaxation tests, held apart from their validation.

The data (``data/rausch_visco_na0wxy.json``, DOI 10.18738/T8/NA0WXY, CC0 1.0): unnotched sheets 40 mm wide, 10 mm
between the grips, 3 mm thick (the pure-shear geometry of ``studies.pure_shear``), pulled at 0.1 mm/s (0.01 /s
clamp strain). Samples 4..7: ramp to 10, 20, 30, 40 % and hold about 300 s. Sample 1: cycles to 1..7 mm, then a pull
to failure. Sample 2: a pull to failure. The species is not stated; the forces are 3 to 4 times Lohr's human clots'.

The split, committed before any force was read:
- **calibration:** samples 4..7, ramp and hold. The Prony terms (beta_1, tau_1, beta_2, tau_2) shared, one amplitude A
  (the long-term C1, Pa, which also takes in the grip's edge effect) per sample. The load cell's offset is each sample's
  first reading (the mean of its first 0.1 s, before the grip has moved 0.01 mm), measured, not fitted: fitted offsets
  came out at 11 and 24 mN for samples 6 and 7 against readings under 1.1 mN, taking in the law's shape error (a change
  from the committed plan, logged). The Yeoh shape C2 / C1 is declared from the pure-shear replication, not refitted;
- **validation, never fitted:** leave-one-out over 4..7 (each hold predicted from the other three, its amplitude from
  its own ramp only); sample 1's cycles and sample 2's pull, each amplitude from its first millimetre only.

Every sample is replayed through ``fit.viscous.pure_shear_path`` (the engine's own law at one material point of a
pure-shear sheet) along its measured displacement, on a 0.01 s grid. The residual is the force over the sample's peak
force (each sample weighs the same) at the shipped points (10 Hz on the ramp and the first 10 s of the hold, then 1 Hz)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..engine.material import Material, Prony
from ..fit.viscous import pure_shear_path

GAUGE, WIDTH, THICKNESS = 10e-3, 40e-3, 3e-3
GRIP_RATE = 0.1e-3  # m/s
CALIBRATION = ("4", "5", "6", "7")
VALIDATION = ("1", "2")
#: The pure-shear replication's Yeoh constants on Lohr's raw data (round 3): C1, C2 (Pa), fitted at a grip rate of
#: 0.2 mm/s through the phase-field run at l = 0.25 mm. Here only their ratio (the shape) is used.
BASE_C1, BASE_C2 = 363.0, 434.0
DT = 0.01  # s, the replay's step (r_1 = DT / tau_1 ~ 0.003)
_path = jax.jit(pure_shear_path)


def data() -> dict[str, Any]:
    return json.loads(resources.files(__package__).joinpath("data/rausch_visco_na0wxy.json").read_text())


@dataclass(frozen=True)
class Sample:
    name: str
    time: np.ndarray  # s
    displacement: np.ndarray  # mm
    force: np.ndarray  # mN
    hold_start: float | None  # s


def samples() -> dict[str, Sample]:
    d = data()["samples"]
    return {k: Sample(k, np.asarray(v["time_s"]), np.asarray(v["displacement_mm"]), np.asarray(v["force_mN"]),
                      v["hold_start_s"]) for k, v in d.items()}


def unit_material(prony: Prony | None, c2_over_c1: float = BASE_C2 / BASE_C1) -> Material:
    """The law per unit long-term C1 (mu = 2 Pa): the force is then linear in the amplitude A."""
    one = jnp.asarray(1.0)
    return Material(mu=2.0 * one, kappa=0.0 * one, gc=one, l=one, k_res=0.0 * one, stretch_limit=10.0 * one,
                    c2=c2_over_c1 * one, prony=prony)


def replay(s: Sample, prony: Prony | None, c2_over_c1: float = BASE_C2 / BASE_C1, t_end: float | None = None) -> np.ndarray:
    """The nominal stress per unit C1 at the sample's times (up to ``t_end``), along its measured displacement."""
    t_end = float(s.time[-1]) if t_end is None else t_end
    grid = np.arange(0.0, t_end + DT, DT)
    u = np.interp(grid, s.time, s.displacement)
    stretch = 1.0 + u * 1e-3 / GAUGE
    stress = np.asarray(_path(jnp.asarray(stretch), jnp.full(len(grid), DT), unit_material(prony, c2_over_c1)).stress)
    keep = s.time <= t_end + 1e-9
    return np.interp(s.time[keep], grid, stress)


def area() -> float:
    return WIDTH * THICKNESS


def offset(s: Sample) -> float:
    """The load cell's reading before the grip moves (mN): the first 0.1 s mean."""
    return float(s.force[0])


def amplitude(p: np.ndarray, force_mN: np.ndarray, zero: float) -> tuple[float, float]:
    """(A Pa, offset mN) by linear least squares: force = A area p + offset, the offset given."""
    m = p * area() * 1e3
    return float(m @ (force_mN - zero) / (m @ m)), zero


def prony_of(theta: np.ndarray) -> Prony:
    """theta = (ln beta_1, ln tau_1, ln beta_2, ln tau_2)."""
    th = np.asarray(theta)
    return Prony(jnp.asarray(np.exp(th[0::2])), jnp.asarray(np.exp(th[1::2])))


def _shape(theta: np.ndarray) -> float:
    """C2 / C1: the declared base's, or theta's fifth entry (ln C2/C1) in the diagnostic with the shape freed."""
    return float(np.exp(theta[4])) if len(theta) > 4 else BASE_C2 / BASE_C1


def residuals(theta: np.ndarray, group: dict[str, Sample]) -> np.ndarray:
    """Each sample's force residual over its peak, its amplitude and offset solved for (variable projection)."""
    pr = prony_of(theta[:4])
    out = []
    for s in group.values():
        p = replay(s, pr, _shape(theta))
        a, b = amplitude(p, s.force, offset(s))
        out.append((a * area() * 1e3 * p + b - s.force) / s.force.max())
    return np.concatenate(out)


STARTS = ((0.3, 1.0, 0.1, 30.0), (0.3, 3.0, 0.1, 60.0), (0.3, 7.0, 0.1, 150.0), (0.3, 15.0, 0.1, 300.0),
          (0.3, 0.5, 0.1, 100.0))


def profile(group: dict[str, Sample], tau_1: float) -> dict[str, Any]:
    """The fit with tau_1 held at ``tau_1`` (the cost's profile along the fast branch's time)."""
    from scipy.optimize import least_squares

    def r(th):
        return residuals(np.array([th[0], np.log(tau_1), th[1], th[2]]), group)

    best = None
    for b1, b2, t2 in ((0.3, 0.1, 60.0), (1.0, 0.3, 30.0), (0.1, 0.1, 200.0)):
        res = least_squares(r, np.log([b1, b2, t2]), bounds=(np.log([1e-4, 1e-4, 5.0]), np.log([10.0, 10.0, 5000.0])),
                            diff_step=1e-4)
        if best is None or res.cost < best.cost:
            best = res
    assert best is not None
    return {"tau_1": tau_1, "beta": np.exp(best.x[0:2]).tolist(), "tau_2": float(np.exp(best.x[2])),
            "rms_over_peak": float(np.sqrt(np.mean(best.fun ** 2)))}


def fit(group: dict[str, Sample], start: np.ndarray | None = None, shape_free: bool = False) -> dict[str, Any]:
    """The Prony terms (2 terms) fitted to ``group``: scipy's trust-region least squares in log variables from each of
    ``STARTS`` (or ``start``), the best kept; then the covariance of theta from the Jacobian at the optimum (residual
    variance times (J^T J)^-1). ``shape_free``: the diagnostic that also fits C2 / C1 (not the calibration)."""
    from scipy.optimize import least_squares

    starts = [np.log(x) for x in STARTS] if start is None else [np.asarray(start)[:4]]
    lo = np.log([1e-4, 0.05, 1e-4, 5.0])
    hi = np.log([10.0, 30.0, 10.0, 5000.0])
    if shape_free:
        starts = [np.append(x, np.log(BASE_C2 / BASE_C1)) for x in starts]
        lo, hi = np.append(lo, np.log(1e-3)), np.append(hi, np.log(1e2))
    best = None
    for x0 in starts:
        r = least_squares(residuals, x0, args=(group,), bounds=(lo, hi), x_scale=1.0, diff_step=1e-4)
        if best is None or r.cost < best.cost:
            best = r
    res = best
    assert res is not None
    J = res.jac
    dof = max(len(res.fun) - len(res.x) - len(group), 1)
    s2 = float(res.fun @ res.fun) / dof
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((len(res.x), len(res.x)), np.nan)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    pr = prony_of(res.x[:4])
    amps = {k: amplitude(replay(s, pr, _shape(res.x)), s.force, offset(s)) for k, s in group.items()}
    return {"theta": res.x.tolist(), "beta": np.exp(res.x[0:4:2]).tolist(), "tau_s": np.exp(res.x[1:4:2]).tolist(),
            "c2_over_c1": _shape(res.x), "cost": float(res.cost),
            "rms_over_peak": float(np.sqrt(np.mean(res.fun ** 2))), "max_abs_over_peak": float(np.max(np.abs(res.fun))),
            "theta_sd": sd.tolist(), "correlation": corr.tolist(), "amplitude_Pa": {k: a for k, (a, _) in amps.items()},
            "offset_mN": {k: b for k, (_, b) in amps.items()}, "status": int(res.status), "nfev": int(res.nfev),
            "points": int(len(res.fun))}


def predicted_hold(s: Sample, prony: Prony, c2_over_c1: float = BASE_C2 / BASE_C1) -> dict[str, float]:
    """A held-out relaxation sample: its amplitude and offset from its own ramp only, the hold predicted."""
    assert s.hold_start is not None
    p = replay(s, prony, c2_over_c1)
    ramp = s.time <= s.hold_start
    a, b = amplitude(p[ramp], s.force[ramp], offset(s))
    model = a * area() * 1e3 * p + b
    hold = s.time >= s.hold_start
    peak = float(s.force.max())
    measured_ratio = float(s.force[-1] / peak)
    predicted_ratio = float(model[-1] / model.max())
    return {"amplitude_Pa": a, "offset_mN": b, "measured_end_over_peak": measured_ratio,
            "predicted_end_over_peak": predicted_ratio, "end_ratio_error": predicted_ratio - measured_ratio,
            "hold_rms_over_peak": float(np.sqrt(np.mean((model[hold] - s.force[hold]) ** 2)) / peak)}


def first_millimetre(s: Sample, prony: Prony | None,
                     c2_over_c1: float = BASE_C2 / BASE_C1) -> tuple[np.ndarray, float, float]:
    """The replay and the amplitude and offset fitted on the sample's first loading up to 1 mm only."""
    p = replay(s, prony, c2_over_c1)
    first = (s.time <= s.time[np.argmax(s.displacement >= 1.0)])
    a, b = amplitude(p[first], s.force[first], offset(s))
    return a * area() * 1e3 * p + b, a, b


def cycles(s: Sample) -> list[dict[str, Any]]:
    """Sample 1's loading and unloading segments: (start, peak, end) indices of each cycle."""
    u = s.displacement
    v = np.gradient(u, s.time)
    out = []
    i = 0
    n = len(u)
    while i < n - 1:
        while i < n - 1 and v[i] <= 1e-3:
            i += 1
        start = i
        while i < n - 1 and v[i] > -1e-3:
            i += 1
        peak = start + int(np.argmax(u[start:i + 1]))
        while i < n - 1 and v[i] < 1e-3:
            i += 1
        end = i
        if peak > start and end > peak:
            out.append({"start": start, "peak": peak, "end": end})
        elif peak > start:
            out.append({"start": start, "peak": peak, "end": None})
            break
    return out


def hysteresis(u: np.ndarray, f: np.ndarray, c: dict[str, Any]) -> dict[str, float]:
    """Loading work, unloading work and their difference over the loading work (the loop's share)."""
    a, p, e = c["start"], c["peak"], c["end"]
    w_load = float(np.trapezoid(f[a:p + 1], u[a:p + 1]))
    w_unload = float(-np.trapezoid(f[p:e + 1], u[p:e + 1]))
    return {"peak_mm": float(u[p]), "load_mJ": w_load * 1e-3, "unload_mJ": w_unload * 1e-3,
            "ratio": (w_load - w_unload) / w_load}


# --- the demo's material ---------------------------------------------------------------------------------------------
#: The published relaxation times (Sugerman et al. 2024, J. Mech. Behav. Biomed. Mater. 154, 106508), declared: this data
#: set does not pin tau_1 (its cost is flat from 0.3 to 7 s, see the report), so the demo holds the published times and
#: fits the betas on samples 4..7.
PUBLISHED_TAU = (3.0, 61.0)
#: The pure-shear replication's grip rate (Lohr's raw files), at which its Yeoh constants are an apparent response.
BASE_GRIP_RATE = 0.2e-3  # m/s


def rate_consistent_scale(prony: Prony, grip_rate: float = BASE_GRIP_RATE, stretch_max: float = 1.5,
                          c2_over_c1: float = BASE_C2 / BASE_C1) -> tuple[float, float]:
    """The long-term scale s such that the viscoelastic law with long-term s (C1, C2), pulled in pure shear at
    ``grip_rate`` (gauge 10 mm) to ``stretch_max``, best carries the elastic law with (C1, C2) (least squares over 20
    points of stretch 1.025..``stretch_max``, the replication's fit points): (s, the remaining RMS over the largest
    stress). Model against model: it treats the replication's Yeoh constants as the response at that rate."""
    rate = grip_rate / GAUGE
    t_end = (stretch_max - 1.0) / rate
    grid = np.arange(0.0, t_end + DT, DT)
    stretch = 1.0 + rate * grid
    pv = np.asarray(_path(jnp.asarray(stretch), jnp.full(len(grid), DT), unit_material(prony, c2_over_c1)).stress)
    pe = np.asarray(_path(jnp.asarray(stretch), jnp.full(len(grid), DT), unit_material(None, c2_over_c1)).stress)
    at = np.linspace(1.025, stretch_max, 20)
    v, e = np.interp(at, stretch, pv), np.interp(at, stretch, pe)
    s = float(v @ e / (v @ v))
    return s, float(np.sqrt(np.mean((s * v - e) ** 2)) / e.max())


#: The demo's betas: fitted on samples 4..7 with tau held at ``PUBLISHED_TAU`` (RMS 3.5 % of each sample's peak; per
#: sample beta_1 runs 0.51..2.46 from 10 to 40 % strain, beta_2 0.15..0.19). Valid for clamp stretch 1.0..1.4, a grip
#: rate of 0.01 /s and times 0.1..300 s, on this data set's clot (species not stated, 3 to 4 times stiffer than
#: Lohr's human clots); anything else (puncture's stretches, other rates, the human clot) is an extrapolation.
DEMO_BETA = (0.8759, 0.2189)


def demo_prony() -> Prony:
    return Prony(jnp.asarray(DEMO_BETA), jnp.asarray(PUBLISHED_TAU))


def viscous_clot(c1: float = BASE_C1, c2: float = BASE_C2, prony: Prony | None = None) -> dict[str, Any]:
    """The rate-consistent long-term constants for the replication's Yeoh (C1, C2), taken as the response at 0.2 mm/s:
    {"c1", "c2" (Pa, long-term), "scale", "prony"}. The toughness and the length scale are the replication's, unchanged
    (only the stored elastic energy drives the fracture)."""
    pr = demo_prony() if prony is None else prony
    s, _ = rate_consistent_scale(pr, c2_over_c1=c2 / c1)
    return {"c1": s * c1, "c2": s * c2, "scale": s, "prony": pr}
