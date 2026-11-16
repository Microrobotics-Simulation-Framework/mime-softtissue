"""Notched pure-shear tearing of a thin clot sheet (Lohr et al. 2025, Int. J. Fract., doi 10.1007/s10704-024-00820-4):
a 40 x 10 x 3 mm sheet held along its long edges, an edge notch, pulled across its height.

The build: the upper half by symmetry about the crack plane, incompressible plane stress (``kinematics="plane_stress"``),
the Yeoh law (C1 = mu / 2, C2) inside the AT2 phase field, the notch as the released part of the symmetry plane (a sharp
notch whose length the grid's morph places exactly). Lengths in metres, forces in newtons, as the engine; the data in
``data/`` are in millimetres and newtons.

The calibration, in the order the study runs it:
1. the stiffness (C1, C2) from the unnotched curve, fitted through the engine's own phase-field run of the unnotched
   sheet (the AT2 field softens the bulk before any crack; the fit takes that in, so each toughness has its own fit);
2. the toughness Gc from one notch group's peak force;
3. the other groups' force against displacement and critical displacements are predictions.

``critical_displacement`` reads the crack's start as the displacement at which the damage (>= ``D_CRACK``) has run
``ADVANCE`` beyond the notch tip along the crack plane."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..engine.fracture.phase_field import discretisation_toughness
from ..engine.loadpath import forces, run
from ..engine.material import Material
from ..engine.solver import Solver
from ..engine.specimen import Specimen, edge_notched_strip, morph_columns

WIDTH, HALF_HEIGHT, THICKNESS = 40e-3, 5e-3, 3e-3
K_RES = 1e-6
D_CRACK = 0.9
ADVANCE = 0.5e-3


@dataclass(frozen=True)
class Stiffness:
    """Yeoh's two constants (Pa): W = C1 (I1 - 3) + C2 (I1 - 3)^2."""

    c1: float
    c2: float


def data() -> dict[str, Any]:
    """The digitised data set (see its ``labels`` and ``source``)."""
    return json.loads(resources.files(__package__).joinpath("data/lohr2025_digitised.json").read_text())


def specimen(notch: float, h: float, l: float) -> Specimen:
    """The half sheet on a grid of spacing ``h`` along the crack plane, its fine band three length scales tall (at least
    four cells), the rows above it growing by 1.5 to at most 16 h. ``notch`` 0 is the unnotched sheet; otherwise the tip
    node is the nearest, placed exactly by the morph."""
    fine = max(3 * l, 4 * h)
    tip = round(notch / h) * h
    return edge_notched_strip(width=WIDTH, half_height=HALF_HEIGHT, thickness=THICKNESS, notch=tip, h=h, fine=fine,
                              ratio=1.5, cap=16.0, kinematics="plane_stress")


def material(k: Stiffness, gc: float, l: float, h: float) -> Material:
    """The engine's numbers: ``gc`` is the toughness the grid's phase field dissipates (its input is corrected for the
    grid, ``discretisation_toughness``)."""
    return Material(mu=jnp.asarray(2 * k.c1), kappa=jnp.asarray(0.0), gc=jnp.asarray(discretisation_toughness(gc, h, l)),
                    l=jnp.asarray(l), k_res=jnp.asarray(K_RES), stretch_limit=jnp.asarray(10.0), c2=jnp.asarray(k.c2))


def solver(sp: Specimen, mode: str = "phase_field", tol_d: float = 1e-4, max_staggered: int = 300) -> Solver:
    return Solver(sp, mode, tol_d=tol_d, max_staggered=max_staggered, newton_regularise=True)


def fit_stiffness(delta_mm: np.ndarray, force_n: np.ndarray, gc: float, l: float, h: float, start: Stiffness,
                  h_fit: float = 0.5e-3, iterations: int = 6, mode: str = "phase_field") -> tuple[Stiffness, dict[str, Any]]:
    """(C1, C2) by Gauss-Newton on the unnotched sheet's force at ``delta_mm``, the derivatives by forward mode through
    the load path. ``gc``, ``l`` and ``h`` set the bulk's phase-field softening (as the notched run on grid ``h`` will
    have it); the unnotched sheet itself is solved on the coarser ``h_fit`` (it carries no crack)."""
    sp = specimen(0.0, h_fit, l)
    s = solver(sp, mode)
    deltas = jnp.asarray(np.asarray(delta_mm) * 1e-3)
    y = np.asarray(force_n)
    p = np.array([start.c1, start.c2])

    def model(q):
        return forces(s, material(Stiffness(q[0], q[1]), gc, l, h), deltas)

    history = []
    for _ in range(iterations):
        q = jnp.asarray(p)
        f, J = jax.jvp(model, (q,), (jnp.array([1.0, 0.0]),))
        _, J2 = jax.jvp(model, (q,), (jnp.array([0.0, 1.0]),))
        A = np.stack([np.asarray(J), np.asarray(J2)], axis=1)
        r = y - np.asarray(f)
        step = np.linalg.lstsq(A, r, rcond=None)[0]
        history.append({"c1": float(p[0]), "c2": float(p[1]), "rms_N": float(np.sqrt(np.mean(r ** 2)))})
        p = np.maximum(p + step, [1.0, 0.0])
        if np.max(np.abs(step) / np.maximum(np.abs(p), 1.0)) < 1e-6:
            break
    f = np.asarray(model(jnp.asarray(p)))
    rms = float(np.sqrt(np.mean((y - f) ** 2)))
    return Stiffness(float(p[0]), float(p[1])), {"iterations": history, "rms_N": rms, "nrmse_of_max": rms / float(y.max()),
                                                 "max_abs_rel": float(np.max(np.abs(y - f) / np.maximum(y, 1e-12))),
                                                 "model_N": f.tolist()}


def crack_extension(d: np.ndarray, x_nodes: np.ndarray, notch: float) -> float:
    """How far the damage (>= D_CRACK) runs, unbroken, along the crack plane beyond the notch tip (m)."""
    row = np.asarray(d)[:, 0]
    ahead = np.where(x_nodes > notch + 1e-12)[0]
    k = 0
    while k < len(ahead) and row[ahead[k]] >= D_CRACK:
        k += 1
    return float(x_nodes[ahead[k - 1]] - notch) if k else 0.0


def run_case(notch: float, k: Stiffness, gc: float, l: float, h: float, delta_max_mm: float = 7.0,
             step_mm: float = 0.05, keep_every: int = 2, stop_below: float | None = 0.03,
             verbose: bool = False) -> dict[str, Any]:
    """One sheet's load path: force (N) against grip displacement (mm), the crack's extension at every step, and
    every ``keep_every``-th step's displacement and damage fields (for the animation)."""
    sp = specimen(notch, h, l)
    s = solver(sp)
    mat = material(k, gc, l, h)
    deltas = np.arange(1, int(round(delta_max_mm / step_mm)) + 1) * step_mm * 1e-3
    tr = notch if notch > 0 else None
    r = run(s, mat, deltas, notch=tr, keep_fields=True, keep_u=True, stop_below=stop_below, verbose=verbose)
    x_nodes = np.asarray(morph_columns(sp.grid.x, sp.morph, notch)) if sp.morph is not None else sp.grid.x
    ext = np.array([crack_extension(dd, x_nodes, notch) for dd in r["damage"]]) if notch > 0 else np.zeros(len(r["delta"]))
    keep = np.arange(0, len(r["delta"]), keep_every)
    return {"delta_mm": np.asarray(r["delta"]) * 1e3, "force_N": np.asarray(r["force"]),
            "energy_J": np.asarray(r["energy"]), "crack_energy_J": np.asarray(r["crack_energy"]),
            "staggered": np.asarray(r["iterations"]), "newton": np.asarray(r["newton"]),
            "newton_capped": np.asarray(r["newton_capped"]), "residual": np.asarray(r["residual"]),
            "extension_mm": ext * 1e3, "x_nodes_mm": x_nodes * 1e3, "y_nodes_mm": np.asarray(sp.grid.y) * 1e3,
            "frames_step": keep, "frames_u_mm": r["u_fields"][keep].astype(np.float32) * 1e3,
            "frames_d": r["damage"][keep].astype(np.float32), "notch_mm": notch * 1e3}


def critical_displacement(result: dict[str, Any]) -> float | None:
    """The displacement (mm) at which the crack has run ``ADVANCE`` beyond the notch (linear between steps); None if
    it never does."""
    ext, dl = result["extension_mm"], result["delta_mm"]
    a = ADVANCE * 1e3
    over = np.where(ext >= a)[0]
    if not len(over):
        return None
    i = int(over[0])
    if i == 0:
        return float(dl[0])
    return float(dl[i - 1] + (a - ext[i - 1]) / (ext[i] - ext[i - 1]) * (dl[i] - dl[i - 1]))


def peak(result: dict[str, Any]) -> tuple[float, float]:
    """(peak force N, its displacement mm)."""
    i = int(np.argmax(result["force_N"]))
    return float(result["force_N"][i]), float(result["delta_mm"][i])
