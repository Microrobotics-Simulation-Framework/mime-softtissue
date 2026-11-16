"""The reference cases of the prototype (h = 0.3 mm, coarse), written against the refactored engine. Each function
returns ``(arrays, scalars)`` in the layout of ``tests/reference/R<n>.npz`` and ``.json``, which were produced by the
prototype's untouched code (``tests/reference/README.md``)."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from mime_softtissue.engine.fracture.phase_field import discretisation_toughness
from mime_softtissue.engine.loadpath import forces, run
from mime_softtissue.engine.material import Material, shear_and_bulk
from mime_softtissue.engine.readouts.toughness import implied
from mime_softtissue.engine.solver import Solver
from mime_softtissue.engine.specimen import blade_block, edge_notched_strip

# the strip benchmark's operating point [read: Tutwiler 2020, 2021; the grip spacing assumed]
WIDTH, HALF_HEIGHT, THICKNESS = 30e-3, 10e-3, 6.5e-3
E, NU, GC, L, STRETCH_LIMIT, K_RES = 5.7e3, 0.3, 7.6, 0.3e-3, 2.8, 1e-6


def strip_material(h: float, mode: str = "phase_field", gc_factor: float = 1.0, l: float = L) -> Material:
    mu, kappa = shear_and_bulk(E, NU)
    gc = GC * gc_factor
    if mode == "phase_field":
        gc = discretisation_toughness(gc, h, l)
    return Material(mu=jnp.asarray(mu), kappa=jnp.asarray(kappa), gc=jnp.asarray(gc), l=jnp.asarray(l),
                    k_res=jnp.asarray(K_RES), stretch_limit=jnp.asarray(STRETCH_LIMIT))


def strip(notch: float, h: float, fine: float = 1.8e-3):
    return edge_notched_strip(width=WIDTH, half_height=HALF_HEIGHT, thickness=THICKNESS, notch=notch, h=h, fine=fine)


def peak_parabola(delta, force) -> tuple[float, float, int]:
    """The parabola's vertex through the three steps about the largest force (scripts/analysis.py of the prototype)."""
    f, d = np.asarray(force), np.asarray(delta)
    k = int(np.argmax(f))
    if 0 < k < len(f) - 1:
        A = np.polyfit(d[k - 1:k + 2], f[k - 1:k + 2], 2)
        dv = -A[1] / (2 * A[0])
        if d[k - 1] <= dv <= d[k + 1]:
            return float(np.polyval(A, dv)), float(dv), k
    return float(f[k]), float(d[k]), k


def _path(mode: str, notch_mm: float, h_mm: float, dmax_mm: float, steps: int, after: int, tol: float = 1e-5,
          solver_kw=None):
    s = Solver(strip(notch_mm * 1e-3, h_mm * 1e-3), mode, tol_d=tol, **(solver_kw or {}))
    deltas = np.linspace(0, dmax_mm * 1e-3, steps + 1)[1:]
    return run(s, strip_material(h_mm * 1e-3, mode), deltas, stop_below=0.3, after_peak=after)


def elastic(notch_mm: float, h_mm: float = 0.3):
    s = Solver(strip(notch_mm * 1e-3, h_mm * 1e-3), "elastic")
    return run(s, strip_material(h_mm * 1e-3), np.linspace(0, 13e-3, 131)[1:])


def R1(**kw):
    r = _path("phase_field", 10.5, 0.3, 12, 160, 3, **kw)
    F, D, _ = peak_parabola(r["delta"], r["force"])
    k = int(np.argmax(r["force"]))
    arrays = {key: r[key] for key in ("force", "energy", "crack_energy", "delta", "iterations", "newton", "final_u")}
    return arrays, dict(peak_force_parabola=F, delta_star_parabola=D, peak_step=k, peak_force_raw=float(r["force"][k]),
                        steps=len(r["force"]))


def R2(r1_arrays):
    e = {n: elastic(n) for n in (10.2, 10.5, 10.8)}
    _, D, _ = peak_parabola(r1_arrays["delta"], r1_arrays["force"])
    tg = implied(e[10.2], e[10.8], 10.2e-3, 10.8e-3, D, THICKNESS)
    arrays = {f"el{n}_{k}": r[k] for n, r in e.items() for k in ("force", "energy", "delta")}
    return arrays, dict(elastic_force_at_2mm_notch105=float(e[10.5]["force"][19]), delta_at_index_19=float(e[10.5]["delta"][19]),
                        G_exact=tg["exact"], G_linear=tg["linear"], delta_star_used=D)


def R3():
    r = _path("local", 10.5, 0.3, 12, 120, 2, tol=1e-6)
    k = int(np.argmax(r["force"]))
    D = float(r["delta"][k])
    e = {n: elastic(n) for n in (10.2, 10.8)}
    tg = implied(e[10.2], e[10.8], 10.2e-3, 10.8e-3, D, THICKNESS)
    return ({key: r[key] for key in ("force", "energy", "delta", "iterations", "final_u")},
            dict(peak_force=float(r["force"].max()), delta_star=D, peak_step=k, G_exact=tg["exact"], G_linear=tg["linear"]))


def R4():
    from mime.units import s
    from tutwiler_strip import OBS, strip_experiment
    h = strip_experiment().build().run(s(11.0), history=True).history
    f = np.asarray(h[OBS]["force"])
    return ({"force": f, "displacement": np.asarray(h[OBS]["displacement"]),
             "crack_length": np.asarray(h[OBS]["crack_length"])},
            dict(force_last=float(f[-1]), displacement_last=float(np.asarray(h[OBS]["displacement"])[-1]), n=len(f)))


def blade_specimen(**kw):
    return blade_block(half_width=5e-3, height=5e-3, thickness=1e-3, punch=0.5e-3, h=0.05e-3, fine_x=1e-3, fine_y=1e-3,
                       ratio=1.3, cap=16.0, **kw)


def blade_material(E_=50e3, nu=0.4, gc=1.0, h=0.05e-3, l=0.1e-3) -> Material:
    mu, kappa = shear_and_bulk(E_, nu)
    return Material(mu=jnp.asarray(mu), kappa=jnp.asarray(kappa), gc=jnp.asarray(gc / (1.0 + h / (2.0 * l))),
                    l=jnp.asarray(l), k_res=jnp.asarray(1e-6), stretch_limit=jnp.asarray(10.0))


def R5(steps: int = 12, specimen_kw=None):
    s = Solver(blade_specimen(**(specimen_kw or {})), "phase_field", tol_d=1e-5, max_staggered=300, max_newton=30)
    mat = blade_material()
    state, rows = s.initial(), []
    for delta in np.linspace(0, 1.0e-3, 101)[1:1 + steps]:
        state, o = s.step(state, jnp.asarray(delta), mat)
        rows.append(o)
    arr = {k: np.asarray([getattr(r, k) for r in rows]) for k in ("force", "energy", "crack_energy")}
    arr["final_u"], arr["final_d"] = np.asarray(state.u), np.asarray(state.d)
    return arr, dict(force_step0=float(arr["force"][0]), force_step12=float(arr["force"][-1]), n=len(rows))


def R6():
    h, a0 = 0.3e-3, 10.5e-3
    deltas = np.linspace(0, 9e-3, 61)[1:][:52]
    base = Solver(strip(a0, h), "phase_field", tol_d=1e-6)
    mat = strip_material(h)
    F, dF = jax.jvp(lambda x: forces(base, mat, deltas, x), (jnp.asarray(a0),), (jnp.ones(()),))
    F, dF = np.asarray(F), np.asarray(dF)
    k = int(np.argmax(F))
    return {"force": F, "dforce": dF}, dict(peak_step=k, peak_force=float(F[k]), derivative=float(dF[k]),
                                            log_derivative=float(dF[k] * a0 / F[k]))


def run_strip_prefix(steps: int) -> dict:
    """The first ``steps`` load steps of R1 (no stopping rule): what a planted error is looked for in cheaply."""
    s = Solver(strip(10.5e-3, 0.3e-3), "phase_field", tol_d=1e-5)
    deltas = np.linspace(0, 12e-3, 161)[1:][:steps]
    return run(s, strip_material(0.3e-3), deltas)
