"""The fitting harness's controls (PREDICTIONS_R3.md, item 1). ``python scripts/fit_controls.py PART --out FILE``.

PART: coverage_A, coverage_B, bootstrap_A, misfit, resolution, plants, identifiability. Plants are subclasses here,
never switches in the package."""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from mime_softtissue.fit import core, synthetic as syn

TRUTH_A = {"mu": 1000.0 / 2.6, "kappa": 1000.0 / 1.2}  # E 1 kPa, nu 0.3 [placeholder]
TRUTH_B = {"mu": 200.0, "kappa": 200.0 * 2.6 / 1.2, "k1": 100.0, "k2": 1.0}  # nu 0.3 [placeholder]
DESIGN = ("strain", "frequency", "compression")
NOISE = 0.03
Z95 = 1.959964


class ConeAsPlate(core.Problem):
    """Plant: every rheology set fitted as cone-plate whatever its geometry."""

    def _one(self, ds, mat):
        ds2 = core.Dataset(ds.kind, ds.x, ds.y, ds.name, "cone_plate", ds.gamma0, ds.loss, ds.dropped)
        return super()._one(ds2, mat)


class ScaledDerivative(core.Problem):
    """Plant: one parameter's derivative multiplied by ``scale`` (primal unchanged); ``scale`` 0 is stop_gradient."""

    def __init__(self, *a, name: str, scale: float, **k):
        self.plant = (name, scale)
        super().__init__(*a, **k)

    def params(self, theta):
        p = super().params(theta)
        n, c = self.plant
        v = p[n]
        p[n] = jax.lax.stop_gradient(v) + c * (v - jax.lax.stop_gradient(v))
        return p


def coverage(truth, model, draws, seed, problem_cls=core.Problem, problem_kw=None, cov_plant=None,
             design=DESIGN, starts=None):
    rng = np.random.default_rng(seed)
    clean = syn.clean(design, truth)
    prob = problem_cls(clean, model, **(problem_kw or {}))
    t_true = syn.truth_theta(prob.names, truth)
    hits = np.zeros(len(prob.names), int)
    est, ses, flags_any, fails = [], [], [], 0
    rows = []
    for _ in range(draws):
        data = syn.noisy(clean, NOISE, rng)
        y = np.concatenate([d.y for d in data])
        prob.datasets = data  # the same points (x), so the traced prediction is reused
        f = core.fit(prob, y, starts=starts)
        cov = f.cov if cov_plant is None else cov_plant(prob, f, y)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        if not np.all(np.isfinite(f.theta)):
            fails += 1
            continue
        inside = np.abs(f.theta - t_true) <= Z95 * se
        hits += inside
        est.append(f.theta - t_true)
        ses.append(se)
        fl = core.misfit(prob, f)
        flags_any.append(any(r["flagged"] for r in fl))
        rows.append({"theta_minus_truth": (f.theta - t_true).tolist(), "se": se.tolist(),
                     "flagged": [r["flagged"] for r in fl], "trend_p": [r["trend_p"] for r in fl],
                     "trend_size": [r["trend_size"] for r in fl], "mean": [r["mean"] for r in fl]})
    n = draws - fails
    E, S = np.array(est), np.array(ses)
    return {"names": prob.names, "draws": draws, "fails": fails, "coverage": (hits / n).tolist(),
            "bias_log": E.mean(axis=0).tolist(), "sd_log": E.std(axis=0, ddof=1).tolist(),
            "mean_se": np.mean(S, axis=0).tolist(), "median_se": np.median(S, axis=0).tolist(),
            "false_alarm_or_flag_rate": float(np.mean(flags_any)), "rows": rows}


def unweighted_cov(prob, f, y):
    """Plant: the covariance without the noise level (sigma = 1 assumed)."""
    Jm = np.asarray(prob.jacobian(jnp.asarray(f.theta))) / np.abs(y)[:, None]
    return np.linalg.inv(Jm.T @ Jm)


def misfit_rate(law_truth, energy_law, draws, seed, design=DESIGN):
    rng = np.random.default_rng(seed)
    clean = syn.clean(design, law_truth, energy_law=energy_law)
    prob = core.Problem(clean, "neo_hookean")
    caught, rows = 0, []
    for _ in range(draws):
        data = syn.noisy(clean, NOISE, rng)
        prob.datasets = data
        f = core.fit(prob, np.concatenate([d.y for d in data]))
        fl = core.misfit(prob, f)
        hit = any(r["flagged"] for r in fl)
        caught += hit
        rows.append({"flagged": [r["flagged"] for r in fl], "trend_p": [r["trend_p"] for r in fl],
                     "trend_size": [r["trend_size"] for r in fl], "mean": [r["mean"] for r in fl],
                     "rms": [r["rms"] for r in fl]})
    return {"draws": draws, "caught": caught, "rate": caught / draws, "rows": rows}


def identifiability():
    out = {}
    for label, truth, model in (("A", TRUTH_A, "neo_hookean"), ("B", TRUTH_B, "fibre")):
        for design in (("frequency",), ("strain",), ("compression",), ("strain", "compression"), DESIGN):
            clean = syn.clean(design, truth)
            prob = core.Problem(clean, model)
            th = syn.truth_theta(prob.names, truth)
            y = prob.y()
            Jm = np.asarray(prob.jacobian(jnp.asarray(th))) / np.abs(y)[:, None] / NOISE
            cov, ident, (vals, vecs) = core._covariance(Jm, 1.0, prob.names)
            se = np.sqrt(np.maximum(np.diag(cov), 0.0))
            out[f"{label}:{'+'.join(design)}"] = {
                "rel_se": dict(zip(prob.names, se.tolist())), "identified": ident,
                "weakest": dict(zip(prob.names, vecs[:, 0].tolist())), "eig_ratio": float(vals[0] / vals[-1])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part")
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()
    t0 = time.time()
    res: dict = {"part": a.part, "noise": NOISE, "truth_A": TRUTH_A, "truth_B": TRUTH_B}
    if a.part == "coverage_A":
        res["hessian"] = coverage(TRUTH_A, "neo_hookean", a.draws, 1)
    elif a.part == "coverage_B":
        res["hessian"] = coverage(TRUTH_B, "fibre", a.draws, 2)
    elif a.part == "bootstrap_A":
        rng = np.random.default_rng(3)
        clean = syn.clean(DESIGN, TRUTH_A)
        prob = core.Problem(clean, "neo_hookean")
        t_true = syn.truth_theta(prob.names, TRUTH_A)
        hits = np.zeros(2, int)
        widths = []
        for _ in range(a.draws):
            data = syn.noisy(clean, NOISE, rng)
            prob.datasets = data
            f = core.fit(prob, np.concatenate([d.y for d in data]))
            b = core.bootstrap(prob, f, 100, seed=int(rng.integers(1 << 30)))
            lo = np.log([b["interval"][n][0] for n in prob.names])
            hi = np.log([b["interval"][n][1] for n in prob.names])
            hits += (lo <= t_true) & (t_true <= hi)
            widths.append((hi - lo).tolist())
        res["bootstrap"] = {"draws": a.draws, "B": 100, "coverage": (hits / a.draws).tolist(),
                            "mean_width_log": np.mean(widths, axis=0).tolist()}
    elif a.part == "misfit":
        res["gent_jm2"] = misfit_rate(TRUTH_A, syn.gent_law(2.0), a.draws, 4)
    elif a.part == "resolution":
        res["sweep"] = {str(jm): misfit_rate(TRUTH_A, syn.gent_law(jm), a.draws, 5 + i)
                        for i, jm in enumerate((2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0))}
        for v in res["sweep"].values():
            v.pop("rows")
    elif a.part == "plants":
        res["unweighted_cov_A"] = coverage(TRUTH_A, "neo_hookean", a.draws, 6, cov_plant=unweighted_cov)
        res["cone_as_plate_A"] = coverage(TRUTH_A, "neo_hookean", a.draws, 7, problem_cls=ConeAsPlate)
        res["cone_as_plate_B"] = coverage(TRUTH_B, "fibre", a.draws, 8, problem_cls=ConeAsPlate)
        res["kappa_stop_gradient_A"] = coverage(TRUTH_A, "neo_hookean", a.draws, 9, problem_cls=ScaledDerivative,
                                                problem_kw={"name": "kappa", "scale": 0.0})
        res["mu_derivative_x2_A"] = coverage(TRUTH_A, "neo_hookean", a.draws, 10, problem_cls=ScaledDerivative,
                                             problem_kw={"name": "mu", "scale": 2.0})
        for v in res.values():
            if isinstance(v, dict):
                v.pop("rows", None)
    elif a.part == "identifiability":
        res["fisher"] = identifiability()
    else:
        raise SystemExit(f"unknown part {a.part}")
    res["wall_s"] = time.time() - t0
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("done", a.part, f"{res['wall_s']:.1f} s")


if __name__ == "__main__":
    main()
