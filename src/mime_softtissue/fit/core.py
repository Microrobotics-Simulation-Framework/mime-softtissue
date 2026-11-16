"""Fitting the plugin's law to rheology and unconfined compression: weighted least squares on relative residuals
(Levenberg-Marquardt, the Jacobian by forward-mode differentiation in JAX), parameters in logarithms (so that they stay
positive and their standard errors read as relative ones), each data set weighted by its own noise level estimated
from its residuals; the uncertainty from the Gauss-Newton Hessian (or a residual bootstrap); a misfit flag per data
set; and the identifiability of each parameter from the Fisher information."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from . import homogeneous as hom

KINDS = ("shear_strain", "shear_frequency", "compression")
MODELS = {"neo_hookean": ("mu", "kappa"), "fibre": ("mu", "kappa", "k1", "k2")}
NOT_IDENTIFIED = 0.5  # a relative standard error above this: the data do not identify the parameter
FLAG_P, FLAG_SIZE = 1e-3, 0.03  # the misfit flag: a cubic trend in the residuals with p below FLAG_P and size FLAG_SIZE


@dataclass
class Dataset:
    """One test's points. ``x``: the strain amplitude (shear_strain), the frequency in Hz (shear_frequency) or the
    engineering strain (compression); ``y``: G' or the nominal stress in Pa; ``gamma0``: a frequency sweep's
    amplitude; ``loss``: G'' if given (reported, never fitted)."""

    kind: str
    x: np.ndarray
    y: np.ndarray
    name: str = ""
    geometry: str = "cone_plate"
    gamma0: float = 0.01
    loss: np.ndarray | None = None
    dropped: int = 0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind is one of {KINDS}")


def keep_measurable(ds: Dataset, floor: float = 0.01) -> Dataset:
    """Drops points whose value is below ``floor`` of the set's largest (a relative residual is meaningless there)."""
    ok = np.abs(ds.y) >= floor * np.max(np.abs(ds.y))
    loss = None if ds.loss is None else ds.loss[ok]
    return Dataset(ds.kind, ds.x[ok], ds.y[ok], ds.name, ds.geometry, ds.gamma0, loss, int(np.sum(~ok)))


class Problem:
    """The model's prediction for a fixed set of tests (their x), as a function of the log parameters."""

    def __init__(self, datasets: list[Dataset], model: str = "neo_hookean", angle: float = np.pi / 4,
                 ) -> None:
        if model not in MODELS:
            raise ValueError(f"model is one of {tuple(MODELS)}")
        self.datasets, self.model, self.angle = datasets, model, angle
        self.names = MODELS[model]
        self.sizes = [len(d.x) for d in datasets]
        self.predict = jax.jit(self._predict)
        self.jacobian = jax.jit(jax.jacfwd(self._predict))

    def params(self, theta: Any) -> dict[str, Any]:
        return {n: jnp.exp(theta[i]) for i, n in enumerate(self.names)}

    def _one(self, ds: Dataset, mat: Any) -> jax.Array:
        geom = ds.geometry
        x = jnp.asarray(ds.x)
        if ds.kind == "shear_strain":
            return jax.vmap(lambda g: hom.storage_modulus(g, mat, geom))(x)
        if ds.kind == "shear_frequency":
            return jnp.full(x.shape, hom.storage_modulus(jnp.asarray(ds.gamma0), mat, geom))
        return jax.vmap(lambda e: hom.compression_stress(e, mat))(x)

    def _predict(self, theta: Any) -> jax.Array:
        mat = hom.material_from(self.params(theta), self.angle)
        return jnp.concatenate([self._one(d, mat) for d in self.datasets])

    def y(self) -> np.ndarray:
        return np.concatenate([d.y for d in self.datasets])

    def set_index(self) -> np.ndarray:
        return np.concatenate([np.full(n, k) for k, n in enumerate(self.sizes)])

    def initial(self) -> np.ndarray:
        """A first guess from the data: mu from the smallest-strain G' or the compression's initial slope."""
        mu0 = None
        for d in self.datasets:
            if d.kind.startswith("shear"):
                mu0 = float(d.y[np.argmin(d.x)]) if d.kind == "shear_strain" else float(np.median(d.y))
                break
        if mu0 is None:
            d = next(d for d in self.datasets if d.kind == "compression")
            k = np.argsort(d.x)[:max(3, len(d.x) // 10)]
            mu0 = float(np.sum(d.y[k] * d.x[k]) / np.sum(d.x[k] ** 2)) / 2.6
        guess = {"mu": mu0, "kappa": 2.17 * mu0, "k1": 0.3 * mu0, "k2": 1.0}
        return np.log(np.array([guess[n] for n in self.names]))


@dataclass
class Fit:
    names: tuple[str, ...]
    theta: np.ndarray  # log parameters
    cov: np.ndarray  # covariance of theta (Gauss-Newton)
    sigma: np.ndarray  # each data set's relative noise level, estimated
    cost: float
    iterations: int
    prediction: np.ndarray
    residual: np.ndarray  # relative, (model - data) / |data|
    identified: dict[str, bool] = field(default_factory=dict)
    eig: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def values(self) -> dict[str, float]:
        return {n: float(np.exp(t)) for n, t in zip(self.names, self.theta)}

    @property
    def rel_se(self) -> dict[str, float]:
        return {n: float(np.sqrt(max(self.cov[i, i], 0.0))) for i, n in enumerate(self.names)}

    def interval(self, z: float = 1.959964) -> dict[str, tuple[float, float]]:
        se = np.sqrt(np.maximum(np.diag(self.cov), 0.0))
        return {n: (float(np.exp(t - z * s)), float(np.exp(t + z * s))) for n, t, s in zip(self.names, self.theta, se)}


def _lm(problem: Problem, y: np.ndarray, theta: np.ndarray, w: np.ndarray, max_iter: int = 200
        ) -> tuple[np.ndarray, float, int]:
    absy = np.abs(y)

    def resid(th):
        m = np.asarray(problem.predict(jnp.asarray(th)))
        return (m - y) / absy * w

    r = resid(theta)
    cost = float(r @ r)
    lam = 1e-3
    for it in range(max_iter):
        Jm = np.asarray(problem.jacobian(jnp.asarray(theta))) / absy[:, None] * w[:, None]
        A, g = Jm.T @ Jm, Jm.T @ r
        D = np.diag(np.maximum(np.diag(A), 1e-12 * max(np.max(np.diag(A)), 1e-300)))
        improved = False
        step = trial = theta  # the 30 trials below always assign these; the checker cannot see it
        rt, ct = r, np.inf
        for _ in range(30):
            step = -np.linalg.solve(A + lam * D, g)
            step = step * min(1.0, 2.0 / (np.max(np.abs(step)) + 1e-300))  # at most a factor e^2 a step
            trial = theta + step
            rt = resid(trial)
            ct = float(rt @ rt) if np.all(np.isfinite(rt)) else np.inf
            if ct < cost:
                improved = True
                break
            lam *= 4.0
        if not improved:
            return theta, cost, it
        done = (cost - ct) <= 1e-14 * cost or np.max(np.abs(step)) < 1e-11
        theta, r, cost = trial, rt, ct
        lam = max(lam / 3.0, 1e-12)
        if done:
            return theta, cost, it + 1
    return theta, cost, max_iter


def _covariance(Jm: np.ndarray, dof_scale: float, names: tuple[str, ...]
                ) -> tuple[np.ndarray, dict[str, bool], tuple[np.ndarray, np.ndarray]]:
    A = Jm.T @ Jm
    vals, vecs = np.linalg.eigh(A)
    floor = 1e-12 * max(vals[-1], 1e-300)
    inv = np.where(vals > floor, 1.0 / np.maximum(vals, floor), np.inf)
    with np.errstate(invalid="ignore"):
        cov = (vecs * np.where(np.isfinite(inv), inv, 0.0)) @ vecs.T * dof_scale
    singular = ~np.isfinite(inv)
    for k in np.where(singular)[0]:  # a direction the data do not see: every parameter on it is unidentified
        for i in np.where(np.abs(vecs[:, k]) > 0.1)[0]:
            cov[i, i] = np.inf
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    identified = {n: bool(np.isfinite(s) and s <= NOT_IDENTIFIED) for n, s in zip(names, se)}
    return cov, identified, (vals, vecs)


def fit(problem: Problem, y: np.ndarray | None = None, theta0: np.ndarray | None = None,
        starts: int | None = None) -> Fit:
    """The fit: two passes (equal weights, then each data set at its own estimated noise level); for the fibre model,
    several starts in (k1, k2) and the best kept."""
    y = problem.y() if y is None else np.asarray(y)
    base = problem.initial() if theta0 is None else np.asarray(theta0, float)
    idx = problem.set_index()
    candidates = [base]
    if problem.model == "fibre" and theta0 is None:
        n_k = 3 if starts is None else starts
        for k1f in (0.1, 1.0):
            for k2 in np.logspace(-1, 1, n_k):
                c = base.copy()
                c[2] = base[0] + np.log(k1f)
                c[3] = np.log(k2)
                candidates.append(c)
    best = None
    for c in candidates:
        th, cost, it = _lm(problem, y, c, np.ones_like(y))
        if best is None or cost < best[1]:
            best = (th, cost, it)
    assert best is not None
    theta = best[0]
    m = np.asarray(problem.predict(jnp.asarray(theta)))
    e = (m - y) / np.abs(y)
    sigma = np.array([np.sqrt(np.mean(e[idx == k] ** 2)) for k in range(len(problem.sizes))])
    sigma = np.maximum(sigma, 1e-12)
    w = 1.0 / sigma[idx]
    theta, cost, it2 = _lm(problem, y, theta, w)
    m = np.asarray(problem.predict(jnp.asarray(theta)))
    e = (m - y) / np.abs(y)
    Jm = np.asarray(problem.jacobian(jnp.asarray(theta))) / np.abs(y)[:, None] * w[:, None]
    n, p = len(y), len(problem.names)
    cov, identified, eig = _covariance(Jm, cost / max(n - p, 1), problem.names)
    return Fit(problem.names, theta, cov, sigma, cost, best[2] + it2, m, e, identified, eig)


def bootstrap(problem: Problem, f: Fit, B: int = 200, seed: int = 0) -> dict[str, Any]:
    """A residual bootstrap: each data set's relative residuals resampled with replacement onto the fitted curve,
    refitted from the fit; percentile intervals (2.5, 97.5) of the parameters."""
    rng = np.random.default_rng(seed)
    idx = problem.set_index()
    thetas = []
    for _ in range(B):
        e_star = np.empty_like(f.residual)
        for k in range(len(problem.sizes)):
            pool = f.residual[idx == k]
            e_star[idx == k] = rng.choice(pool, size=pool.size, replace=True)
        y_star = f.prediction * (1.0 - e_star)
        thetas.append(fit(problem, y_star, theta0=f.theta).theta)
    T = np.array(thetas)
    lo, hi = np.percentile(T, 2.5, axis=0), np.percentile(T, 97.5, axis=0)
    return {"theta": T, "interval": {n: (float(np.exp(a)), float(np.exp(b))) for n, a, b in zip(f.names, lo, hi)},
            "rel_se": {n: float(s) for n, s in zip(f.names, T.std(axis=0, ddof=1))}}


def misfit(problem: Problem, f: Fit) -> list[dict[str, Any]]:
    """Per data set: a cubic in the log of x fitted to the relative residuals (F-test against a constant, and the
    trend's largest size over the data), and the mean residual (a data set the others disagree with); flagged by the
    rule fixed in advance (``FLAG_P``, ``FLAG_SIZE``)."""
    out = []
    idx = problem.set_index()
    for k, ds in enumerate(problem.datasets):
        e = f.residual[idx == k]
        x = np.log(ds.x)
        n = e.size
        rec: dict[str, Any] = {"name": ds.name, "kind": ds.kind, "points": int(n), "dropped": ds.dropped,
                               "rms": float(np.sqrt(np.mean(e ** 2)))}
        mean = float(np.mean(e))
        rec["mean"] = mean
        t = mean / (np.std(e, ddof=1) / np.sqrt(n)) if n > 2 and np.std(e) > 0 else 0.0
        rec["mean_p"] = float(2 * stats.t.sf(abs(t), n - 1)) if n > 2 else 1.0
        if n >= 8 and np.ptp(x) > 0:
            xs = (x - x.mean()) / (np.ptp(x) / 2)
            V = np.vander(xs, 4)
            coef, *_ = np.linalg.lstsq(V, e, rcond=None)
            trend = V @ coef
            rss3 = float(np.sum((e - trend) ** 2))
            rss0 = float(np.sum((e - mean) ** 2))
            F = ((rss0 - rss3) / 3) / (rss3 / (n - 4)) if rss3 > 0 else np.inf
            rec["trend_p"] = float(stats.f.sf(F, 3, n - 4)) if np.isfinite(F) else 0.0
            rec["trend_size"] = float(np.max(np.abs(trend)))  # the cubic itself, as the rule fixed in advance
            rec["trend"] = trend
        else:
            rec["trend_p"], rec["trend_size"], rec["trend"] = 1.0, 0.0, np.zeros(n)
        rec["flag_shape"] = bool(rec["trend_p"] < FLAG_P and rec["trend_size"] >= FLAG_SIZE)
        rec["flag_offset"] = bool(len(problem.datasets) > 1 and rec["mean_p"] < FLAG_P and abs(mean) >= FLAG_SIZE)
        rec["flagged"] = rec["flag_shape"] or rec["flag_offset"]
        out.append(rec)
    return out
