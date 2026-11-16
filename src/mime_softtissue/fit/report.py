"""The fit's record (JSON-ready) and its residual plot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import core

NEEDS = {"mu": "G' at small strain (any sweep), or the initial slope of compression together with nu",
         "kappa": "shear and compression together (G and E give nu), or compression to large strain (weakly)",
         "k1": "data beyond the linear range: G' rising with strain amplitude, or compression past about 20 %",
         "k2": "the curvature of the stiffening: data well into it (G' over at least a decade of its rise)"}


def summary(prob: core.Problem, f: core.Fit, flags: list[dict[str, Any]], boot: dict[str, Any] | None) -> dict[str, Any]:
    v = f.values
    nu = (3 * v["kappa"] - 2 * v["mu"]) / (2 * (3 * v["kappa"] + v["mu"]))
    rec: dict[str, Any] = {
        "model": prob.model, "parameters": v, "poisson_ratio_derived": nu,
        "relative_standard_error": f.rel_se, "interval_95_hessian": f.interval(),
        "identified": f.identified, "needs": {n: NEEDS[n] for n in prob.names},
        "noise_estimated": {d.name: float(s) for d, s in zip(prob.datasets, f.sigma)},
        "fisher_eigenvalues": f.eig[0].tolist() if f.eig else None,
        "weakest_direction": dict(zip(prob.names, f.eig[1][:, 0].tolist())) if f.eig else None,
        "datasets": [{k: r[k] for k in r if k != "trend"} for r in flags],
        "flagged": any(r["flagged"] for r in flags),
        "iterations": f.iterations,
    }
    if prob.model == "fibre":
        rec["fibre_angle_deg"] = float(np.rad2deg(prob.angle))
    loss = {d.name: float(np.median(d.loss / d.y)) for d in prob.datasets if d.loss is not None}
    if loss:
        rec["loss_tangent_median"] = loss  # G''/G': what the elastic law leaves out
    if boot:
        rec["interval_95_bootstrap"] = boot["interval"]
        rec["relative_standard_error_bootstrap"] = boot["rel_se"]
    return rec


def write_plot(prob: core.Problem, f: core.Fit, flags: list[dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(prob.datasets)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 6.0), squeeze=False, sharex="col",
                             gridspec_kw={"height_ratios": [2, 1]})
    idx = prob.set_index()
    label = {"shear_strain": ("strain amplitude", "G' (Pa)"), "shear_frequency": ("frequency (Hz)", "G' (Pa)"),
             "compression": ("engineering strain", "nominal stress (Pa)")}
    for k, (ds, fl) in enumerate(zip(prob.datasets, flags)):
        top, bot = axes[0, k], axes[1, k]
        m = f.prediction[idx == k]
        order = np.argsort(ds.x)
        top.plot(ds.x, ds.y, "o", ms=3, color="#444444", label="data")
        top.plot(ds.x[order], m[order], "-", color="#1f6fb4", label=f"fit ({prob.model})")
        top.set_ylabel(label[ds.kind][1])
        top.set_title(f"{ds.name}{'  FLAGGED' if fl['flagged'] else ''}", fontsize=9)
        if ds.kind != "compression":
            top.set_xscale("log")
            top.set_yscale("log")
        top.legend(fontsize=7)
        bot.plot(ds.x, 100 * f.residual[idx == k], "o", ms=3, color="#444444")
        bot.plot(ds.x[order], 100 * fl["trend"][order], "-", color="#c0392b",
                 label=f"cubic trend, p={fl['trend_p']:.1e}")
        bot.axhline(0, color="k", lw=0.5)
        bot.set_xlabel(label[ds.kind][0])
        bot.set_ylabel("(model - data)/data, %")
        bot.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
