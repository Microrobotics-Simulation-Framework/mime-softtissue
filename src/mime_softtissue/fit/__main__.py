"""``python -m mime_softtissue.fit MANIFEST.toml --out DIR [--model neo_hookean|fibre] [--bootstrap B]``: the fit,
its uncertainty, the misfit flags and the residual plot, written to ``DIR/fit.json`` and ``DIR/residuals.png``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import core
from .data import manifest
from .report import summary, write_plot

CANNOT = ("viscoelasticity: G'' is read and reported, never fitted; the law is elastic",
          "rate dependence: a frequency sweep's G' is fitted by one number; a trend with frequency is flagged, "
          "not modelled",
          "poroelastic drainage and fluid loss in compression (the rate of the test is not used)",
          "fracture: toughness and length scale need a notched test or a puncture, not these data",
          "the Poisson ratio from shear alone (simple shear does not change volume)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mime_softtissue.fit")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", choices=tuple(core.MODELS), default=None)
    ap.add_argument("--bootstrap", type=int, default=0)
    a = ap.parse_args(argv)
    datasets, opts = manifest(a.manifest)
    model = a.model or opts.get("model", "neo_hookean")
    angle = np.deg2rad(float(opts.get("fibre_angle_deg", 45.0)))
    prob = core.Problem(datasets, model, angle)
    f = core.fit(prob)
    flags = core.misfit(prob, f)
    boot = core.bootstrap(prob, f, a.bootstrap) if a.bootstrap else None
    a.out.mkdir(parents=True, exist_ok=True)
    rec = summary(prob, f, flags, boot)
    rec["cannot_identify"] = list(CANNOT)
    (a.out / "fit.json").write_text(json.dumps(rec, indent=1))
    write_plot(prob, f, flags, a.out / "residuals.png")
    print(json.dumps({k: rec[k] for k in ("model", "parameters", "relative_standard_error", "identified",
                                          "flagged")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
