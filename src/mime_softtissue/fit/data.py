"""Reading a lab's data by the spec in ``docs/fit_spec.md``: a manifest (TOML) naming CSV files, each one test.

Columns (names fixed, any order, extra columns ignored, ``#`` lines are comments):

- rheology, strain sweep: ``strain`` (amplitude, a fraction: 0.01 is 1 %), ``storage_modulus_Pa``; optional
  ``loss_modulus_Pa`` (read and reported, never fitted), ``frequency_Hz``;
- rheology, frequency sweep: ``frequency_Hz``, ``storage_modulus_Pa``; optional ``loss_modulus_Pa``, ``strain``
  (else the manifest's ``strain``);
- unconfined compression: ``strain`` (engineering, compressive positive, a fraction), ``stress_Pa`` (nominal: force
  over the undeformed area, compressive positive)."""

from __future__ import annotations

import csv
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from .core import Dataset, keep_measurable

REQUIRED = {"shear_strain": ("strain", "storage_modulus_Pa"),
            "shear_frequency": ("frequency_Hz", "storage_modulus_Pa"),
            "compression": ("strain", "stress_Pa")}


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(line for line in fh if line.strip() and not line.lstrip().startswith("#"))]
    head = [h.strip() for h in rows[0]]
    cols: dict[str, list[float]] = {h: [] for h in head}
    for r in rows[1:]:
        for h, v in zip(head, r):
            v = v.strip()
            cols[h].append(float(v) if v else np.nan)
    return {h: np.asarray(v, float) for h, v in cols.items()}


def dataset(kind: str, path: Path, *, geometry: str = "cone_plate", strain: float | None = None,
            name: str | None = None) -> Dataset:
    cols = read_csv(path)
    missing = [c for c in REQUIRED[kind] if c not in cols]
    if missing:
        raise ValueError(f"{path.name}: a {kind} file needs the columns {REQUIRED[kind]}; missing {missing}")
    x = cols[REQUIRED[kind][0]]
    y = cols[REQUIRED[kind][1]]
    loss = cols.get("loss_modulus_Pa")
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    gamma0 = 0.01
    if kind == "shear_frequency":
        if "strain" in cols and np.all(np.isfinite(cols["strain"][ok])):
            gamma0 = float(np.median(cols["strain"][ok]))
        elif strain is not None:
            gamma0 = float(strain)
        else:
            raise ValueError(f"{path.name}: a frequency sweep needs its strain amplitude (a column or the manifest)")
    if np.any(y[ok] < 0):
        raise ValueError(f"{path.name}: negative values; the spec's sign convention is compressive positive")
    ds = Dataset(kind, x[ok], y[ok], name or path.stem, geometry, gamma0, None if loss is None else loss[ok])
    return keep_measurable(ds)


def manifest(path: Path) -> tuple[list[Dataset], dict[str, Any]]:
    """The tests a manifest names, and its ``[fit]`` options."""
    path = Path(path)
    spec = tomllib.loads(path.read_text())
    out: list[Dataset] = []
    for r in spec.get("rheology", []):
        sweep = r.get("sweep", "strain")
        kind = {"strain": "shear_strain", "frequency": "shear_frequency"}[sweep]
        geometry = r["geometry"]
        out.append(dataset(kind, path.parent / r["file"], geometry=geometry, strain=r.get("strain"),
                           name=r.get("name")))
    for c in spec.get("compression", []):
        out.append(dataset("compression", path.parent / c["file"], name=c.get("name")))
    if not out:
        raise ValueError("the manifest names no [[rheology]] or [[compression]] file")
    return out, spec.get("fit", {})


def write_csv(path: Path, columns: dict[str, np.ndarray], comment: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns)
    with open(path, "w", newline="") as fh:
        if comment:
            for line in comment.splitlines():
                fh.write(f"# {line}\n")
        w = csv.writer(fh)
        w.writerow(names)
        for row in zip(*[columns[n] for n in names]):
            w.writerow([f"{v:.10g}" for v in row])
