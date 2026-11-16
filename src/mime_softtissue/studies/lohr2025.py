"""Lohr et al. 2025's pure-shear data, read from the plugin's records, and the study's target built from them.

The records (CC0 1.0, in ``records/data/``): ``softtissue.lohr2025_pure_shear``, the 15 force-displacement CSVs of
doi:10.18738/T8/97ZAZN verbatim, and ``softtissue.lohr2025_critical_displacements``, the DIC critical displacements of
doi:10.18738/T8/DUJWSR. Each CSV's text is checked against its sha256 when read.

What the data set does not carry, and where it comes from here:
- **each notched sample's notch length** is the paper's Fig. 5A [digitised] (``FIG5A``); the files carry only nominal
  sizes (5, 10, 15, 20 mm) in the sheet's sample names. Each file was given the Fig. 5A marker of its nominal size whose
  Griffith-A toughness matches the one recomputed from the raw data (``griffith_a``): all 12 differ by the same
  -0.015 J/m^2, and the match over all 12 at once, ignoring the names, gives the same groups.
- **the groups** are the paper's five (mean notches 3.70, 7.33, 9.70, 11.90 and 16.51 mm).

The displacement in every file rises at 0.2000 mm/s; the paper's text says 0.1 mm/s. The model is quasi-static, so its
force against displacement does not depend on it."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from importlib import resources
from typing import Any

import numpy as np

CURVES = "data/lohr2025_pure_shear.json"
CRITICAL = "data/lohr2025_critical_displacements.json"
WIDTH_MM, HEIGHT_MM, THICKNESS_MM = 40.0, 10.0, 3.0
GRIP_RATE_MM_S = 0.2
STEP_MM = 0.02
UNNOTCHED = ("Unnotched_1", "Unnotched_2", "Unnotched_3")

#: Fig. 5A [digitised, vector centres]: each notched file's notch length (mm), its Griffith-A toughness as plotted
#: (J/m^2) and its group.
FIG5A: dict[str, tuple[float, float, int]] = {
    "Notched_1": (3.839, 1.0594, 1), "Notched_2": (3.783, 1.4483, 1), "Notched_3": (3.503, 1.3109, 1),
    "Notched_4": (7.067, 0.9175, 2), "Notched_5": (7.603, 0.3695, 2), "Notched_6": (9.709, 1.4134, 3),
    "Notched_7": (12.198, 0.7130, 4), "Notched_8": (11.226, 0.7615, 4), "Notched_9": (12.276, 0.4959, 4),
    "Notched_10": (16.754, 0.5986, 5), "Notched_11": (15.850, 0.6209, 5), "Notched_12": (16.935, 1.3453, 5),
}


def _record(path: str) -> dict[str, Any]:
    return json.loads(resources.files("mime_softtissue.records").joinpath(path).read_text(encoding="utf-8"))


def curves() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each file's (displacement mm, force N), its text checked against the record's sha256."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, f in _record(CURVES)["files"].items():
        text = f["text"]
        if hashlib.sha256(text.encode("ascii")).hexdigest() != f["sha256"]:
            raise ValueError(f"{name}: its text does not match its sha256")
        rows = list(csv.reader(io.StringIO(text, newline="")))
        a = np.array([[float(v) for v in r] for r in rows[2:] if r])
        out[name.removesuffix(".csv")] = (a[:, 1], a[:, 2])
    return out


def critical_displacements() -> dict[str, float]:
    """Each notched file's critical displacement (mm), from the DIC images (the sheet's row i is ``Notched_i``)."""
    return {r["csv"].removesuffix(".csv"): float(r["critical_displacement_mm"]) for r in _record(CRITICAL)["rows"]}


def mean_curve(data: dict[str, tuple[np.ndarray, np.ndarray]], names: list[str] | tuple[str, ...]
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The files' mean force and its sample SD (NaN for one file) on a ``STEP_MM`` grid, to the shortest record."""
    x_end = min(float(data[n][0][-1]) for n in names)
    xs = np.round(np.arange(0.0, x_end + 1e-9, STEP_MM), 6)
    F = np.array([np.interp(xs, *data[n]) for n in names])
    sd = F.std(axis=0, ddof=1) if len(names) > 1 else np.full(len(xs), np.nan)
    return xs, F.mean(axis=0), sd


def griffith_a(delta_c_mm: float, data: dict[str, tuple[np.ndarray, np.ndarray]] | None = None) -> float:
    """Griffith A (J/m^2): the unnotched mean's work to ``delta_c_mm`` over the nominal section (width x thickness)."""
    xs, m, _ = mean_curve(data or curves(), UNNOTCHED)
    work = np.concatenate([[0.0], np.cumsum(0.5 * (m[1:] + m[:-1]) * np.diff(xs))]) * 1e-3
    return float(np.interp(delta_c_mm, xs, work)) / (WIDTH_MM * THICKNESS_MM * 1e-6)


def target() -> dict[str, Any]:
    """The study's target: the unnotched mean and +-1 SD, each notched sample, each group's notch, crack start and mean
    curve with the SD across its samples at the mean's peak [measured]; notch lengths [digitised, Fig. 5A]."""
    data = curves()
    dc = critical_displacements()
    ux, um, us = mean_curve(data, UNNOTCHED)
    i = int(np.argmax(um))
    samples = [{"csv": n, "notch_mm": a, "group": g, "delta_c_mm": dc[n],
                "gc_griffith_A_raw": round(griffith_a(dc[n], data), 4), "gc_griffith_A_fig5": gA}
               for n, (a, gA, g) in FIG5A.items()]
    groups: dict[str, dict[str, Any]] = {}
    group_curves: dict[str, dict[str, Any]] = {}
    for g in range(1, 6):
        names = [n for n, (_, _, group) in FIG5A.items() if group == g]
        s = [x for x in samples if x["group"] == g]
        a = np.array([FIG5A[n][0] for n in names])
        d = np.array([dc[n] for n in names])
        xs, m, sd = mean_curve(data, names)
        k = int(np.argmax(m))
        peaks = np.array([data[n][1].max() for n in names])
        groups[str(g)] = {"n": len(s), "samples": names, "notch_mean_mm": round(float(a.mean()), 3),
                          "notch_sd_mm": round(float(a.std(ddof=1)), 3) if len(s) > 1 else None,
                          "delta_c_mean_mm": round(float(d.mean()), 4),
                          "delta_c_sd_mm": round(float(d.std(ddof=1)), 4) if len(s) > 1 else None}
        group_curves[str(g)] = {"points": np.column_stack([xs, m])[::5].round(5).tolist(),
                                "sd": None if len(s) == 1 else sd[::5].round(5).tolist(),
                                "peak_N": round(float(m[k]), 5), "peak_at_mm": round(float(xs[k]), 3),
                                "peak_sd_N": None if len(s) == 1 else round(float(sd[k]), 5),
                                "sample_peaks_N": peaks.round(5).tolist(),
                                "sample_peak_sd_N": None if len(s) == 1 else round(float(peaks.std(ddof=1)), 5)}
    unnotched_samples = {n: {"peak_N": float(data[n][1].max()), "peak_at_mm": float(data[n][0][int(np.argmax(data[n][1]))])}
                         for n in UNNOTCHED}
    return {"source": "Lohr, Bechtel, Dortdivanlioglu, Rausch (2025) Int J Fract, doi:10.1007/s10704-024-00820-4; data "
                      "doi:10.18738/T8/97ZAZN and doi:10.18738/T8/DUJWSR (CC0 1.0), the records "
                      "softtissue.lohr2025_pure_shear and softtissue.lohr2025_critical_displacements",
            "labels": {"curves": "[measured] the raw files; means and sample SDs on a 0.02 mm grid, to the shortest record",
                       "delta_c": "[measured] the DIC critical displacements",
                       "notch_mm": "[digitised] Fig. 5A, matched to the files by Griffith A within each nominal size"},
            "specimen_mm": {"width": WIDTH_MM, "height": HEIGHT_MM, "thickness": THICKNESS_MM},
            "rate_mm_s": GRIP_RATE_MM_S,
            "unnotched": {"mean": np.column_stack([ux, um])[::5].round(5).tolist(),
                          "band": {"x": ux[::5].round(3).tolist(), "upper": (um + us)[::5].round(5).tolist(),
                                   "lower": (um - us)[::5].round(5).tolist()},
                          "peak_N": round(float(um[i]), 5), "peak_at_mm": round(float(ux[i]), 3),
                          "peak_sd_N": round(float(us[i]), 5), "samples": unnotched_samples},
            "samples": samples, "groups": groups, "group_curves": group_curves}
