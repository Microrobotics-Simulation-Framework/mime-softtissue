"""Reading the clot material records (``mime_softtissue.clot``) through MIME's resolver, so a licence's gate applies."""

import json
from collections.abc import Mapping
from typing import Any

from .engine.fracture.phase_field import discretisation_toughness
from .engine.material import Material, shear_and_bulk


def read_material(name: str) -> dict[str, float]:
    """The record's numbers, from the asset ``name`` through ``mime.assets.resolve``."""
    from mime.assets import reference, resolve

    path = resolve(reference(name))
    with open(path, encoding="utf-8") as f:
        return {key: float(value) for key, value in json.load(f)["values"].items()}


def engine_material(p: Mapping[str, Any], mode: str) -> Material:
    """The engine's :class:`Material` from a node's parameters (``young``, ``poisson``, ``toughness``,
    ``length_scale``, ``residual``, ``stretch_limit``, ``h``, and ``yeoh_c2`` where the record has Yeoh's stiffening);
    the phase field's input toughness carries the grid bias."""
    mu, kappa = shear_and_bulk(p["young"], p["poisson"])
    gc = p["toughness"]
    if mode == "phase_field":
        gc = discretisation_toughness(gc, p["h"], p["length_scale"])
    return Material(mu=mu, kappa=kappa, gc=gc, l=p["length_scale"], k_res=p["residual"],
                    stretch_limit=p["stretch_limit"], c2=p.get("yeoh_c2", 0.0))
