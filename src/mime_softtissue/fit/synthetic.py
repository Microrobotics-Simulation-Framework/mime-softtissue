"""Synthetic lab data from known parameters, for the harness's controls: the design (which tests, at which points),
the truth (the plugin's law, or a Gent law the fit does not have), and multiplicative Gaussian noise."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import homogeneous as hom
from .core import Dataset, keep_measurable

STRAIN_SWEEP = np.logspace(-3, 0, 21)  # amplitude 0.1 % to 100 %
FREQUENCIES = np.logspace(-1, 1, 16)  # 0.1 to 10 Hz at strain 0.01
COMPRESSION = np.linspace(0.0, 0.6, 61)  # engineering strain


def gent_law(jm: float):
    """psi = -(mu J_m / 2) ln(1 - (J^-2/3 I1 - 3) / J_m) + the plugin's volumetric term: stiffening without a fibre."""
    def energy(F, mat):
        J = jnp.linalg.det(F)
        I1b = J ** (-2.0 / 3.0) * jnp.sum(F * F)
        iso = -0.5 * mat.mu * jm * jnp.log(1.0 - (I1b - 3.0) / jm)
        vol = 0.25 * mat.kappa * (J ** 2 - 1.0 - 2.0 * jnp.log(J))
        return iso + vol
    return energy


def clean(design: tuple[str, ...], params: dict[str, float], angle: float = np.pi / 4, energy_law: Any = None,
          geometry: str = "parallel_plate") -> list[Dataset]:
    """Noise-free data sets of ``design`` (any of "strain", "frequency", "compression")."""
    mat = hom.material_from({k: jnp.asarray(v) for k, v in params.items()}, angle)
    out = []
    for kind in design:
        if kind == "strain":
            y = jax.vmap(lambda g: hom.storage_modulus(g, mat, geometry, energy_law))(jnp.asarray(STRAIN_SWEEP))
            out.append(Dataset("shear_strain", STRAIN_SWEEP.copy(), np.asarray(y), "strain sweep", geometry))
        elif kind == "frequency":
            g = hom.storage_modulus(jnp.asarray(0.01), mat, geometry, energy_law)
            out.append(Dataset("shear_frequency", FREQUENCIES.copy(), np.full(FREQUENCIES.size, float(g)),
                               "frequency sweep", geometry, 0.01))
        elif kind == "compression":
            y = jax.vmap(lambda e: hom.compression_stress(e, mat, energy_law))(jnp.asarray(COMPRESSION))
            out.append(keep_measurable(Dataset("compression", COMPRESSION.copy(), np.asarray(y), "compression")))
        else:
            raise ValueError(kind)
    return out


def noisy(sets: list[Dataset], rel: float, rng: np.random.Generator) -> list[Dataset]:
    return [Dataset(d.kind, d.x, d.y * (1.0 + rel * rng.standard_normal(d.y.size)), d.name, d.geometry, d.gamma0,
                    d.loss, d.dropped) for d in sets]


def truth_theta(names: tuple[str, ...], params: dict[str, float]) -> np.ndarray:
    return np.log(np.array([params[n] for n in names]))
