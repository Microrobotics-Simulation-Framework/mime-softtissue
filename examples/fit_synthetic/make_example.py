"""Writes a synthetic data set in the spec's layout (``docs/fit_spec.md``) from known parameters, for trying the fit
before real data arrive: ``python examples/fit_synthetic/make_example.py OUTDIR [--law neo_hookean|fibre|gent]``.
The values are placeholders, not a clot's."""
import argparse
from pathlib import Path

import numpy as np

from mime_softtissue.fit import synthetic as syn
from mime_softtissue.fit.data import write_csv

LAWS = {"neo_hookean": ({"mu": 1000.0 / 2.6, "kappa": 1000.0 / 1.2}, None),
        "fibre": ({"mu": 200.0, "kappa": 433.3, "k1": 100.0, "k2": 1.0}, None),
        "gent": ({"mu": 1000.0 / 2.6, "kappa": 1000.0 / 1.2}, 2.0)}

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--law", default="neo_hookean", choices=tuple(LAWS))
ap.add_argument("--noise", type=float, default=0.03)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
params, jm = LAWS[a.law]
law = syn.gent_law(jm) if jm else None
sets = syn.noisy(syn.clean(("strain", "frequency", "compression"), params, energy_law=law), a.noise,
                 np.random.default_rng(a.seed))
note = f"synthetic, law {a.law}, parameters {params}, {a.noise:.0%} noise [placeholder, not a measurement]"
strain, freq, comp = sets
write_csv(a.out / "strain_sweep.csv", {"strain": strain.x, "storage_modulus_Pa": strain.y,
                                       "frequency_Hz": np.ones_like(strain.x)}, note)
write_csv(a.out / "frequency_sweep.csv", {"frequency_Hz": freq.x, "storage_modulus_Pa": freq.y,
                                          "strain": np.full_like(freq.x, 0.01)}, note)
write_csv(a.out / "compression.csv", {"strain": comp.x, "stress_Pa": comp.y}, note)
(a.out / "manifest.toml").write_text(f"""# {note}
[fit]
model = "{'fibre' if a.law == 'fibre' else 'neo_hookean'}"
fibre_angle_deg = 45.0

[[rheology]]
file = "strain_sweep.csv"
sweep = "strain"
geometry = "parallel_plate"
name = "strain sweep"

[[rheology]]
file = "frequency_sweep.csv"
sweep = "frequency"
geometry = "parallel_plate"
name = "frequency sweep"

[[compression]]
file = "compression.csv"
name = "compression"
""")
print("wrote", a.out)
