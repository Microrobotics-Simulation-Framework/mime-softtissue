"""The friction law's closed-form check: an elastic tube on a rigid rod with an interference fit, the rod slid along
its axis. Steady force F = (mu p + tau_a) 2 pi a L, with p from Lame's thick cylinder in plane strain, the outer face
free: p = E delta (b^2 - a^2) / (a (1 + nu) ((1 - 2 nu) a^2 + b^2)), delta the interference less the penalty's
penetration p/k [derived].

    python scripts/pullout.py --out FILE [--friction 0.3] [--adhesion 0] [--plant none|no_2_pi_r|one_lag|normal_z]"""
import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from mime_softtissue.engine import contact as ct
from mime_softtissue.engine.material import Material, shear_and_bulk
from mime_softtissue.engine.solver import Solver
from mime_softtissue.engine.specimen import tube_on_rod

E, NU = 50e3, 0.4
A, B, LEN, H = 0.5e-3, 2.5e-3, 1.0e-3, 0.05e-3
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--friction", type=float, default=0.3)
ap.add_argument("--adhesion", type=float, default=0.0)
ap.add_argument("--interference", type=float, default=1e-3, help="relative to the inner radius")
ap.add_argument("--plant", default="none", choices=("none", "no_2_pi_r", "one_lag", "normal_z"))
ap.add_argument("--lag-relaxation", type=float, default=1.0)
ap.add_argument("--max-newton", type=int, default=40)
a = ap.parse_args()

if a.plant == "normal_z":  # the lagged normal along the axis instead of the surface's
    real = ct.pressure_and_normal

    def planted(c, X, u, depth):
        p, n = real(c, X, u, depth)
        return p, jnp.zeros_like(n).at[..., 1].set(1.0)
    ct.pressure_and_normal = planted

k = 100 * E / H
sp = tube_on_rod(inner=A, outer=B, length=LEN, h=H, interference=a.interference * A, stiffness=k,
                 friction=a.friction, slip=1e-2 * H, adhesion=a.adhesion, weight_plant=a.plant == "no_2_pi_r")
mu, kappa = shear_and_bulk(E, NU)
mat = Material(mu=jnp.asarray(mu), kappa=jnp.asarray(kappa), gc=jnp.asarray(1.0), l=jnp.asarray(1e-4),
               k_res=jnp.asarray(1e-6), stretch_limit=jnp.asarray(10.0))
solver = Solver(sp, "elastic", tol_p=1e30 if a.plant == "one_lag" else 1e-3, max_newton=a.max_newton,
                lag_relaxation=a.lag_relaxation)
state = solver.initial()
base = 5e-3  # the rod's end 5 mm below the tube
rows = []
for n, s in enumerate([0.0, 0.02e-3, 0.04e-3, 0.06e-3, 0.08e-3]):
    state, out = solver.step(state, jnp.asarray(base + s), mat)
    rows.append({key: float(v) for key, v in out._asdict().items()} | {"slide_mm": s * 1e3})
p_inner = float(np.max(np.asarray(ct.pressure_and_normal(sp.contact, solver.X, state.u, jnp.asarray(base + 0.08e-3))[0])[0, 1:-1]))
delta = a.interference * A
p_lame = E * delta * (B ** 2 - A ** 2) / (A * (1 + NU) * ((1 - 2 * NU) * A ** 2 + B ** 2))
p_lame_pen = p_lame / (1 + p_lame / (k * delta))  # the penetration p/k taken off the interference
F_exact = (a.friction * p_lame_pen + a.adhesion) * 2 * np.pi * A * LEN
F = rows[-1]["force"]
rec = dict(args=vars(a), rows=rows, pressure_inner_max=p_inner, pressure_lame=p_lame,
           pressure_lame_less_penetration=p_lame_pen, force=F, force_exact=F_exact, error=F / F_exact - 1.0,
           force_previous_step=rows[-2]["force"])
Path(a.out).write_text(json.dumps(rec, indent=1))
print(json.dumps({k_: v for k_, v in rec.items() if k_ != "rows"}, indent=1))
