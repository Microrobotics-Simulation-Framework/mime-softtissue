"""C-d: the detection rate of the plant (the axisymmetric mean dilatation without the 2 pi R weight) by the element oracle
(the deformed ring's volume by Pappus: the polygon through the displaced nodes, its area and centroid, no quadrature), over
N random draws at physical scales: element sizes 0.2-2 mm, the left radius 0 (the axis column, a quarter of the draws) or
0.1-10 mm, nodal displacements up to 0.15 of the element (u_R = 0 on the axis). Reads the error of the energy at a purely
volumetric material (mu = 0) over its own scale, for the correct build and the planted one.

python plant_rate.py N SEED OUT.json"""
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from mime_softtissue.engine.kinematics import axisymmetric as axi  # noqa: E402
from mime_softtissue.engine.material import Material, shear_and_bulk  # noqa: E402

n, seed, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
rng = np.random.default_rng(seed)
mu, kappa = shear_and_bulk(50e3, 0.45)
one = jnp.asarray(1.0)
mat = Material(mu=jnp.asarray(0.0), kappa=jnp.asarray(kappa), gc=one, l=one, k_res=jnp.asarray(0.0), stretch_limit=one)


def ring_jbar(x0, dx, dy, ue):
    p = np.array([[x0, 0.0], [x0 + dx, 0.0], [x0 + dx, dy], [x0, dy]]) + ue
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    return (np.sum((x + xn) * cross) / 6.0) / ((x0 + 0.5 * dx) * dx * dy)


energy = jax.jit(lambda ue, x0, dx, dy: axi.element_strain_energy(ue, jnp.zeros(4), x0, dx, dy, mat, "mean"))


def errors(draws):
    e_out = []
    for x0, dx, dy, ue in draws:
        jo = ring_jbar(x0, dx, dy, ue)
        V = 2 * np.pi * (x0 + 0.5 * dx) * dx * dy
        expect = V * 0.25 * kappa * (jo * jo - 1 - 2 * np.log(jo))
        e = float(energy(jnp.asarray(ue), jnp.asarray(x0), jnp.asarray(dx), jnp.asarray(dy)))
        e_out.append(abs(e - expect) / (V * kappa * (abs(jo - 1 / jo) + 1e-3)))
    return np.array(e_out)


draws = []
for k in range(n):
    dx, dy = rng.uniform(0.2e-3, 2e-3, 2)
    x0 = 0.0 if rng.random() < 0.25 else rng.uniform(0.1e-3, 10e-3)
    ue = rng.uniform(-0.15, 0.15, (4, 2)) * min(dx, dy)
    if x0 == 0.0:
        ue[[0, 3], 0] = 0.0
    draws.append((x0, dx, dy, ue))
good = errors(draws)
axi.mean_dilatation = lambda J, w: jnp.mean(J)
energy = jax.jit(lambda ue, x0, dx, dy: axi.element_strain_energy(ue, jnp.zeros(4), x0, dx, dy, mat, "mean"))
bad = errors(draws)
line = 1e-10
res = dict(n=n, seed=seed, line=line, correct_pass=int(np.sum(good < line)), correct_max=float(good.max()),
           plant_caught=int(np.sum(bad > line)), plant_min=float(bad.min()), plant_median=float(np.median(bad)),
           plant_min_axis=float(bad[[d[0] == 0.0 for d in draws]].min()),
           plant_min_off=float(bad[[d[0] > 0.0 for d in draws]].min()))
print(json.dumps(res, indent=1))
json.dump(res, open(out, "w"), indent=1)
