"""The axisymmetric blunt punch with HERTZ-CHECK's axisymmetric settings (spike 3D-PUNCTURE's arm 2, `src/axi_run.py`, variant
v1, no plant), on this plugin's engine, with the volumetric switch: block radius 5 mm, height 5 mm, bottom fixed, side free;
gel E 50 kPa, nu 0.4, Gc 1 J/m^2 (input Gc/(1 + h/2l)), l 0.1 mm, k_res 1e-6; punch radius 0.5 mm, fillet 0.05 mm,
frictionless, nodal penalty 100 E/h, slip 1e-2 h; steps 0.02 mm to 0.44 mm; regularised Newton, cap 100; staggered cap
400, tol_d 1e-5; no viscosity; the proportional predictor; fine region 1.0 mm.

python axi_punch.py OUT_DIR --volumetric gauss|mean --h 0.1,0.0707,... [--step 0.02] [--max-depth 0.44]
Writes OUT_DIR/axi_<vol>_h<h>.json per grid."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from mime_softtissue.engine import contact as ct  # noqa: E402
from mime_softtissue.engine.fracture.phase_field import discretisation_toughness  # noqa: E402
from mime_softtissue.engine.material import Material, shear_and_bulk  # noqa: E402
from mime_softtissue.engine.solver import Solver  # noqa: E402
from mime_softtissue.engine.specimen import axisymmetric_block, with_indenter  # noqa: E402

E, NU, GC, LEN = 50e3, 0.4, 1.0, 0.1e-3
HB = 5e-3

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--h", default="0.1", help="mm, comma-separated")
ap.add_argument("--volumetric", default="gauss")
ap.add_argument("--radius", type=float, default=5.0, help="mm")
ap.add_argument("--fine", type=float, default=1.0, help="mm")
ap.add_argument("--step", type=float, default=0.02, help="mm")
ap.add_argument("--max-depth", type=float, default=0.44, help="mm")
ap.add_argument("--max-newton", type=int, default=100)
a = ap.parse_args()
Path(a.out).mkdir(parents=True, exist_ok=True)

for hs in a.h.split(","):
    h = float(hs) * 1e-3
    mu, kappa = shear_and_bulk(E, NU)
    gc = discretisation_toughness(GC, h, LEN)
    mat = Material(mu=jnp.asarray(mu), kappa=jnp.asarray(kappa), gc=jnp.asarray(gc), l=jnp.asarray(LEN),
                   k_res=jnp.asarray(1e-6), stretch_limit=jnp.asarray(2.0))
    ind = ct.Indenter("punch", radius=0.5e-3, fillet=0.05e-3, top=HB)
    block = axisymmetric_block(radius=a.radius * 1e-3, height=HB, h=h, fine_r=a.fine * 1e-3, fine_z=a.fine * 1e-3,
                               predictor="proportional")
    sp = with_indenter(block, ind, stiffness=100.0 * E / h, friction=0.0, slip=1e-2 * h)
    vol = {} if a.volumetric == "gauss" else {"volumetric": a.volumetric}
    solver = Solver(sp, "phase_field", tol_d=1e-5, max_newton=a.max_newton, max_staggered=400, viscosity=0.0,
                    newton_regularise=True, **vol)
    g = sp.grid
    rec = dict(args=dict(vars(a), h=float(hs)), cells=[g.nx, g.ny], gc_input=float(gc), rows=[])
    state = solver.initial()
    t0 = time.time()
    name = Path(a.out, f"axi_{a.volumetric}_h{hs}.json")
    for n in range(1, int(round(a.max_depth / a.step)) + 1):
        dl = n * a.step * 1e-3
        ts = time.time()
        state, out = solver.step(state, jnp.asarray(dl), mat)
        row = {k: float(v) for k, v in out._asdict().items()}
        row.update(depth_mm=dl * 1e3, step_s=time.time() - ts, d_max=float(jnp.max(state.d)))
        rec["rows"].append(row)
        print(f"h {hs} {a.volumetric} {n:3d} depth {dl * 1e3:6.3f} mm F {row['force'] * 1e3:9.4f} mN it "
              f"{int(row['iterations']):4d} newton {int(row['newton']):5d} capped {int(row['newton_capped'])} "
              f"dmax {row['d_max']:.3f} {row['step_s']:.1f}s", flush=True)
        name.write_text(json.dumps(rec, indent=1))
        if not np.isfinite(row["force"]):
            break
    rec["wall_s"] = time.time() - t0
    name.write_text(json.dumps(rec, indent=1))
