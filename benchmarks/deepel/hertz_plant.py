"""Controls (b) and (c) for the edge contact. (b) Hertz: R 4, block 50, nu 0.49, mean, to delta/R 0.2, h 0.2 and 0.1, edge
(1000 E/h) beside nodal (100 E/h). (c) The plant: the contact's tip radius 10 % wrong (x1.1 or x0.9, drawn), against the
declared radius's Hertz force; random draws at physical scales, each with its unplanted twin.
python hertz_plant.py hertz | plant SEED N (run from a folder holding ctl/; local runs pinned to cores 0-1)"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PY = sys.executable
RUN = str(Path(__file__).parent / "axi_deep.py")


def run(out, args):
    cmd = ["taskset", "-c", "0,1", PY, RUN, out] + [str(a) for a in args]
    subprocess.run(cmd, check=True, stdout=open(out.replace(".json", ".log"), "w"), stderr=subprocess.STDOUT)
    return json.load(open(out))


def hertz_ratio(d):
    a = d["args"]
    mu, nu, R = a["mu"], a["nu"], a["R"] * 1e-3
    E = 2 * mu * (1 + nu)
    Es = E / (1 - nu * nu)
    z = np.array([r["depth_mm"] for r in d["rows"]]) * 1e-3
    f = np.array([r["force_N"] for r in d["rows"]])
    return z / R, f / (4 / 3 * Es * np.sqrt(R) * z ** 1.5)


if sys.argv[1] == "hertz":
    for h in (0.2, 0.1):
        for c, k in (("edge", 1000), ("nodal", 100)):
            out = f"ctl/hz_{c}_h{h}.json"
            d = run(out, ["--R", 4.0, "--h", h, "--depth", 0.8, "--step", 0.05, "--block", 50, "--fine-r", 5,
                          "--fine-z", 5, "--cap", 2.0, "--contact", c, "--kfac", k])
            dr, r = hertz_ratio(d)
            keep = dr >= 0.0375 - 1e-9
            s01 = keep & (dr <= 0.1 + 1e-9)
            print(f"{c:5s} h {h}: F/Hertz {r[keep].min():.4f}-{r[keep].max():.4f} (to d/R 0.1: {r[s01].min():.4f}-"
                  f"{r[s01].max():.4f}); at d/R 0.05/0.1/0.2 {np.interp(0.05, dr, r):.4f} {np.interp(0.1, dr, r):.4f} "
                  f"{r[-1]:.4f}", flush=True)
else:
    seed, n = int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        R = float(rng.uniform(0.5, 4.0))
        hr = float(rng.uniform(0.025, 0.06))
        nu = float(rng.uniform(0.45, 0.49))
        dmax = float(rng.uniform(0.08, 0.15))
        fac = float(rng.choice([0.9, 1.1]))
        h = round(R * hr, 5)
        step = round(R * dmax / 4, 6)
        common = ["--R", R, "--h", h, "--nu", nu, "--depth", step * 4, "--step", step, "--block", 15 * R,
                  "--fine-r", 1.5 * R, "--fine-z", 1.5 * R, "--cap", R, "--contact", "edge", "--kfac", 1000]
        res = {}
        for tag, pf in (("ok", 1.0), ("plant", fac)):
            d = run(f"ctl/pl_{seed}_{i}_{tag}.json", common + ["--plant-radius", pf])
            dr, r = hertz_ratio(d)
            res[tag] = float(r[-1])
        line = dict(i=i, R=R, h_over_R=hr, nu=nu, d_over_R=dmax, factor=fac, ok=res["ok"], plant=res["plant"])
        rows.append(line)
        print(json.dumps(line), flush=True)
    json.dump(rows, open(f"ctl/plant_{seed}.json", "w"), indent=1)
