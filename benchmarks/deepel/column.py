"""Elements of the first columns near the tip: python column.py FIELDS.npz KEY cols rows_from rows_to
Per element (centre): reference (R, depth), current position relative to the tip, principal in-plane stretches, hoop
stretch, J."""
import json, sys
import numpy as np
f = np.load(sys.argv[1]); key = sys.argv[2]; cols = int(sys.argv[3]); j0, j1 = int(sys.argv[4]), int(sys.argv[5])
x, y = f["x"], f["y"]; u = f[key].astype(float)
args = json.load(open(sys.argv[1].replace("_fields.npz", ".json")))["args"]
depth = int(key[2:]) * args["step"] * 1e-3; top = y[-1]; tip = top - depth
for i in range(cols):
    for j in range(j0, j1):
        ue = np.array([u[i, j], u[i + 1, j], u[i + 1, j + 1], u[i, j + 1]])
        dx, dy = x[i + 1] - x[i], y[j + 1] - y[j]
        dudr = 0.5 * ((ue[1] - ue[0]) + (ue[2] - ue[3])) / dx
        dudz = 0.5 * ((ue[3] - ue[0]) + (ue[2] - ue[1])) / dy
        F = np.eye(2) + np.stack([dudr, dudz], axis=1)
        Rc = 0.5 * (x[i] + x[i + 1]); uc = ue.mean(0)
        Ftt = 1 + uc[0] / Rc
        s = np.linalg.svd(F, compute_uv=False)
        J = np.linalg.det(F) * Ftt
        pos = np.array([Rc, 0.5 * (y[j] + y[j + 1])]) + uc
        print(f"col {i} row {j}: ref R {Rc*1e3:.3f} depth {(top-0.5*(y[j]+y[j+1]))*1e3:.3f} | now R {pos[0]*1e3:.3f} "
              f"Z-tip {(pos[1]-tip)*1e3:+.3f} | in-plane {s[0]:.2f} {s[1]:.3f} hoop {Ftt:.2f} J {J:.3f}")
