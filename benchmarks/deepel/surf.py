"""The deformed top surface near the tip: python surf.py FIELDS.npz R_mm KEY [n]: per top node (from the axis), the
reference R, deformed (R, Z - tip), its signed distance to the tip, and the edge length; and the sub-surface nodes inside."""
import sys
import numpy as np
f = np.load(sys.argv[1]); R = float(sys.argv[2]) * 1e-3; key = sys.argv[3]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 14
x, y = f["x"], f["y"]; u = f[key].astype(float)
import json
step = int(key[2:])
args = json.load(open(sys.argv[1].replace("_fields.npz", ".json")))["args"]
depth = step * args["step"] * 1e-3
top = y[-1]; tip = top - depth
def phi(r, z):
    zc = tip + R
    qz = zc - z
    return np.where(qz > 0, np.hypot(r, qz) - R, np.abs(r) - R)
X = x + u[:, -1, 0]; Z = top + u[:, -1, 1]
print(f"{key}: depth {depth*1e3:.2f} mm")
for i in range(n):
    L = np.hypot(X[i+1]-X[i], Z[i+1]-Z[i]) if i + 1 < len(X) else 0
    print(f"  node {i:3d} R0 {x[i]*1e3:7.3f}  R {X[i]*1e3:7.4f}  Z-tip {(Z[i]-tip)*1e3:8.4f}  phi {phi(X[i], Z[i])*1e3:8.4f}  edge {L*1e3:.4f}")
XX = x[:, None] + u[..., 0]; ZZ = y[None, :] + u[..., 1]
ph = phi(XX, ZZ)
ins = np.argwhere(ph[:, :-1] < 0)
print("  sub-surface nodes inside:", len(ins), [(int(i), int(j), round(float(x[i]*1e3),3), round(float((top-y[j])*1e3),3)) for i, j in ins[:8]])
