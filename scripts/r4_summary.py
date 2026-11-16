"""Round 4's item-1 table from the retrieval records (recomputed from the rows, so a record cut short reads too).

    python scripts/r4_summary.py RUNS_DIR/ret [--json OUT]"""
import json
import sys
from pathlib import Path



def summary(rec: dict) -> dict:
    rows = rec["rows"]
    F = [r["force"] for r in rows]
    Fn = [r["force_normal"] for r in rows]
    D = [r["depth_mm"] * 1e-3 for r in rows]
    out = [r["phase"] == "out" for r in rows]
    n_in = sum(1 for o in out if not o)
    k_peak = max(range(n_in), key=lambda k: F[k]) if n_in else 0
    outs = [k for k in range(len(rows)) if out[k]]
    k_ret = max(outs, key=lambda k: -F[k]) if outs else None
    retrieval = max(-F[k_ret], 0.0) if k_ret is not None else None

    def work(n):
        return sum(0.5 * (F[k + 1] + F[k]) * (D[k + 1] - D[k]) for k in range(n - 1)) + 0.5 * F[0] * D[0]
    W, W_in = work(len(rows)), work(n_in)
    last = rows[-1]
    dissipated = sum(r["friction_work"] + r["viscous_work"] for r in rows)
    balance = W - (last["energy"] + last["contact_energy"] + last["crack_energy"] + dissipated)
    return dict(
        steps=len(rows), depth_end_mm=D[-1] * 1e3,
        puncture_force_mN=F[k_peak] * 1e3, puncture_depth_mm=D[k_peak] * 1e3,
        force_at_set_mN=F[n_in - 1] * 1e3 if n_in else None,
        falls_pct=100 * (1 - min(F[k_peak:n_in]) / F[k_peak]) if n_in else None,
        retrieval_force_mN=None if retrieval is None else retrieval * 1e3,
        retrieval_depth_mm=None if k_ret is None else D[k_ret] * 1e3,
        retrieval_shear_mN=None if k_ret is None else -(F[k_ret] - Fn[k_ret]) * 1e3,
        ratio=None if retrieval is None else retrieval / F[k_peak],
        least_force_out_mN=min(F[k] for k in outs) * 1e3 if outs else None,
        balance_pct_of_work_in=100 * balance / W_in if W_in else None,
        unconverged_in=sum(1 for r in rows if r["phase"] == "in" and not r["converged"]),
        unconverged_out=sum(1 for r in rows if r["phase"] == "out" and not r["converged"]),
        wall_s=sum(r["step_s"] for r in rows))


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    table = {}
    for p in sorted(folder.glob("*.json")):
        rec = json.loads(p.read_text())
        if rec.get("rows"):
            table[p.stem] = summary(rec)
    keys = ["steps", "puncture_force_mN", "puncture_depth_mm", "falls_pct", "force_at_set_mN", "retrieval_force_mN",
            "retrieval_depth_mm", "least_force_out_mN", "ratio", "balance_pct_of_work_in", "unconverged_in",
            "unconverged_out", "wall_s"]
    print("run".ljust(22) + " ".join(k[:12].rjust(12) for k in keys))
    for name, s in table.items():
        cells = []
        for k in keys:
            v = s[k]
            cells.append(("-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))).rjust(12))
        print(name.ljust(22) + " ".join(cells))
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(table, indent=1))
