"""Round 4's figures from the run records (placeholder material throughout; every curve is the model's).

    python scripts/r4_figures.py RUNS_DIR FIGURES_DIR"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS, FIG = Path(sys.argv[1]), Path(sys.argv[2])
FIG.mkdir(parents=True, exist_ok=True)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # the reference palette's first four categorical slots
INK, MUTED = "#2b2b29", "#8a8980"
SHAPES = {"P": "blunt punch r 0.5 mm", "C": "cone 20 deg, r 0.5 mm", "B": "headed tip 0.6 / neck 0.4 mm"}
MUS = ["0", "0.1", "0.3", "0.6"]


def load(name):
    p = RUNS / "ret" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
for ax, (key, label) in zip(axes, SHAPES.items()):
    for colour, mu in zip(SERIES, MUS):
        r = load(f"{key}_mu{mu}")
        if r is None:
            continue
        rows = r["rows"]
        n_in = sum(1 for x in rows if x["phase"] == "in")
        d = np.asarray([x["depth_mm"] for x in rows])
        f = np.asarray([x["force"] for x in rows]) * 1e3
        bad = np.asarray([not x["converged"] for x in rows])
        ax.plot(d[:n_in], f[:n_in], color=colour, linewidth=2, label=f"mu {mu}")
        ax.plot(d[n_in - 1:], f[n_in - 1:], color=colour, linewidth=2, linestyle="--")
        if bad.any():
            ax.plot(d[bad], f[bad], linestyle="none", marker="x", markersize=8, color=INK)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_title(label, fontsize=10, color=INK)
    ax.set_xlabel("tip depth (mm); solid in, dashed out", fontsize=9, color=INK)
    style(ax)
axes[0].set_ylabel("force on the tip (mN; negative: the gel holds it)", fontsize=9, color=INK)
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Puncture to 0.6 mm and withdrawal, axisymmetric phase field, placeholder gel (x: a capped solve)",
             fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(FIG / "r4_fig1_retrieval_curves.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for colour, (key, label) in zip(SERIES, SHAPES.items()):
    xs, punct, ret = [], [], []
    for mu in MUS:
        r = load(f"{key}_mu{mu}")
        if r is None or "summary" not in r:
            continue
        xs.append(float(mu))
        punct.append(r["summary"]["puncture_force"] * 1e3)
        ret.append(max(r["summary"]["retrieval_force"] or 0.0, 0.0) * 1e3)
    axes[0].plot(xs, punct, color=colour, linewidth=2, marker="o", markersize=8, label=label)
    axes[1].plot(xs, ret, color=colour, linewidth=2, marker="o", markersize=8, label=label)
axes[0].set_ylabel("puncture force (mN)", fontsize=9, color=INK)
axes[1].set_ylabel("retrieval force, largest pull (mN)", fontsize=9, color=INK)
for ax in axes:
    ax.set_xlabel("friction coefficient (placeholder)", fontsize=9, color=INK)
    style(ax)
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Per shape against friction: placeholder gel, set depth 0.6 mm, h 0.05 mm", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(FIG / "r4_fig2_forces_vs_friction.png", dpi=150)

grid = sorted((RUNS / "grid").glob("*.json"))
if grid:
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for colour, tag, label in ((SERIES[0], "corr", "toughness corrected, Gc/(1 + h/2l)"),
                               (SERIES[1], "raw", "toughness as given")):
        pts = []
        for p in grid:
            if p.stem.endswith(tag):
                r = json.loads(p.read_text())
                if "summary" in r:
                    pts.append((r["args"]["h"], r["summary"]["puncture_force"] * 1e3))
        pts.sort()
        if pts:
            ax.plot(*zip(*pts), color=colour, linewidth=2, marker="o", markersize=8, label=label)
    ax.set_xlabel("element size h (mm); l = 0.1 mm", fontsize=9, color=INK)
    ax.set_ylabel("puncture peak (mN)", fontsize=9, color=INK)
    ax.set_title("Blunt punch, friction 0.3: the peak against the grid", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "r4_fig3_peak_grids.png", dpi=150)
print("figures in", FIG)
