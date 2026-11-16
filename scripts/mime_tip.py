"""The barb-like tip as a MIME body (examples/tip_retrieval.py), one puncture-and-retrieval run through MIME:

    python scripts/mime_tip.py run --out FILE [--set-depth 0.6] [--end-depth -0.1] [--steps N]
        the MIME run, its force and depth each step, and the engine's own run replayed at the MIME run's depths
        (the same arithmetic outside MIME: P4.1)
    python scripts/mime_tip.py save --folder DIR [--steps N]     saved through MIME (then: mime verify DIR)
    python scripts/mime_tip.py gradient --out FILE [--rel 1e-2 1e-3 1e-4] [--plant]
        forward-mode Study.gradient of the retrieval force along tip.parts.tip.head_radius, and central differences
        of two MIME runs at each relative step; --plant: the head radius behind stop_gradient in the law's node"""
import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from tip_retrieval import DT, tip_experiment  # noqa: E402

from mime.units import mm, s  # noqa: E402

OBS = "softtissue.tip_force:clot"
ap = argparse.ArgumentParser()
ap.add_argument("what", choices=("run", "save", "gradient"))
ap.add_argument("--out")
ap.add_argument("--folder")
ap.add_argument("--set-depth", type=float, default=0.6)
ap.add_argument("--end-depth", type=float, default=-0.1)
ap.add_argument("--friction", type=float, default=0.3)
ap.add_argument("--head", type=float, default=0.6)
ap.add_argument("--h", type=float, default=0.05)
ap.add_argument("--steps", type=int, default=None, help="base steps (default: the whole insertion and withdrawal)")
ap.add_argument("--rel", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4])
ap.add_argument("--plant", action="store_true")
ap.add_argument("--objective", default="retrieval_force", choices=("retrieval_force", "withdrawal_work",
                                                                     "puncture_force"))
a = ap.parse_args()


def experiment(head_mm: float):
    exp, n = tip_experiment(head_radius_mm=head_mm, friction=a.friction, set_depth_mm=a.set_depth,
                            end_depth_mm=a.end_depth, h_mm=a.h)
    return exp, (a.steps or n)


def block_node(run) -> str:
    found = [k for k in run.final if "tip_pose" in run.final[k]]
    if len(found) != 1:
        raise KeyError(f"the block's node among {sorted(run.final)}")
    return found[0]


t0 = time.time()
if a.what == "run":
    exp, n = experiment(a.head)
    built = exp.build()
    run = built.run(s(DT * n), history=True)
    hist = run.history[OBS]
    force, depth = np.asarray(hist["force"]), np.asarray(hist["depth"])
    rec = dict(args=vars(a), steps=n, wall_s=time.time() - t0, force=force.tolist(), depth=depth.tolist(),
               puncture_force=float(np.asarray(run.final[OBS]["puncture_force"])),
               retrieval_force=float(np.asarray(run.final[OBS]["retrieval_force"])))
    # the engine's own run at the MIME run's depths: the block node's solvers, stepped outside MIME
    from mime_softtissue.dynamics.gel_block import BlockNode, unpack
    from mime_softtissue.dynamics import gel_block as gb
    bname = block_node(run)
    bhist = run.history[bname]
    rec["capped"] = np.asarray(bhist["capped"]).tolist()
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(f"MIME run: {n} steps in {time.time() - t0:.0f} s; puncture {rec['puncture_force'] * 1e3:.4f} mN, "
          f"retrieval {rec['retrieval_force'] * 1e3:.4f} mN", flush=True)
    # the replay: an engine Solver pair built as the node builds them, fed the depth the node used each step and the
    # same tip numbers (pose at the node's depth), outside MIME's graph
    import mime_softtissue.engine.solver as es
    from mime_softtissue.engine.material import Material
    from mime_softtissue._material_record import engine_material, read_material
    node = BlockNode("replay", DT, radius=5e-3, height=5e-3, h=a.h * 1e-3, fine=1.5e-3,
                     **{k: v for k, v in zip(("young", "poisson", "toughness", "length_scale", "stretch_limit",
                                               "residual"),
                                              (lambda r: (r["young_modulus"], r["poisson_ratio"], r["toughness"],
                                                          r["length_scale"], r["stretch_limit"],
                                                          r.get("residual_stiffness", 1e-6)))(
                                                  read_material("softtissue_gel_placeholder")))},
                     viscosity=0.1, origin=(0.0, 0.0, 0.0), shape="softtissue.headed_tip", mode="phase_field")
    ins, wd = node.solvers()
    step_in, step_out = jax.jit(ins.advance), jax.jit(wd.advance)
    p = {**node.constants()}
    mat = engine_material({k: jnp.asarray(v) for k, v in p.items() if isinstance(v, float)}, "phase_field")
    region = dict(neck_radius=jnp.asarray(0.4e-3), head_radius=jnp.asarray(a.head * 1e-3),
                  head_length=jnp.asarray(0.2e-3), fillet=jnp.asarray(0.05e-3))
    state = ins.initial()
    replay = []
    poses = np.asarray(bhist["tip_pose"])
    for k, dk in enumerate(depth):
        dk = jnp.asarray(float(dk))
        tip = dict(pose=jnp.asarray(poses[k]), centre=jnp.zeros(3), rotation=jnp.eye(3), region=region,
                   friction=jnp.asarray(a.friction), adhesion=jnp.asarray(0.0), stiffness=jnp.asarray(1e11),
                   depth_ref=dk)
        start = state._replace(delta=jnp.maximum(state.delta, 0.0))
        step = step_in if float(dk) >= float(start.delta) else step_out
        state, out = step(start, dk, mat, None, tip)
        state = state._replace(delta=dk)
        replay.append(float(out.force))
    replay = np.asarray(replay)
    rel = np.abs(replay - force) / np.maximum(np.max(np.abs(force)), 1e-30)
    rec.update(replay_force=replay.tolist(), replay_worst_rel=float(np.max(rel)),
               replay_bits_equal=bool(np.array_equal(replay, force)), wall_s=time.time() - t0)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(f"replay: worst relative {rec['replay_worst_rel']:.2e}, bits equal {rec['replay_bits_equal']}; "
          f"{time.time() - t0:.0f} s", flush=True)
elif a.what == "save":
    exp, n = experiment(a.head)
    exp.licence("CC0-1.0")
    exp.save(a.folder, steps=n)
    print("saved", a.folder, n, "steps", f"{time.time() - t0:.0f} s")
else:
    from mime.study import Study, along
    if a.plant:
        from mime_softtissue.interactions import tip_contact as tc
        real = tc.TipContactNode.region

        def cut(self, p, dtype):
            return [jax.lax.stop_gradient(v) for v in real(self, p, dtype)]
        tc.TipContactNode.region = cut
    exp, n = experiment(a.head)
    axis = "tip.parts.tip.head_radius"
    study = Study(exp).sweep(along({axis: [mm(a.head)]}), duration=s(DT * n))
    result = study.gradient(lambda final: final[OBS][a.objective], wrt=[axis], mode="forward")
    member = result.member(0)
    g = float(np.asarray(member.gradient[axis]).reshape(-1)[0])
    rec = dict(args=vars(a), steps=n, forward=g, status=str(getattr(member, "status", "")),
               flags=str(getattr(member, "flags", "")), wall_forward_s=time.time() - t0, central={})
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(f"forward d(retrieval)/d(head radius) = {g:.6e} N/m ({time.time() - t0:.0f} s)", flush=True)

    def retrieval(head_mm: float) -> float:
        e, m = experiment(head_mm)
        return float(np.asarray(e.build().run(s(DT * m), history=False).final[OBS][a.objective]))

    if not a.plant:
        for rel in a.rel:
            hi, lo = retrieval(a.head * (1 + rel)), retrieval(a.head * (1 - rel))
            c = (hi - lo) / (2 * rel * a.head * 1e-3)
            rec["central"][str(rel)] = dict(value=c, hi=hi, lo=lo, rel_diff=abs(g - c) / max(abs(c), 1e-30))
            Path(a.out).write_text(json.dumps(rec, indent=1))
            print(f"central rel {rel:g}: {c:.6e} N/m, relative difference {abs(g - c) / max(abs(c), 1e-30):.2e} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    print(json.dumps(rec, indent=1))
