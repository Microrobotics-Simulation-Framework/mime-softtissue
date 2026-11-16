"""The tip-as-a-body entries (round 4): the registry lists them; the lab kind refuses settings it cannot be; its shape
is registered and evaluated through MIME's registry; a short MIME run's force equals the engine's own at the depths the
run used (the plumbing adds nothing), on a coarse block (h = 0.1 mm, the checks are about the entries' contracts)."""
import math

import jax
import jax.numpy as jnp
import numpy as np

import mime_softtissue.entries as st
from mime.core import shape_named
from mime.testing import installed_entries
from mime.units import s
from tip_retrieval import DT, tip_experiment

OBS = "softtissue.tip_force:clot"


def test_the_registry_walk_lists_the_tip_entries():
    listed = {name for _, name, _ in installed_entries()}
    for name in ("softtissue.gel_block", "softtissue.tip_contact", "softtissue.servo", "softtissue.tip_force",
                 "softtissue.headed_tip"):
        assert name in listed, name


def test_the_kind_refuses_a_neck_wider_than_its_head():
    kind = st.HeadedTip.__new__(st.HeadedTip)
    found = kind.setting_problems({"neck_radius": 0.7e-3, "head_radius": 0.6e-3, "head_length": 0.2e-3,
                                   "fillet": 0.05e-3, "length": 3e-3})
    assert any("wider" in line for line in found), found
    assert kind.setting_problems({"neck_radius": 0.4e-3, "head_radius": 0.6e-3, "head_length": 0.2e-3,
                                  "fillet": 0.05e-3, "length": 3e-3}) == []


def test_the_registered_shape_is_the_engines_profile():
    shape = shape_named("softtissue.headed_tip")
    params = {"neck_radius": 0.4e-3, "head_radius": 0.6e-3, "head_length": 0.2e-3, "fillet": 0.05e-3}
    side = jnp.asarray([0.6e-3 * math.cos(1.0), 0.6e-3 * math.sin(1.0), 0.1e-3])  # the head's side, any azimuth
    assert abs(float(shape.signed_distance(params, side))) < 1e-15
    out = jnp.asarray([0.0, 0.7e-3, 0.1e-3])
    assert abs(float(shape.signed_distance(params, out)) - 0.1e-3) < 1e-15


def test_a_short_mime_run_equals_the_engine_at_its_depths():
    exp, _ = tip_experiment(set_depth_mm=0.06, end_depth_mm=0.02, h_mm=0.1)
    run = exp.build().run(s(DT * 5), history=True)
    force = np.asarray(run.history[OBS]["force"])
    (block,) = [k for k in run.final if "tip_pose" in run.final[k]]
    poses = np.asarray(run.history[block]["tip_pose"])
    depth = np.asarray(run.history[OBS]["depth"])
    assert np.all(np.diff(depth[:3]) > 0) and force[2] > 0, (depth, force)
    from mime_softtissue.entries.gel_block import BlockNode
    from mime_softtissue.entries._material_record import engine_material
    node = BlockNode("replay", DT, radius=5e-3, height=5e-3, h=1e-4, fine=1.5e-3, young=50e3, poisson=0.4,
                     toughness=1.0, length_scale=1e-4, stretch_limit=2.0, residual=1e-6, viscosity=0.1,
                     origin=(0.0, 0.0, 0.0), shape="softtissue.headed_tip", mode="phase_field")
    ins, wd = node.solvers()
    step_in, step_out = jax.jit(ins.advance), jax.jit(wd.advance)
    mat = engine_material({k: jnp.asarray(v) for k, v in node.constants().items() if isinstance(v, float)},
                          "phase_field")
    region = dict(neck_radius=jnp.asarray(0.4e-3), head_radius=jnp.asarray(0.6e-3), head_length=jnp.asarray(0.2e-3),
                  fillet=jnp.asarray(0.05e-3))
    state, replay = ins.initial(), []
    for k, dk in enumerate(depth):
        dk = jnp.asarray(float(dk))
        tip = dict(pose=jnp.asarray(poses[k]), centre=jnp.zeros(3), rotation=jnp.eye(3), region=region,
                   friction=jnp.asarray(0.3), adhesion=jnp.asarray(0.0), stiffness=jnp.asarray(1e11), depth_ref=dk)
        start = state._replace(delta=jnp.maximum(state.delta, 0.0))
        step = step_in if float(dk) >= float(start.delta) else step_out
        state, out = step(start, dk, mat, None, tip)
        state = state._replace(delta=dk)
        replay.append(float(out.force))
    np.testing.assert_allclose(replay, force, rtol=1e-9, atol=1e-12 * float(np.max(np.abs(force))))
