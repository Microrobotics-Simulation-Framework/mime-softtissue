"""MIME's node verification and parameter reach, on the strip's solid and its observation.

`node_failures` runs MADDENING's battery (finite, jit-consistent, differentiable, bounded) on every node the entry builds.
`param_reach_failures` writes a new value into each parameter the build declares and requires that the run changes.

Two gaps in this MIME that this plugin meets (the scaffold's note, held here): `NodeCase` cannot say a node is differentiable
in forward mode only, so the staggered Newton solve (`lax.while_loop`) fails three checks by design; and no hook marks a
constructor argument as structure, so parameter reach names four. Each failing set is held to exactly what it is, and a
check that starts to pass or a new one that fails is news.
"""

import jax.numpy as jnp
from mime.experiment import Experiment
from mime.testing import NodeCase, node_failures, param_reach_failures

import mime_softtissue.entries as st

NODE_STEPS = 5


def draw_state(state):
    """A state the solid can have: small displacements (strain well under the 2.8 stretch the gel fractures at), a
    damage in [0, 1], a history that is not negative, the grip and the notch where the build put them."""
    d, hist = jnp.clip(state["d"], 0.0, 1.0), jnp.abs(state["history"])
    return {**state, "u": jnp.clip(state["u"], -2e-5, 2e-5), "d": d, "history": jnp.clip(hist, 0.0, 1e3),
            "delta": jnp.clip(jnp.abs(state["delta"]), 0.0, 1e-3), "force": state["force"],
            "crack_length": jnp.abs(state["crack_length"]), "notch": jnp.full_like(state["notch"], 10.5e-3)}


NODE_CASE = NodeCase(options={"constrain_state": draw_state,
                              "bounds": {"u": (-2e-5, 2e-5), "d": (0.0, 1.0), "history": (0.0, 1e3),
                                         "delta": (0.0, 1e-3), "force": (-1.0, 1.0), "crack_length": (0.0, 1e-2),
                                         "notch": (10.4e-3, 10.6e-3)}})
GRIP_CASE = NodeCase(options={"boundary_bounds": {"force": (-1.0, 1.0), "displacement": (0.0, 1e-2),
                                                  "crack_length": (0.0, 1e-2)}})

#: MADDENING's battery differentiates a node's step in reverse mode (jax.grad); the solid's step is a staggered Newton solve
#: in ``lax.while_loop``, which JAX differentiates in forward mode only (``jax.jvp``: what ``Study.gradient`` uses). The error
#: is JAX's "Reverse-mode differentiation does not work for lax.while_loop" [measured].
FORWARD_ONLY = {"gradient_finite", "params_gradient_finite", "params_effective"}

#: Parameter reach finds four leaves the compiled step does not read [measured]: ``width``, ``half_height`` and ``fine`` fix
#: the grid's topology when the solver is built (structure, not parameters), and ``stretch_limit`` is read only by the
#: element-deletion control, not by ``phase_field``.
UNREAD = {"fine", "half_height", "stretch_limit", "width"}


def test_node_verification_of_the_solid(experiment: Experiment):
    failed = {line.split(": ")[1].split(":")[0].split()[-1]
              for line in node_failures(experiment.build(), st.NotchedStrip, NODE_CASE) if ": FAIL" in line.splitlines()[0]}
    assert failed == FORWARD_ONLY, failed


def test_node_verification_of_the_observation(experiment: Experiment):
    assert node_failures(experiment.build(), st.GripForce, GRIP_CASE) == []


def test_parameter_reach(experiment: Experiment):
    found = {line.split("['")[3].split("']")[0] for line in param_reach_failures(experiment.build(), steps=NODE_STEPS)}
    assert found == UNREAD, found
