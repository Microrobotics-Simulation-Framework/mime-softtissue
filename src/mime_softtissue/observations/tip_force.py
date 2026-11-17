"""``softtissue.tip_force``: the force on the tip against its depth, each step, and at the end the largest push
(the puncture force) and the largest pull (the retrieval force). The force is read from the block's published value:
a rigid body publishes no applied wrench (LF-1's seam 4a, met again), so this holds while the block is the tip's only
loader besides its servo. The largest values are running maxima in the node's state (no max reduction; seam 4b)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, override

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float
from maddening.core.node import BoundaryInputSpec
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.build import ObservationBuilder
from mime.core import MimeNode, NodeFields, Units, node_fields, switch, typed_view
from mime.gradients import GradientStatus
from mime.registry import BuildContext, GradientRecord, Observation, Output, OutputKind, Read, RoleKind, register_observation
from mime.units import FORCE, LENGTH

from .._vocabulary import TIP_DEPTH, TIP_FORCE, VERSION

ENERGY = FORCE * LENGTH


class TipForceState(TypedDict, closed=True):
    force: ReadOnly[Annotated[Float[Array, ""], Units("N")]]
    depth: ReadOnly[Annotated[Float[Array, ""], Units("m")]]
    puncture_force: ReadOnly[Annotated[Float[Array, ""], Units("N")]]
    retrieval_force: ReadOnly[Annotated[Float[Array, ""], Units("N")]]
    withdrawal_work: ReadOnly[Annotated[Float[Array, ""], Units("J")]]


class TipForceInputs(TypedDict, closed=True):
    force: NotRequired[ReadOnly[Float[Array, ""]]]
    depth: NotRequired[ReadOnly[Float[Array, ""]]]


class TipForceNode(MimeNode):
    @override
    def initial_state(self) -> NodeFields:
        z = jnp.zeros(())
        return node_fields(TipForceState(force=z, depth=z, puncture_force=z, retrieval_force=z, withdrawal_work=z))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        dtype = jnp.zeros(()).dtype
        z = jnp.zeros((), dtype=dtype)
        return {"force": BoundaryInputSpec(shape=(), dtype=dtype, default=z, expected_units="N"),
                "depth": BoundaryInputSpec(shape=(), dtype=dtype, default=z, expected_units="m")}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        last, given = typed_view(TipForceState, state), typed_view(TipForceInputs, boundary_inputs)
        force = given.get("force", jnp.zeros_like(last["force"]))
        depth = given.get("depth", jnp.zeros_like(last["depth"]))
        rise = jnp.maximum(last["depth"] - depth, 0.0)  # how far the tip came out this step
        return node_fields(TipForceState(force=force, depth=depth,
                                         puncture_force=jnp.maximum(last["puncture_force"], force),
                                         retrieval_force=jnp.maximum(last["retrieval_force"], -force),
                                         withdrawal_work=last["withdrawal_work"]
                                         + 0.5 * (force + last["force"]) * rise))


PEAKS = switch.Declaration(
    owner="softtissue.tip_force.peaks", indicator="the force against the largest push or pull so far, N",
    orders={"puncture_force": 0, "retrieval_force": 0},
    route="none needed: a running maximum is continuous, with a kink where the step that holds it changes")


@register_observation
@dataclass(frozen=True, kw_only=True)
class TipForce(Observation):
    """The force the block puts on the tip (positive pushing it out) and the tip's depth, each step; the largest push
    and the largest pull at the end."""

    name = "softtissue.tip_force"
    version = VERSION
    differentiability = GradientStatus.PIECEWISE_KINK
    gradient = GradientRecord(switches=(PEAKS,))
    references = ()
    builds = {"": (TipForceNode,)}
    roles = {"solid": RoleKind.BODY}
    reads = {"force": Read("solid", TIP_FORCE.name), "depth": Read("solid", TIP_DEPTH.name)}
    outputs = (Output("force", FORCE, "the block's force on the tip along +z, each step"),
               Output("depth", LENGTH, "the tip's lowest point below the block's top face, each step"),
               Output("puncture_force", FORCE, "the largest push so far", kind=OutputKind.AT_END),
               Output("retrieval_force", FORCE, "the largest pull so far", kind=OutputKind.AT_END),
               Output("withdrawal_work", ENERGY, "the work the gel did on the tip as it came out (positive: it "
                      "pushed the tip out; a pull counts against it)", kind=OutputKind.AT_END))
    setting_roles = {}
    solid: Any = None

    @override
    def build(self, ctx: BuildContext, b: ObservationBuilder) -> None:
        node = b.add_node(TipForceNode(ctx.node_name(), ctx.timestep))
        b.connect(b.read("force"), into=node["force"])
        b.connect(b.read("depth"), into=node["depth"])
