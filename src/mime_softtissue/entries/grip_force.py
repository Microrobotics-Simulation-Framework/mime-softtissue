"""``softtissue.grip_force``: an observation of the specimen's grip force, displacement and crack length each step,
and the peak force at the end."""

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

from ._vocabulary import CRACK_LENGTH, GRIP_DISPLACEMENT, GRIP_FORCE, VERSION


class GripForceState(TypedDict, closed=True):
    force: ReadOnly[Annotated[Float[Array, ""], Units("N")]]
    displacement: ReadOnly[Annotated[Float[Array, ""], Units("m")]]
    crack_length: ReadOnly[Annotated[Float[Array, ""], Units("m")]]
    peak_force: ReadOnly[Annotated[Float[Array, ""], Units("N")]]


class GripForceInputs(TypedDict, closed=True):
    force: NotRequired[ReadOnly[Float[Array, ""]]]
    displacement: NotRequired[ReadOnly[Float[Array, ""]]]
    crack_length: NotRequired[ReadOnly[Float[Array, ""]]]


class GripForceNode(MimeNode):
    """The grip's force and displacement, the crack length, and the largest force so far."""

    @override
    def initial_state(self) -> NodeFields:
        z = jnp.zeros(())
        return node_fields(GripForceState(force=z, displacement=z, crack_length=z, peak_force=z))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        dtype = jnp.zeros(()).dtype
        spec = {"force": "N", "displacement": "m", "crack_length": "m"}
        return {k: BoundaryInputSpec(shape=(), dtype=dtype, default=jnp.zeros((), dtype=dtype), expected_units=u)
                for k, u in spec.items()}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        last, given = typed_view(GripForceState, state), typed_view(GripForceInputs, boundary_inputs)
        z = jnp.zeros((), dtype=last["force"].dtype)
        force = given.get("force", z)
        return node_fields(GripForceState(force=force, displacement=given.get("displacement", z),
                                          crack_length=given.get("crack_length", z),
                                          peak_force=jnp.maximum(last["peak_force"], force)))


@register_observation
@dataclass(frozen=True, kw_only=True)
class GripForce(Observation):
    """The specimen's grip force against grip displacement and its crack length, each step; the peak force at the
    end."""

    name = "softtissue.grip_force"
    version = VERSION
    differentiability = GradientStatus.PIECEWISE_KINK
    gradient = GradientRecord(switches=(switch.Declaration(
        owner="softtissue.peak", indicator="the force against the largest so far, N", orders={"peak_force": 0},
        route="none needed: a running maximum is continuous, with a kink where the largest changes step"),))
    references = ()
    validity = ("the peak force is the largest of the steps' forces: a peak between two load steps is missed, and on a "
                "flat maximum the step it lands on moves with the load step (the strip's moved from 7.41 to 7.06 mm over "
                "three grids while the peak force moved 0.1 %)",)
    builds = {"": (GripForceNode,)}
    roles = {"specimen": RoleKind.BODY}
    reads = {"force": Read("specimen", GRIP_FORCE.name), "displacement": Read("specimen", GRIP_DISPLACEMENT.name),
             "crack_length": Read("specimen", CRACK_LENGTH.name)}
    outputs = (Output("force", FORCE, "the grip's force along the pull, each step"),
               Output("displacement", LENGTH, "the grip's displacement, each step"),
               Output("crack_length", LENGTH, "the phase field's crack length, each step"),
               Output("peak_force", FORCE, "the largest grip force so far", kind=OutputKind.AT_END))
    setting_roles = {}
    specimen: Any = None

    @override
    def build(self, ctx: BuildContext, b: ObservationBuilder) -> None:
        node = b.add_node(GripForceNode(ctx.node_name(), ctx.timestep))
        for label in ("force", "displacement", "crack_length"):
            b.connect(b.read(label), into=node[label])


