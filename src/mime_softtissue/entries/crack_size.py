"""``softtissue.crack_size``: the crack's area each step (the phase field's surface energy over the toughness, the whole
block) and its largest value at the end."""

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
from mime.core import MimeNode, NodeFields, Units, node_fields, typed_view
from mime.gradients import GradientStatus
from mime.registry import BuildContext, GradientRecord, Observation, Output, OutputKind, Read, RoleKind, register_observation
from mime.units import LENGTH

from ._vocabulary import VERSION
from .gel_block import CRACK_AREA

AREA = LENGTH * LENGTH


class CrackState(TypedDict, closed=True):
    area: ReadOnly[Annotated[Float[Array, ""], Units("m^2")]]
    largest_area: ReadOnly[Annotated[Float[Array, ""], Units("m^2")]]


class CrackInputs(TypedDict, closed=True):
    area: NotRequired[ReadOnly[Float[Array, ""]]]


class CrackNode(MimeNode):
    @override
    def initial_state(self) -> NodeFields:
        z = jnp.zeros(())
        return node_fields(CrackState(area=z, largest_area=z))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        dtype = jnp.zeros(()).dtype
        return {"area": BoundaryInputSpec(shape=(), dtype=dtype, default=jnp.zeros((), dtype=dtype),
                                          expected_units="m^2")}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        last, given = typed_view(CrackState, state), typed_view(CrackInputs, boundary_inputs)
        area = given.get("area", jnp.zeros_like(last["area"]))
        return node_fields(CrackState(area=area, largest_area=jnp.maximum(last["largest_area"], area)))


@register_observation
@dataclass(frozen=True, kw_only=True)
class CrackSize(Observation):
    """The crack's area in the block (``solid``) each step, the phase field's surface energy over the toughness (the
    regularised crack's measure: a band of width about the length scale counts as its area), and the largest at the
    end."""

    name = "softtissue.crack_size"
    version = VERSION
    differentiability = GradientStatus.EXACT
    gradient = GradientRecord()
    references = ("Miehe2010",)
    validity = ("the phase field's regularised measure: over the grid's bias it counts a band about the length scale "
                "wide as the crack's area, and a partly damaged region counts in part",
                "the whole block's: in 3-D the quarter's surface energy times four, in axisymmetry the ring's")
    builds = {"": (CrackNode,)}
    roles = {"solid": RoleKind.BODY}
    reads = {"area": Read("solid", CRACK_AREA.name)}
    outputs = (Output("area", AREA, "the crack's area, each step"),
               Output("largest_area", AREA, "the largest crack area so far", kind=OutputKind.AT_END))
    setting_roles = {}
    solid: Any = None

    @override
    def build(self, ctx: BuildContext, b: ObservationBuilder) -> None:
        node = b.add_node(CrackNode(ctx.node_name(), ctx.timestep))
        b.connect(b.read("area"), into=node["area"])
