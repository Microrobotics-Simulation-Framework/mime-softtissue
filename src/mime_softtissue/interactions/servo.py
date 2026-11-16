"""``softtissue.servo``: moves a rigid body down at a steady speed for a number of base steps, then up at it (an
insertion, then a withdrawal): LF-1's stand-in for a prescribed motion, a stiff resistance about the commanded velocity
(MIME's drives command an apparatus, not a body: gap G8)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, override

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float
from typing_extensions import ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Units, node_fields, typed_view
from mime.gradients import GradientStatus
from mime.registry import DESIGN, NUMERICAL, Delivery, Interaction, RoleKind, convention, register_interaction
from mime.units import Force, N, Quantity, Velocity, m_per_s
from mime.vocabulary import BACKGROUND_LOAD, RESISTANCE

from .._vocabulary import VERSION


class ServoState(TypedDict, closed=True):
    resistance: ReadOnly[Annotated[Float[Array, "6 6"], Units("N s/m; N s; N m s")]]
    background: ReadOnly[Annotated[Float[Array, "6"], Units("N; N m")]]
    steps: ReadOnly[Annotated[Float[Array, ""], Units("1")]]


class ServoNode(MimeNode):
    """A resistance ``gain`` about a commanded velocity along z: ``-speed`` for the first ``insert_steps`` steps, then
    ``+speed`` (the body integrates it implicitly); an integer step count as its clock."""

    def __init__(self, name: str, timestep: float, *, gain: float, turn_gain: float, speed: float,
                 insert_steps: int) -> None:
        super().__init__(name, timestep, gain=float(gain), turn_gain=float(turn_gain), speed=float(speed),
                         insert_steps=int(insert_steps))

    def made(self, p: Mapping[str, Any], steps: Array, dtype: Any) -> tuple[Array, Array]:
        sense = jnp.where(steps < self.constants()["insert_steps"], -1.0, 1.0).astype(dtype)
        gains = jnp.concatenate([jnp.full(3, p["gain"]), jnp.full(3, p["turn_gain"])]).astype(dtype)
        background = jnp.zeros(6, dtype=dtype).at[2].set(p["gain"] * p["speed"] * sense)
        return jnp.diag(gains), background

    @override
    def initial_state(self) -> NodeFields:
        dtype = jnp.zeros(()).dtype
        zero = jnp.zeros((), dtype=dtype)
        resistance, background = self.made(self.constants(), zero, dtype)
        return node_fields(ServoState(resistance=resistance, background=background, steps=zero))

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        last = typed_view(ServoState, state)
        steps = last["steps"] + 1.0
        resistance, background = self.made(p, steps, last["steps"].dtype)
        return node_fields(ServoState(resistance=resistance, background=background, steps=steps))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Servo(Interaction):
    """Moves a rigid body down at ``speed`` for ``insert_steps`` base steps, then up at it: a stiff resistance about
    that velocity (MIME's drives command an apparatus, not a body: the stand-in for a prescribed motion, gap G8)."""

    name = "softtissue.servo"
    version = VERSION
    unmetered = "a prescribed motion's stand-in"
    differentiability = GradientStatus.EXACT
    references = ()
    builds = {"": (ServoNode,)}
    roles = {"body": RoleKind.BODY}
    delivers = {"resistance": Delivery("body", RESISTANCE.name), "background": Delivery("body", BACKGROUND_LOAD.name)}
    setting_roles = {"gain": convention(NUMERICAL, "stiff against the contact: the velocity error is F/gain"),
                     "turn_gain": convention(NUMERICAL, "holds the tip's orientation"),
                     "speed": DESIGN, "insert_steps": DESIGN}
    speed: Annotated[Quantity, Velocity]
    insert_steps: int
    gain: Annotated[Quantity, Force / Velocity] = N(1e6) / m_per_s(1.0)
    turn_gain: float = 1.0

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        node = b.add_response_node(ServoNode(ctx.node_name(), ctx.timestep, gain=ctx.scalar("gain"),
                                             turn_gain=self.turn_gain, speed=ctx.scalar("speed"),
                                             insert_steps=self.insert_steps))
        b.connect(node["resistance"], into=b.target("resistance"))
        b.connect(node["background"], into=b.target("background"))
