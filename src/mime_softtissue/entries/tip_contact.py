"""``softtissue.tip_contact``: the one law joining a MIME rigid body's tip to the gel block (``softtissue.gel_block``),
and ``softtissue.servo``, LF-1's stand-in for a prescribed motion, here an insertion then a withdrawal.

The law reads the tip's shape through ``ctx.geometry(of="tip")`` (its kind's one region, the registered shape and its
parameters, which are this law's node's leaves, so a study moves them), the tip's pose, and the block's force on the
tip; it hands the block the tip (pose, part frame, shape parameters, friction, adhesive shear, penalty stiffness) and
the tip the block's force as an applied wrench, equal and opposite to what the block's solve put on its nodes. The
block solves the contact inside its own equilibrium (an implicit contact cannot be a load computed from states)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, override

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float
from maddening.core.node import BoundaryInputSpec
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.build import InteractionBuilder
from mime.core import MimeNode, NodeFields, Units, node_fields, shape_named, typed_view
from mime.gradients import GradientStatus
from mime.registry import (DESIGN, NUMERICAL, PHYSICAL, BuildContext, Delivery, GradientRecord, Interaction, Read,
                           RoleKind, convention, register_interaction)
from mime.units import Force, Magnitude, N, Quantity, Velocity, m_per_s
from mime.vocabulary import APPLIED_WRENCH, BACKGROUND_LOAD, POSE, RESISTANCE

from ._vocabulary import VERSION
from .gel_block import INDENTER, N_REGION, TIP_FORCE


def _scalar(parameters: Mapping[str, Magnitude], name: str) -> float:
    value = parameters[name]
    if isinstance(value, tuple):
        raise TypeError(f"parameter {name!r} is one number, not {value!r}")
    return float(value)


class ContactState(TypedDict, closed=True):
    indenter: ReadOnly[Annotated[Float[Array, " n"], Units("1")]]
    wrench: ReadOnly[Annotated[Float[Array, "6"], Units("N; N m")]]


class ContactInputs(TypedDict, closed=True):
    pose: NotRequired[ReadOnly[Float[Array, "7"]]]
    force: NotRequired[ReadOnly[Float[Array, ""]]]


class TipContactNode(MimeNode):
    """The tip handed to the block as one vector, and the block's force handed to the tip as a wrench along +z."""

    def __init__(self, name: str, timestep: float, *, shape: str, hull_centre: Sequence[float],
                 hull_rotation: Sequence[float], friction: float, adhesion: float, stiffness: float,
                 **region: float) -> None:
        super().__init__(name, timestep, shape=str(shape), hull_centre=tuple(float(c) for c in hull_centre),
                         hull_rotation=tuple(float(c) for c in hull_rotation), friction=float(friction),
                         adhesion=float(adhesion), stiffness=float(stiffness),
                         **{f"region_{k}": float(v) for k, v in region.items()})

    def region(self, p: Mapping[str, Any], dtype: Any) -> list[Array]:
        names = shape_named(self.constants()["shape"]).parameter_names
        return [jnp.asarray(p[f"region_{n}"], dtype=dtype) for n in names]

    def made(self, p: Mapping[str, Any], pose: Array, force: Array) -> tuple[Array, Array]:
        dtype = pose.dtype
        region = self.region(p, dtype)
        region += [jnp.zeros((), dtype=dtype)] * (N_REGION - len(region))  # a shape of fewer parameters, padded
        vector = jnp.concatenate([pose, jnp.asarray(p["hull_centre"], dtype=dtype),
                                  jnp.asarray(p["hull_rotation"], dtype=dtype), jnp.stack(region),
                                  jnp.stack([jnp.asarray(p["friction"], dtype=dtype),
                                             jnp.asarray(p["adhesion"], dtype=dtype),
                                             jnp.asarray(p["stiffness"], dtype=dtype)])])
        wrench = jnp.zeros(6, dtype=dtype).at[2].set(force)
        return vector, wrench

    @override
    def initial_state(self) -> NodeFields:
        dtype = jnp.zeros(()).dtype
        vector, wrench = self.made(self.constants(), jnp.asarray((0, 0, 1e3, 1, 0, 0, 0), dtype=dtype),
                                   jnp.zeros((), dtype=dtype))
        return node_fields(ContactState(indenter=vector, wrench=wrench))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        dtype = jnp.zeros(()).dtype
        return {"pose": BoundaryInputSpec(shape=(7,), dtype=dtype, default=jnp.asarray((0, 0, 1e3, 1, 0, 0, 0),
                                                                                         dtype=dtype),
                                          expected_units="m; 1"),
                "force": BoundaryInputSpec(shape=(), dtype=dtype, default=jnp.zeros((), dtype=dtype),
                                           expected_units="N")}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        given = typed_view(ContactInputs, boundary_inputs)
        dtype = typed_view(ContactState, state)["wrench"].dtype
        pose = given.get("pose", jnp.asarray((0, 0, 1e3, 1, 0, 0, 0), dtype=dtype))
        vector, wrench = self.made(p, pose, given.get("force", jnp.zeros((), dtype=dtype)))
        return node_fields(ContactState(indenter=vector, wrench=wrench))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class TipContact(Interaction):
    """A rigid tip (``tip``, a body whose hull's kind has one region with a registered shape) against the gel block
    (``solid``): the penalty ``stiffness``, regularised Coulomb ``friction`` and an adhesive shear ``adhesion`` (a
    placeholder interface law, tau = tau_a + mu p) solved inside the block's equilibrium; the block's force back on the
    tip."""

    name = "softtissue.tip_contact"
    version = VERSION
    unmetered = "a penalty and friction the block's own account books; MIME's ledger has no pair kind for it"
    differentiability = GradientStatus.PIECEWISE_KINK
    gradient = GradientRecord()
    references = ("FregoneseBacca2024",)
    builds = {"": (TipContactNode,)}
    roles = {"tip": RoleKind.BODY, "solid": RoleKind.BODY}
    reads = {"pose": Read("tip", POSE.name), "force": Read("solid", TIP_FORCE.name)}
    delivers = {"indenter": Delivery("solid", INDENTER.name), "wrench": Delivery("tip", APPLIED_WRENCH.name)}
    setting_roles = {"friction": PHYSICAL, "adhesion": PHYSICAL,
                     "stiffness": convention(NUMERICAL, "100 E/h for the placeholder gel at h = 0.05 mm (round 3's)")}
    friction: float
    #: The adhesive shear, Pa, a plain number.
    adhesion: float
    #: The penalty's stiffness per unit area, Pa/m, a plain number.
    stiffness: float = 1e11

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        hull = ctx.geometry(of="tip")
        assert hull is not None, "the tip has a hull"
        kind = hull.kind
        (region_name,) = kind.regions
        shape = kind.region_shapes[region_name]
        region = kind.region_parameters(region_name, hull.settings)
        names = shape_named(shape).parameter_names
        if len(names) > N_REGION:
            raise ValueError(f"the tip's shape {shape!r} has {len(names)} parameters; the block's interface holds "
                             f"{N_REGION} (an interface's shape is fixed when it is defined)")
        node = b.add_response_node(TipContactNode(
            ctx.node_name(), ctx.timestep, shape=shape, hull_centre=hull.centre,
            hull_rotation=tuple(c for row in hull.rotation for c in row), friction=self.friction,
            adhesion=self.adhesion, stiffness=self.stiffness, **{n: _scalar(region, n) for n in names}))
        b.connect(b.read("pose"), into=node["pose"])
        b.connect(b.read("force"), into=node["force"])
        b.connect(node["indenter"], into=b.target("indenter"))
        b.connect(node["wrench"], into=b.target("wrench"))


# --- the servo: insertion to a depth, then withdrawal (LF-1's stand-in for a prescribed motion) ----------------------

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
