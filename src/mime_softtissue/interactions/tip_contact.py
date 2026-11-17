"""``softtissue.tip_contact``: the one law joining a MIME rigid body's tip to the gel block (``softtissue.gel_block``).

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
from mime.registry import (NUMERICAL, PHYSICAL, BuildContext, Delivery, GradientRecord, Interaction, PlanContext,
                           Problem, Read, RoleKind, convention, register_interaction)
from mime.units import Magnitude
from mime.vocabulary import APPLIED_WRENCH, POSE

from .._vocabulary import INDENTER, N_REGION, TIP_FORCE, VERSION


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
    def value_problems(self, values: Mapping[str, Magnitude], ctx: PlanContext) -> list[Problem]:
        """The tip's hull must be one region whose registered shape has at most ``N_REGION`` parameters (the block's
        interface is one vector, its length fixed when it is defined): a tip that does not fit is refused at the plan,
        with what to use instead, never as a failure of the build."""
        hull = ctx.hulls.get("tip")
        if hull is None:
            return []
        regions = hull.kind.regions
        if len(regions) != 1:
            return [Problem("softtissue.tip-regions", f"the tip's hull ({hull.label()}) has {len(regions)} regions; "
                            f"the contact presses one", "give the tip one part of one region: a tip kind of "
                            "mime_softtissue.scene (ConeTip, PunchTip, HeadedTip)")]
        shape = hull.kind.region_shapes[regions[0]]
        count = len(shape_named(shape).parameter_names)
        if count > N_REGION:
            return [Problem("softtissue.tip-shape", f"the tip's shape {shape!r} has {count} parameters; the block's "
                            f"interface holds {N_REGION} (an interface's shape is fixed when it is defined)",
                            "use a tip kind of mime_softtissue.scene (ConeTip, PunchTip, HeadedTip), or a shape of "
                            f"at most {N_REGION} parameters")]
        return []

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        hull = ctx.geometry(of="tip")
        assert hull is not None, "the tip has a hull"
        kind = hull.kind
        (region_name,) = kind.regions
        shape = kind.region_shapes[region_name]
        region = kind.region_parameters(region_name, hull.settings)
        names = shape_named(shape).parameter_names  # at most N_REGION: ``value_problems`` refused more at the plan
        node = b.add_response_node(TipContactNode(
            ctx.node_name(), ctx.timestep, shape=shape, hull_centre=hull.centre,
            hull_rotation=tuple(c for row in hull.rotation for c in row), friction=self.friction,
            adhesion=self.adhesion, stiffness=self.stiffness, **{n: _scalar(region, n) for n in names}))
        b.connect(b.read("pose"), into=node["pose"])
        b.connect(b.read("force"), into=node["force"])
        b.connect(node["indenter"], into=b.target("indenter"))
        b.connect(node["wrench"], into=b.target("wrench"))
