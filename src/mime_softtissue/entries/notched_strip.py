"""``softtissue.notched_strip``: a notched plane-strain specimen as one body's dynamics, loaded quasi-statically at a
grip rate on the base step's clock (one load step a base step). The notch is the body's initial state (the initial
crack), so a study can take the derivative along it (``strip.initial.notch``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, override

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float
from typing_extensions import ReadOnly, TypedDict

from mime.build import DynamicsBuilder, ResolvedRegistration
from mime.core import MimeNode, NodeFields, Units, node_fields, switch, typed_view
from mime.gradients import GradientStatus
from mime.registry import (DESIGN, NUMERICAL, PHYSICAL, BuildContext, Dynamics, GradientRecord, Port, convention,
                           register_dynamics)
from mime.things import ThingKind
from mime.units import LENGTH, Length, Quantity, Velocity, mm

from ..engine.solver import Solver, State
from ..engine.specimen import edge_notched_strip
from ._material_record import engine_material, read_material
from ._vocabulary import CRACK_LENGTH, GRIP_DISPLACEMENT, GRIP_FORCE, VERSION

DAMAGE_GROWTH = switch.Declaration(
    owner="softtissue.damage_growth", indicator="psi_plus against its history at a Gauss point (J/m^3)",
    orders={"": 0}, route="none needed: the history field's maximum is continuous, with a kink where it starts growing")

UNMETERED = "the specimen keeps its own energy account (stored, crack surface), outside MIME's ledger"


class SpecimenState(TypedDict, closed=True):
    u: ReadOnly[Annotated[Float[Array, "nx ny 2"], Units("m")]]
    d: ReadOnly[Annotated[Float[Array, "nx ny"], Units("1")]]
    history: ReadOnly[Annotated[Float[Array, "e 4"], Units("J/m^3")]]
    delta: ReadOnly[Annotated[Float[Array, ""], Units("m")]]
    force: ReadOnly[Annotated[Float[Array, ""], Units("N")]]
    crack_length: ReadOnly[Annotated[Float[Array, ""], Units("m")]]
    notch: ReadOnly[Annotated[Float[Array, ""], Units("m")]]


class SpecimenNode(MimeNode):
    """The specimen's displacement, damage and history fields; each step the grip moves ``rate`` times the step and
    the staggered solve finds the new equilibrium."""

    def __init__(self, name: str, timestep: float, *, width: float, half_height: float, thickness: float,
                 initial_notch: float, h: float, mode: str, young: float, poisson: float, toughness: float,
                 length_scale: float, stretch_limit: float, residual: float, rate: float, fine: float = 2.4e-3) -> None:
        super().__init__(name, timestep, width=float(width), half_height=float(half_height),
                         thickness=float(thickness), initial_notch=float(initial_notch), h=float(h), mode=str(mode),
                         young=float(young), poisson=float(poisson), toughness=float(toughness),
                         length_scale=float(length_scale), stretch_limit=float(stretch_limit),
                         residual=float(residual), rate=float(rate), fine=float(fine))
        self._solver: Solver | None = None

    def solver(self) -> Solver:
        if self._solver is None:
            own = self.constants()
            self._solver = Solver(edge_notched_strip(width=own["width"], half_height=own["half_height"],
                                                     thickness=own["thickness"], notch=own["initial_notch"],
                                                     h=own["h"], fine=own["fine"]), own["mode"])
        return self._solver

    @override
    def initial_state(self) -> NodeFields:
        s = self.solver().initial(jnp.zeros(()).dtype)
        z = jnp.zeros((), dtype=s.u.dtype)
        notch = jnp.asarray(self.constants()["initial_notch"], dtype=s.u.dtype)
        return node_fields(SpecimenState(u=s.u, d=s.d, history=s.history, delta=z, force=z, crack_length=z,
                                         notch=notch))

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        last = typed_view(SpecimenState, state)
        mat = engine_material(p, self.constants()["mode"])
        delta = last["delta"] + p["rate"] * dt
        new, out = self.solver().advance(State(last["u"], last["d"], last["history"], last["delta"]), delta, mat,
                                         last["notch"])
        crack = out.crack_energy / (mat.gc * p["thickness"])
        return node_fields(SpecimenState(u=new.u, d=new.d, history=new.history, delta=new.delta, force=out.force,
                                         crack_length=crack, notch=last["notch"]))


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class NotchedStrip(Dynamics):
    """A single-edge-notched strip in plane strain, pulled by clamped grips at ``grip_rate``: the upper half by
    symmetry, ``width`` across, ``height`` grip to grip, its initial state's ``notch`` from one edge (within a quarter
    cell of a node of the grid; the tip placed exactly), on a grid of spacing ``h`` along the crack plane; the material
    from the asset ``material``; fracture by the AT2 phase field (``mode="phase_field"``) or, as a control, element
    deletion at a stretch limit with no length scale (``mode="local"``)."""

    name = "softtissue.notched_strip"
    version = VERSION
    unmetered = UNMETERED
    differentiability = GradientStatus.PIECEWISE_KINK
    gradient = GradientRecord(switches=(DAMAGE_GROWTH,))
    references = ("Tutwiler2020", "Miehe2010")
    validity = (
        "plane strain only: the strip is a 2-D section (an assumed thickness scales the force); a thin sheet is "
        "nearer plane stress, which this does not model",
        "quasi-static only: each base step is a static equilibrium at the grip's displacement; no inertia, no "
        "poroelasticity, no rate dependence",
        "the grid's element size h at most half the phase-field length scale l (the toughness is corrected for the "
        "grid's bias, which is a convergence aid, not a licence to coarsen); the implied toughness at h = l is "
        "above the input by 5 to 10 % [measured, the prototype's three grids]",
        "the peak force is not converged under refinement (it moves 0.1 % a halving) and the staggered solve may stop "
        "at its iteration cap after the peak; forces past the peak are an estimate",
    )
    builds = {"": (SpecimenNode,)}
    applies_to = frozenset({ThingKind.BODY})
    publishes = {GRIP_FORCE.name: Port("force"), GRIP_DISPLACEMENT.name: Port("delta"),
                 CRACK_LENGTH.name: Port("crack_length")}
    accepts = {}
    initial_state = {"position": LENGTH, "notch": LENGTH}
    setting_roles = {"width": DESIGN, "height": DESIGN, "thickness": DESIGN, "grip_rate": DESIGN,
                     "material": DESIGN,
                     "h": convention(NUMERICAL, "the grid along the crack plane, at most half the length scale"),
                     "mode": convention(PHYSICAL, "phase_field is the model; local is the control without a length")}
    width: Annotated[Quantity, Length]
    height: Annotated[Quantity, Length]
    thickness: Annotated[Quantity, Length]
    grip_rate: Annotated[Quantity, Velocity]
    material: str
    h: Annotated[Quantity, Length] = mm(0.3)
    mode: str = "phase_field"

    @override
    def assets_read(self) -> tuple[str, ...]:
        return (self.material,)

    @override
    def initial_problems(self, initial: Mapping[str, Any]) -> list[str]:
        notch = _number(initial["notch"]) if "notch" in initial else None
        width = self.width.value if isinstance(self.width.value, (int, float)) else None
        if notch is None or width is None:
            return []
        return [f"the notch {notch} m is not inside the strip's width {width} m"] if not 0.0 < notch < width else []

    @override
    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        values = read_material(self.material)
        b.add_node(SpecimenNode(
            ctx.node_name(), ctx.timestep, width=ctx.scalar("width"), half_height=ctx.scalar("height") / 2,
            thickness=ctx.scalar("thickness"), initial_notch=_number(ctx.initial["notch"]), h=ctx.scalar("h"),
            mode=self.mode,
            young=values["young_modulus"], poisson=values["poisson_ratio"], toughness=values["toughness"],
            length_scale=values["length_scale"], stretch_limit=values["stretch_limit"],
            residual=values.get("residual_stiffness", 1e-6), rate=ctx.scalar("grip_rate")))


def _number(value: Any) -> float:
    """One number in SI from an initial value as the build hands it (a number, a one-element tuple, a quantity)."""
    held = getattr(value, "value", value)
    return float(np.asarray(held, dtype=float).reshape(-1)[0])
