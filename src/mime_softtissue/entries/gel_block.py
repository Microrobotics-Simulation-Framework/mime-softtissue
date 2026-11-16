"""``softtissue.gel_block``: a block of gel or clot as one body's dynamics, axisymmetric or 3-D (its ``dimension``),
pressed and pulled by a rigid tip whose pose, shape and interface numbers a law hands it each step
(``softtissue.tip_contact``, interface ``softtissue.block.indenter``). Each base step is a quasi-static equilibrium at the tip's pose: the contact's penalty,
friction and adhesion sit inside the block's own solve (an implicit contact has no home in a law, which delivers a
load computed from states; the law hands the block the tip instead). The tip's shape is the registered shape its
geometry kind names, evaluated through ``mime.core.shape_named`` at each node (no formula here).

The block's axis is the world z through its body's initial ``position`` (the centre of its bottom face); the tip must
be coaxial (the build cannot see it, so it is a validity line). In 3-D the block is square, ``2 radius`` across, and a
quarter of it is solved (x >= 0, y >= 0 in the block's frame, symmetric about both planes through the axis), with spike
3D-COST's ``all2`` solver (``engine.solid3d``); its surface shows that quarter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, override

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float
from maddening.core.node import BoundaryInputSpec
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.build import DynamicsBuilder, ResolvedRegistration
from mime.core import MimeNode, NodeFields, Units, node_fields, quaternion, shape_named, switch, typed_view
from mime.gradients import GradientStatus
from mime.registry import (DESIGN, NUMERICAL, PHYSICAL, BuildContext, Dynamics, GradientRecord, MeshField, Port,
                           SurfaceMesh, convention, register_dynamics)
from mime.things import ThingKind
from mime.units import LENGTH, Length, Magnitude, Quantity, mm
from mime.vocabulary import Interface, ReadKind, TargetKind, define

from ..engine import contact as ct
from ..engine.kinematics.axisymmetric import nodal_ring_area
from ..engine.solid3d import block as b3
from ..engine.solver import Solver, State
from ..engine.specimen import axisymmetric_block
from ._display import placed, quarter_boundary, revolved
from ._material_record import engine_material, read_material
from ._vocabulary import VERSION

#: The tip's numbers the law hands the block: pose (7), the part's centre (3) and rotation (9, rows) in the body's
#: frame, its region's parameters (``N_REGION``, in the shape's ``parameter_names`` order), friction, adhesive shear
#: (Pa) and the penalty's stiffness (Pa/m). An interface's shape is fixed at definition, so ``N_REGION`` is the one
#: registered tip shape's count (LF-1's seam 1c, met again).
N_REGION = 4
N_INDENTER = 7 + 3 + 9 + N_REGION + 3
INDENTER = define(Interface("softtissue.block.indenter", TargetKind.EQUALITY_CONSTRAINT, "1", "world", (N_INDENTER,),
                            magnitude=1e-3, description="the tip pressing the block: its pose, part frame, shape "
                                                        "parameters and interface numbers, as one vector"))
TIP_FORCE = define(Interface("softtissue.block.tip_force", ReadKind.STATE, "N", None, (), magnitude=1e-2,
                             description="the force the block puts on the tip along +z (positive: pushing it out)"))
TIP_DEPTH = define(Interface("softtissue.block.tip_depth", ReadKind.STATE, "m", None, (), magnitude=1e-3,
                             description="the tip's lowest point below the block's undeformed top face"))
CRACK_AREA = define(Interface("softtissue.block.crack_area", ReadKind.STATE, "m^2", None, (), magnitude=1e-6,
                              description="the crack's area: the phase field's surface energy over the toughness"))
DIMENSIONS = ("axisymmetric", "3d")
#: In 3-D the uniform region reaches this share of ``fine`` from the axis (spike 3D-COST's G1 grid: 0.8 mm across, 1.0
#: mm deep), and grades by 1.5 up to 16 h beyond.
FINE_XY = 0.8
#: Where the tip waits when no law has handed it yet: far above the block.
AWAY = (0.0, 0.0, 1e3, 1.0, 0.0, 0.0, 0.0)

DAMAGE_GROWTH = switch.Declaration(
    owner="softtissue.gel_block.damage_growth", indicator="psi_plus against its history at a Gauss point (J/m^3)",
    orders={"": 0}, route="none needed: the history field's maximum is continuous, with a kink where it starts growing")
CONTACT_ONSET = switch.Declaration(
    owner="softtissue.gel_block.contact_onset", indicator="a node's signed distance from the tip (m)",
    orders={"": 0}, route="none needed: the penalty is continuous with a kink in its force where a node enters")
PREDICTOR = switch.Declaration(
    owner="softtissue.gel_block.predictor", indicator="the tip's depth against the last step's (m)",
    orders={"": 0}, route="none needed: both branches solve the same equilibrium from different first guesses")


@dataclass(frozen=True)
class ShapeTip:
    """The tip as the block's contact evaluates it: the registered shape ``shape``, the block's axis ``origin`` (world),
    and, traced each step, the tip's ``pose``, its part's ``centre`` and ``rotation`` in the body's frame, its region's
    parameters ``region`` and the depth ``depth_ref`` the pose puts its lowest point at."""

    shape: str
    top: float
    origin: tuple[float, float, float]
    pose: Any = None
    centre: Any = None
    rotation: Any = None
    region: Any = None
    depth_ref: Any = 0.0


def registered_distance(ind: ShapeTip, R: Array, Z: Array, depth: Array) -> Array:
    """The tip's signed distance at the block's nodes (R, Z), the tip moved down by ``depth - depth_ref`` from its
    pose: the point is taken into the tip's part frame and the registered shape evaluates it."""
    shape = shape_named(ind.shape)
    shift = depth - ind.depth_ref
    o = jnp.asarray(ind.origin, dtype=R.dtype)
    world = jnp.stack(jnp.broadcast_arrays(o[0] + R, o[1] + jnp.zeros_like(R), o[2] + Z + shift), axis=-1)
    body = (world - ind.pose[:3]) @ quaternion.rotation_matrix(ind.pose[3:])  # R^T (x - p), row-wise
    local = (body - ind.centre) @ ind.rotation  # rot^T (b - c), row-wise
    return shape.signed_distance(ind.region, local)


class BlockState(TypedDict, closed=True):
    u: ReadOnly[Annotated[Float[Array, "nx ny 2"], Units("m")]]
    d: ReadOnly[Annotated[Float[Array, "nx ny"], Units("1")]]
    history: ReadOnly[Annotated[Float[Array, "e 4"], Units("J/m^3")]]
    depth: ReadOnly[Annotated[Float[Array, ""], Units("m")]]
    force: ReadOnly[Annotated[Float[Array, ""], Units("N")]]
    force_normal: ReadOnly[Annotated[Float[Array, ""], Units("N")]]
    energy: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    crack_energy: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    contact_energy: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    friction_work: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    viscous_work: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    capped: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    tip_pose: ReadOnly[Annotated[Float[Array, "7"], Units("m; 1")]]
    crack_area: ReadOnly[Annotated[Float[Array, ""], Units("m^2")]]
    surface_points: ReadOnly[Annotated[Float[Array, "v 3"], Units("m")]]
    surface_damage: ReadOnly[Annotated[Float[Array, " v"], Units("1")]]


class Block3DState(TypedDict, closed=True):
    u: ReadOnly[Annotated[Float[Array, "nx ny nz 3"], Units("m")]]
    d: ReadOnly[Annotated[Float[Array, "nx ny nz"], Units("1")]]
    history: ReadOnly[Annotated[Float[Array, "2 2 2 ex ey ez"], Units("J/m^3")]]
    depth: ReadOnly[Annotated[Float[Array, ""], Units("m")]]
    force: ReadOnly[Annotated[Float[Array, ""], Units("N")]]
    energy: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    crack_energy: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    contact_energy: ReadOnly[Annotated[Float[Array, ""], Units("J")]]
    capped: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    staggered: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    newton: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    krylov: ReadOnly[Annotated[Float[Array, ""], Units("1")]]
    tip_pose: ReadOnly[Annotated[Float[Array, "7"], Units("m; 1")]]
    crack_area: ReadOnly[Annotated[Float[Array, ""], Units("m^2")]]
    surface_points: ReadOnly[Annotated[Float[Array, "v 3"], Units("m")]]
    surface_damage: ReadOnly[Annotated[Float[Array, " v"], Units("1")]]


class BlockInputs(TypedDict, closed=True):
    indenter: NotRequired[ReadOnly[Float[Array, " n"]]]


def unpack(v: Array, names: Sequence[str]) -> dict[str, Any]:
    """The law's vector as the contact's traced numbers (``ShapeTip``'s fields and the interface's)."""
    k = 19 + N_REGION  # a shape of fewer parameters is padded (``softtissue.tip_contact``)
    return dict(pose=v[:7], centre=v[7:10], rotation=v[10:19].reshape(3, 3),
                region={n: v[19 + i] for i, n in enumerate(names)}, friction=v[k], adhesion=v[k + 1],
                stiffness=v[k + 2])


class BlockNode(MimeNode):
    """The block's displacement, damage and history fields; each step the equilibrium at the tip's pose."""

    def __init__(self, name: str, timestep: float, *, radius: float, height: float, h: float, fine: float,
                 young: float, poisson: float, toughness: float, length_scale: float, stretch_limit: float,
                 residual: float, viscosity: float, origin: Sequence[float], shape: str, mode: str,
                 lag_relaxation: float = 0.5, yeoh_c2: float = 0.0) -> None:
        super().__init__(name, timestep, radius=float(radius), height=float(height), h=float(h), fine=float(fine),
                         young=float(young), poisson=float(poisson), toughness=float(toughness),
                         length_scale=float(length_scale), stretch_limit=float(stretch_limit),
                         residual=float(residual), viscosity=float(viscosity),
                         origin=tuple(float(c) for c in origin), shape=str(shape), mode=str(mode),
                         lag_relaxation=float(lag_relaxation), yeoh_c2=float(yeoh_c2))
        self._solvers: tuple[Solver, Solver] | None = None

    def solvers(self) -> tuple[Solver, Solver]:
        """The insertion's (round 3's proportional first guess) and the withdrawal's (the tip moved alone)."""
        if self._solvers is None:
            own = self.constants()
            made = []
            for predictor in ("proportional", "prescribed_only"):
                block = axisymmetric_block(radius=own["radius"], height=own["height"], h=own["h"], fine_r=own["fine"],
                                           fine_z=own["fine"], predictor=predictor)
                g = block.grid
                w = np.broadcast_to(nodal_ring_area(g.x)[:, None], (g.nx + 1, g.ny + 1)).copy()
                tip = ShapeTip(shape=own["shape"], top=own["height"], origin=own["origin"])
                c = ct.Contact(indenter=tip, stiffness=1.0, weights=w, active=np.ones_like(w), friction=0.0,
                               slip=1e-2 * own["h"], distance=registered_distance, always_sliding=True)
                from dataclasses import replace
                made.append(Solver(replace(block, contact=c), own["mode"], tol_d=1e-5,
                                   viscosity=own["viscosity"] * own["toughness"] / own["length_scale"],
                                   newton_regularise=True, lag_relaxation=own["lag_relaxation"]))
            self._solvers = (made[0], made[1])
        return self._solvers

    def surface(self, u: Array, d: Array) -> tuple[Array, Array]:
        """The section shown as a surface (``_display``): each vertex's point and damage."""
        g = self.solvers()[0].grid
        i, j, theta, _ = revolved(g.nx, g.ny)
        points = placed(g.x, g.y, u, i, j, jnp.asarray(theta, dtype=u.dtype), jnp)
        return points, d[i, j]  # type: ignore[return-value]

    @override
    def initial_state(self) -> NodeFields:
        s = self.solvers()[0].initial(jnp.zeros(()).dtype)
        z = jnp.zeros((), dtype=s.u.dtype)
        points, damage = self.surface(s.u, s.d)
        return node_fields(BlockState(u=s.u, d=s.d, history=s.history, depth=z, force=z, force_normal=z, energy=z,
                                      crack_energy=z, contact_energy=z, friction_work=z, viscous_work=z, capped=z,
                                      tip_pose=jnp.asarray(AWAY, dtype=s.u.dtype), crack_area=z,
                                      surface_points=points, surface_damage=damage))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        dtype = jnp.zeros(()).dtype
        away = jnp.zeros(N_INDENTER, dtype=dtype).at[:7].set(jnp.asarray(AWAY, dtype=dtype))
        away = away.at[10:19].set(jnp.eye(3, dtype=dtype).reshape(9))
        return {"indenter": BoundaryInputSpec(shape=(N_INDENTER,), dtype=dtype, default=away, expected_units="1")}

    def tip_depth(self, given: dict[str, Any], dtype: Any) -> Array:
        """The tip's lowest point (its part frame's origin) below the block's top face."""
        apex = given["pose"][:3] + quaternion.rotate(given["pose"][3:], given["centre"])
        own = self.constants()
        return jnp.asarray(own["origin"][2] + own["height"], dtype=dtype) - apex[2]

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        last = typed_view(BlockState, state)
        dtype = last["u"].dtype
        v = typed_view(BlockInputs, boundary_inputs).get("indenter", self.boundary_input_spec()["indenter"].default)
        names = shape_named(self.constants()["shape"]).parameter_names
        given = unpack(jnp.asarray(v, dtype=dtype), names)
        depth = self.tip_depth(given, dtype)
        tip = dict(given, depth_ref=depth)
        mat = engine_material(p, self.constants()["mode"])
        inserting, withdrawing = self.solvers()
        start = State(last["u"], last["d"], last["history"], jnp.maximum(last["depth"], 0.0))

        def step(solver: Solver) -> Any:
            return lambda: solver.advance(start, depth, mat, None, tip)

        new, out = jax.lax.cond(depth >= start.delta, step(inserting), step(withdrawing))
        points, damage = self.surface(new.u, new.d)
        return node_fields(BlockState(u=new.u, d=new.d, history=new.history, depth=depth, force=out.force,
                                      force_normal=out.force_normal, energy=out.energy,
                                      crack_energy=out.crack_energy, contact_energy=out.contact_energy,
                                      friction_work=out.friction_work, viscous_work=out.viscous_work,
                                      capped=out.newton_capped.astype(dtype), tip_pose=given["pose"],
                                      crack_area=out.crack_energy / p["toughness"], surface_points=points,
                                      surface_damage=damage))


def registered_distance_3d(shape: str, origin: Sequence[float], tip: Mapping[str, Any], points: Array,
                           depth: Array) -> Array:
    """The tip's signed distance at ``points`` (..., 3) in the block's frame, the tip moved down by ``depth -
    tip["depth_ref"]`` from its pose: the point is taken into the tip's part frame and the registered shape evaluates
    it (``registered_distance``'s 3-D form)."""
    dtype = points.dtype
    o = jnp.asarray(origin, dtype=dtype)
    shift = jnp.asarray(depth, dtype) - jnp.asarray(tip["depth_ref"], dtype)
    world = points + o + jnp.zeros(3, dtype).at[2].set(shift)
    pose = jnp.asarray(tip["pose"], dtype)
    body = (world - pose[:3]) @ quaternion.rotation_matrix(pose[3:])
    local = (body - jnp.asarray(tip["centre"], dtype)) @ jnp.asarray(tip["rotation"], dtype)
    region = {k: jnp.asarray(v, dtype) for k, v in tip["region"].items()}
    return shape_named(shape).signed_distance(region, local)


class Block3DNode(MimeNode):
    """The quarter block's displacement, damage and history fields; each step the equilibrium at the tip's pose
    (``engine.solid3d.block``, frictionless)."""

    def __init__(self, name: str, timestep: float, *, radius: float, height: float, h: float, fine: float,
                 young: float, poisson: float, toughness: float, length_scale: float, stretch_limit: float,
                 residual: float, origin: Sequence[float], shape: str, yeoh_c2: float = 0.0) -> None:
        super().__init__(name, timestep, radius=float(radius), height=float(height), h=float(h), fine=float(fine),
                         young=float(young), poisson=float(poisson), toughness=float(toughness),
                         length_scale=float(length_scale), stretch_limit=float(stretch_limit),
                         residual=float(residual), origin=tuple(float(c) for c in origin), shape=str(shape),
                         yeoh_c2=float(yeoh_c2))
        self._block: b3.Block3D | None = None

    def block(self) -> b3.Block3D:
        if self._block is None:
            own = self.constants()
            shape, origin = own["shape"], own["origin"]

            def distance(tip: Mapping[str, Any], points: Array, depth: Array) -> Array:
                return registered_distance_3d(shape, origin, tip, points, depth)

            self._block = b3.Block3D(half_width=own["radius"], height=own["height"], h=own["h"], distance=distance,
                                     fine_xy=FINE_XY * own["fine"], fine_z=own["fine"])
        return self._block

    def surface(self, u: Array, d: Array) -> tuple[Array, Array]:
        """The quarter's boundary shown as a surface (``_display.quarter_boundary``): each vertex's point and damage."""
        blk = self.block()
        ijk, _ = quarter_boundary(*blk.grid.shape)
        i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
        points = jnp.asarray(blk.X_np[i, j, k], dtype=u.dtype) + u[i, j, k]
        return points, d[i, j, k]

    @override
    def initial_state(self) -> NodeFields:
        s = self.block().initial(jnp.zeros(()).dtype)
        z = jnp.zeros((), dtype=s.u.dtype)
        points, damage = self.surface(s.u, s.d)
        return node_fields(Block3DState(u=s.u, d=s.d, history=s.history, depth=z, force=z, energy=z, crack_energy=z,
                                        contact_energy=z, capped=z, staggered=z, newton=z, krylov=z,
                                        tip_pose=jnp.asarray(AWAY, dtype=s.u.dtype), crack_area=z,
                                        surface_points=points, surface_damage=damage))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        dtype = jnp.zeros(()).dtype
        away = jnp.zeros(N_INDENTER, dtype=dtype).at[:7].set(jnp.asarray(AWAY, dtype=dtype))
        away = away.at[10:19].set(jnp.eye(3, dtype=dtype).reshape(9))
        return {"indenter": BoundaryInputSpec(shape=(N_INDENTER,), dtype=dtype, default=away, expected_units="1")}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        last = typed_view(Block3DState, state)
        dtype = last["u"].dtype
        v = typed_view(BlockInputs, boundary_inputs).get("indenter", self.boundary_input_spec()["indenter"].default)
        names = shape_named(self.constants()["shape"]).parameter_names
        given = unpack(jnp.asarray(v, dtype=dtype), names)
        apex = given["pose"][:3] + quaternion.rotate(given["pose"][3:], given["centre"])
        own = self.constants()
        depth = jnp.asarray(own["origin"][2] + own["height"], dtype=dtype) - apex[2]
        tip = dict(given, depth_ref=depth)
        mat = engine_material(p, "phase_field")
        blk = self.block()
        start = b3.State(last["u"], last["d"], last["history"], jnp.maximum(last["depth"], 0.0))
        new, out = blk.advance(start, depth, mat, tip)
        points, damage = self.surface(new.u, new.d)
        return node_fields(Block3DState(u=new.u, d=new.d, history=new.history, depth=depth, force=out.force,
                                        energy=out.energy, crack_energy=out.crack_energy,
                                        contact_energy=out.contact_energy, capped=out.newton_capped.astype(dtype),
                                        staggered=out.staggered.astype(dtype), newton=out.newton.astype(dtype),
                                        krylov=out.krylov.astype(dtype), tip_pose=given["pose"],
                                        crack_area=out.crack_energy / p["toughness"], surface_points=points,
                                        surface_damage=damage))


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class GelBlock(Dynamics):
    """A block of gel or clot, ``radius`` and ``height``, its bottom face fixed (its centre at the body's initial
    ``position``), its side and top free, pressed and pulled by a coaxial rigid tip through a law
    (``softtissue.tip_contact``); grid spacing ``h`` within ``fine`` of the axis and of the top, graded beyond; the
    material from the record ``material`` (a Yeoh term where the record has ``yeoh_c2``); fracture by the AT2 phase
    field with a viscous term ``viscosity`` (in units of Gc/l), or, as a control, element deletion (``mode="local"``).
    ``tip_shape`` names the registered shape the tip's geometry kind evaluates (the law cannot hand a name to the
    block's build). ``dimension``: ``"axisymmetric"`` (a cylinder of ``radius``), or ``"3d"`` (a square block
    ``2 radius`` across, a quarter solved by symmetry, frictionless, the phase field only, no viscous term, insertion
    only). Its surface: the axisymmetric section revolved for display, or the 3-D quarter's boundary, both coloured
    by damage."""

    name = "softtissue.gel_block"
    version = VERSION
    unmetered = "the block keeps its own energy account (stored, crack surface, penalty, friction, viscous)"
    differentiability = GradientStatus.PIECEWISE_KINK
    gradient = GradientRecord(switches=(DAMAGE_GROWTH, CONTACT_ONSET, PREDICTOR))
    references = ("Miehe2010",)
    validity = (
        "axisymmetric: the tip must be coaxial with the block, and a sharp tip's planar crack cannot form (a ring "
        "crack is the only one the build has); 3-D: the tip must be coaxial too (a quarter is solved, symmetric about "
        "both planes through the axis), frictionless and without adhesion whatever the law's settings, insertion only",
        "quasi-static only: each base step is a static equilibrium at the tip's pose; no inertia, no rate dependence "
        "beyond the viscous damage term, no poroelasticity",
        "the contact's node weights are ring areas, right where the grid is uniform (within fine of the axis and top)",
        "forces after the puncture's peak are those of a regularised Newton (round 3); a step whose solve capped is "
        "flagged in capped",
    )
    builds = {"": (BlockNode, Block3DNode)}
    applies_to = frozenset({ThingKind.BODY})
    publishes = {TIP_FORCE.name: Port("force"), TIP_DEPTH.name: Port("depth"), CRACK_AREA.name: Port("crack_area")}
    accepts = {INDENTER.name: Port("indenter")}
    initial_state = {"position": LENGTH}
    setting_roles = {"radius": DESIGN, "height": DESIGN, "material": DESIGN,
                     "h": convention(NUMERICAL, "the grid near the tip, at most half the length scale"),
                     "fine": convention(NUMERICAL, "the uniform region near the axis and the top"),
                     "viscosity": convention(NUMERICAL, "a viscous damage term, units of Gc/l (0.1: round 3's)"),
                     "mode": convention(PHYSICAL, "phase_field is the model; local is the control"),
                     "lag_relaxation": convention(NUMERICAL, "friction's lagged pressure under-relaxed (round 4: a "
                                                              "two-cycle at 1.0 on withdrawal)"),
                     "tip_shape": convention(DESIGN, "the registered shape of the tip's kind"),
                     "dimension": convention(NUMERICAL, "axisymmetric, or 3d (a quarter, spike 3D-COST's all2 solver)")}
    radius: Annotated[Quantity, Length]
    height: Annotated[Quantity, Length]
    material: str
    h: Annotated[Quantity, Length] = mm(0.05)
    fine: Annotated[Quantity, Length] = mm(1.5)
    viscosity: float = 0.1
    mode: str = "phase_field"
    tip_shape: str = "softtissue.headed_tip"
    lag_relaxation: float = 0.5
    dimension: str = "axisymmetric"

    @override
    def assets_read(self) -> tuple[str, ...]:
        return (self.material,)

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found: list[str] = []
        if self.dimension not in DIMENSIONS:
            found.append(f"dimension is {self.dimension!r}, not one of {DIMENSIONS}")
        elif self.dimension == "3d" and self.mode != "phase_field":
            found.append(f"mode {self.mode!r} has no 3-D build (the 3-D block is the phase field's)")
        return found

    @override
    def surface_meshes(self, settings: Mapping[str, Magnitude]) -> tuple[SurfaceMesh, ...]:
        """The axisymmetric section revolved for display (``_display.revolved``), or the 3-D quarter's boundary with
        its two symmetry faces (``_display.quarter_boundary``), coloured by damage over its range, 0 to 1."""
        radius, height, h, fine = (float(settings[n]) for n in ("radius", "height", "h", "fine"))  # pyright: ignore[reportArgumentType]
        damage = (MeshField("damage", "1", Port("surface_damage"), (0.0, 1.0), reduction="max"),)
        if self.dimension == "3d":
            x = b3.rows(h, radius, FINE_XY * fine, 1.5, 16.0, False)
            z = b3.rows(h, height, fine, 1.5, 16.0, True)
            ijk, faces = quarter_boundary(len(x) - 1, len(x) - 1, len(z) - 1)
            rest = np.stack([x[ijk[:, 0]], x[ijk[:, 1]], z[ijk[:, 2]]], axis=1)
            return (SurfaceMesh("quarter", rest, faces, Port("surface_points"), damage),)
        g = axisymmetric_block(radius=radius, height=height, h=h, fine_r=fine, fine_z=fine).grid
        i, j, theta, faces = revolved(g.nx, g.ny)
        rest = np.asarray(placed(g.x, g.y, None, i, j, theta))
        return (SurfaceMesh("section", rest, faces, Port("surface_points"), damage, display="revolved"),)

    @override
    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        values = read_material(self.material)
        origin = tuple(float(np.asarray(getattr(c, "value", c))) for c in _components(ctx.initial["position"]))
        if self.dimension == "3d":
            b.add_node(Block3DNode(
                ctx.node_name(), ctx.timestep, radius=ctx.scalar("radius"), height=ctx.scalar("height"),
                h=ctx.scalar("h"), fine=ctx.scalar("fine"), young=values["young_modulus"],
                poisson=values["poisson_ratio"], toughness=values["toughness"], length_scale=values["length_scale"],
                stretch_limit=values["stretch_limit"], residual=values.get("residual_stiffness", 1e-6), origin=origin,
                shape=self.tip_shape, yeoh_c2=values.get("yeoh_c2", 0.0)))
            return
        b.add_node(BlockNode(
            ctx.node_name(), ctx.timestep, radius=ctx.scalar("radius"), height=ctx.scalar("height"),
            h=ctx.scalar("h"), fine=ctx.scalar("fine"), young=values["young_modulus"],
            poisson=values["poisson_ratio"], toughness=values["toughness"], length_scale=values["length_scale"],
            stretch_limit=values["stretch_limit"], residual=values.get("residual_stiffness", 1e-6),
            viscosity=self.viscosity, origin=origin, shape=self.tip_shape, mode=self.mode,
            lag_relaxation=self.lag_relaxation, yeoh_c2=values.get("yeoh_c2", 0.0)))


def _components(value: Any) -> tuple[Any, ...]:
    held = getattr(value, "value", value)
    return tuple(np.asarray(held, dtype=float).reshape(-1))
