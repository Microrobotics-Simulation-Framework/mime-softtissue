"""``softtissue.replayed_tear``: the pure-shear study's saved run (the frames in ``studies/results``) replayed as a body
whose surface MIME records and shows, coloured by damage. It is a replay, not a simulation: step ``k`` holds frame ``k`` of
a run the engine made outside MIME, so it computes nothing and says so in its validity and its gradient record."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cache
from dataclasses import dataclass
from typing import Annotated, Any, override

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float, Int
from typing_extensions import ReadOnly, TypedDict

from mime.build import DynamicsBuilder, ResolvedRegistration
from mime.core import MimeNode, NodeFields, Units, node_fields, typed_view
from mime.gradients import GradientStatus
from mime.registry import (NUMERICAL, BuildContext, Dynamics, GradientRecord, MeshField, Port, SurfaceMesh,
                           convention, register_dynamics)
from mime.things import ThingKind
from mime.units import LENGTH, Length, Quantity, mm

from ..studies import results
from ._vocabulary import VERSION

#: The sheet's thickness (mm) and the test's grip rate (mm/s): the thinning drawn on the front face, and the clock one
#: frame spacing takes at the rate the specimens were pulled.
THICKNESS_MM, GRIP_RATE_MM_S = 3.0, 0.1


class ReplayState(TypedDict, closed=True):
    frame: ReadOnly[Annotated[Int[Array, ""], Units("1")]]
    points: ReadOnly[Annotated[Float[Array, "v 3"], Units("m")]]
    damage: ReadOnly[Annotated[Float[Array, " v"], Units("1")]]


@cache
def replay(every_column: int, every_frame: int, to_mm: float) -> tuple[Any, ...]:
    """The saved run as a surface: ``rest`` ``(V, 3)``, ``faces``, each frame's ``points`` ``(K, V, 3)`` and ``damage``
    ``(K, V)`` (both halves of the sheet, standing upright, the crack plane opened) and the seconds a frame takes.
    Every ``every_column``-th column of the grid and every ``every_frame``-th frame up to ``to_mm`` of grip displacement
    are kept. The front face is displaced by the sheet's thinning, ``1 / det F`` (display only)."""
    f = results.frames()
    x, y = f["x_nodes_mm"], f["y_nodes_mm"]
    cols = np.unique(np.r_[np.arange(0, len(x), every_column), len(x) - 1])
    steps = f["frames_step"]
    shown = np.where(f["delta_mm"][steps] <= to_mm + 1e-9)[0][::every_frame]
    xs = x[cols]
    X, Y = np.meshgrid(xs, y, indexing="ij")
    nx, ny = X.shape

    def halves(u: Any, thinning: Any) -> Any:
        upper = np.stack([X + u[..., 0], -0.5 * THICKNESS_MM * thinning, Y + u[..., 1]], -1)
        lower = np.stack([X + u[..., 0], -0.5 * THICKNESS_MM * thinning, -(Y + u[..., 1])], -1)
        return (np.concatenate([upper.reshape(-1, 3), lower.reshape(-1, 3)]) - [x.mean(), 0.0, 0.0]) * 1e-3

    def thinning(u: Any) -> Any:
        dx = [np.gradient(u[..., i], xs, axis=0) for i in (0, 1)]
        dy = [np.gradient(u[..., i], y, axis=1) for i in (0, 1)]
        return 1.0 / np.clip((1 + dx[0]) * (1 + dy[1]) - dy[0] * dx[1], 0.2, None)

    u = f["frames_u_mm"][shown][:, cols].astype(float)
    points = np.stack([halves(u[k], thinning(u[k])) for k in range(len(shown))])
    damage = f["frames_d"][shown][:, cols].astype(float).reshape(len(shown), -1)
    index = np.arange(nx * ny).reshape(nx, ny)
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    upper = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    faces = np.concatenate([upper[:, ::-1], upper + nx * ny]).astype(np.int64)  # the mirrored half faces the other way
    rest = halves(np.zeros((nx, ny, 2)), np.ones((nx, ny)))
    spacing = float(np.diff(f["delta_mm"][steps][shown[:2]])[0]) if len(shown) > 1 else 1.0
    return rest, faces, points, np.concatenate([damage, damage], axis=1), spacing / GRIP_RATE_MM_S


class ReplayNode(MimeNode):
    """Frame ``k`` at step ``k``, the last one held after the run's end."""

    def __init__(self, name: str, timestep: float, *, every_column: int, every_frame: int, to_mm: float) -> None:
        super().__init__(name, timestep, every_column=int(every_column), every_frame=int(every_frame),
                         to_mm=float(to_mm))
        own = self.constants()
        _, _, self._points, self._damage, _ = replay(own["every_column"], own["every_frame"], own["to_mm"])

    def _at(self, frame: Array, dtype: Any) -> NodeFields:
        k = jnp.clip(frame, 0, len(self._points) - 1)
        return node_fields(ReplayState(frame=frame, points=jnp.asarray(self._points, dtype)[k],
                                       damage=jnp.asarray(self._damage, dtype)[k]))

    @override
    def initial_state(self) -> NodeFields:
        return self._at(jnp.zeros((), dtype=jnp.int32), jnp.zeros(()).dtype)

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        last = typed_view(ReplayState, state)
        return self._at(last["frame"] + 1, last["points"].dtype)


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class ReplayedTear(Dynamics):
    """The pure-shear study's group-4 sheet tearing, replayed from its saved frames (both halves, the crack plane
    opened), its surface coloured by the phase field's damage (0 to 1). Every ``every_column``-th grid column and
    every ``every_frame``-th frame up to the grip displacement ``to_mm`` are kept, which sets the size of the page. One
    step is one kept frame: set the experiment's base step to ``step_seconds`` (the test's clock at 0.1 mm/s) for the
    page's time to be the test's."""

    name = "softtissue.replayed_tear"
    version = VERSION
    unmetered = "a replay of a run the engine made outside MIME, which kept its own energy account"
    differentiability = GradientStatus.CUT
    gradient = GradientRecord(
        where="every input: no setting or state reaches a frame",
        reason="the frames are saved data, so the replay computes nothing to differentiate",
        route="none for the replay, which is the view only; the differentiable path is the engine's solver itself "
              "(mime_softtissue.studies.pure_shear, forward mode)")
    validity = (
        "a replay of saved frames, not a simulation: nothing a setting names changes a frame but which are kept",
        "the group-4 run's two-dimensional incompressible plane-stress half sheet drawn as a surface, both halves, the "
        "crack plane opened; the front face is displaced by 1 / det F for display, so the depth is not a solution",
    )
    builds = {"": (ReplayNode,)}
    applies_to = frozenset({ThingKind.BODY})
    initial_state = {"position": LENGTH}
    setting_roles = {"every_column": convention(NUMERICAL, "the grid columns kept, a size choice for the page"),
                     "every_frame": convention(NUMERICAL, "the frames kept, a size choice for the page"),
                     "to_mm": convention(NUMERICAL, "the last grip displacement shown, just past the crack's run "
                                                    "through the sheet")}
    every_column: int = 4
    every_frame: int = 1
    to_mm: Annotated[Quantity, Length] = mm(3.6)

    def _replayed(self) -> tuple[Any, ...]:
        return replay(self.every_column, self.every_frame, float(getattr(self.to_mm, "value")) * 1e3)

    @property
    def step_seconds(self) -> float:
        """The seconds one kept frame takes at the test's grip rate: the experiment's base step."""
        return self._replayed()[4]

    @property
    def duration_seconds(self) -> float:
        """The seconds from the first kept frame to the last: the run's duration."""
        return (len(self._replayed()[2]) - 1) * self.step_seconds

    @override
    def surface_meshes(self, settings: Mapping[str, Any]) -> tuple[SurfaceMesh, ...]:
        rest, faces, *_ = replay(self.every_column, self.every_frame, float(settings["to_mm"]) * 1e3)
        return (SurfaceMesh("sheet", rest, faces, Port("points"),
                            (MeshField("damage", "1", Port("damage"), (0.0, 1.0), reduction="max"),),
                            display="extruded"),)

    @override
    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        b.add_node(ReplayNode(ctx.node_name(), ctx.timestep, every_column=self.every_column,
                              every_frame=self.every_frame, to_mm=ctx.scalar("to_mm") * 1e3))
