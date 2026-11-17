"""Rigid tips as lab geometry kinds, each with the registered shape its one region is evaluated by: the engine's own
axisymmetric signed distances (``engine.contact``) turned about the tip's z axis, so the same tip presses an
axisymmetric block or a 3-D one through the one contact law (``softtissue.tip_contact``).

- ``softtissue.punch_tip``: a cylinder of ``radius`` whose end's edge is rounded by ``fillet``. ``fillet = radius`` is a
  **sphere-tipped needle** (a capsule's end); a small fillet a **flat punch**.
- ``softtissue.cone_tip``: a cone of ``half_angle`` whose apex is rounded by ``tip_radius``, up to a shaft of ``radius``:
  a **sharp cone** (a wide shaft) or a **needle with a conical point** (the needle's radius as the shaft).

Each kind's frame is at its lowest point on the axis, the tip pointing down -z, as ``softtissue.headed_tip``'s (a design
setting does not move the working surface). Volume and inertia are a cylinder's of ``radius`` and ``length``, the
point left out (the experiment declares the body's mass and inertia)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, override

import jax
import jax.numpy as jnp

from mime.core import register_shape, switch
from mime.gradients import GradientStatus
from mime.registry import PHYSICAL, Drawing, GradientRecord, Solid, Tensor, register_geometry
from mime.units import DIMENSIONLESS, Length, Magnitude, Quantity

from ..engine import contact as ct
from .headed_tip import _radius

AXIS = switch.Declaration(
    owner="softtissue.tips.axis", indicator="the point's distance from the tip's axis, m", orders={"distance": 0, "normal": -1},
    route="none needed: on the axis the radial direction has no limit; the distance is finite and its derivative "
          "along the axis is defined")
EDGES = switch.Declaration(
    owner="softtissue.tips.edges", indicator="which of the tip's surfaces is nearest, m", orders={"distance": 0, "normal": -1},
    route="none needed: the signed distance is continuous with a kink where the nearest surface changes (the medial "
          "set), and the contact reads it only inside the tip")


class _ProfileShape:
    """A registered shape from one of the engine's indenter kinds: the profile at the point's distance from the z axis
    and its z, the lowest point at the origin."""

    kind: str = ""
    parameter_names: tuple[str, ...] = ()
    takes_many = True

    def indenter(self, parameters: Mapping[str, Any]) -> ct.Indenter:
        raise NotImplementedError

    def signed_distance(self, parameters: Mapping[str, Any], point: Any) -> Any:
        point = jnp.asarray(point)
        return ct.signed_distance(self.indenter(parameters), _radius(point), point[..., 2],
                                  jnp.zeros((), dtype=point.dtype))

    def outward_normal(self, parameters: Mapping[str, Any], point: Any) -> Any:
        def one(p: Any) -> Any:
            g = jax.grad(lambda q: self.signed_distance(parameters, q))(p)
            n = jnp.sqrt(jnp.sum(g * g))
            return g / jnp.where(n > 0, n, 1.0)
        point = jnp.asarray(point)
        return one(point) if point.ndim == 1 else jax.vmap(one)(point.reshape(-1, 3)).reshape(point.shape)

    def axis(self, parameters: Mapping[str, Any]) -> Any:
        return jnp.asarray((0.0, 0.0, 1.0))

    def axis_point(self, parameters: Mapping[str, Any]) -> Any:
        return jnp.zeros(3)


class PunchShape(_ProfileShape):
    kind = "punch"
    parameter_names = ("radius", "fillet")

    def indenter(self, parameters: Mapping[str, Any]) -> ct.Indenter:
        return ct.Indenter("punch", radius=parameters["radius"], fillet=parameters["fillet"], top=0.0)


class ConeShape(_ProfileShape):
    kind = "cone"
    parameter_names = ("radius", "tip_radius", "half_angle")

    def indenter(self, parameters: Mapping[str, Any]) -> ct.Indenter:
        return ct.Indenter("cone", radius=parameters["radius"], fillet=parameters["tip_radius"], top=0.0,
                           half_angle=parameters["half_angle"])


PUNCH = register_shape("softtissue.punch_tip", PunchShape(), switches=(AXIS, EDGES))
CONE = register_shape("softtissue.cone_tip", ConeShape(), switches=(AXIS, EDGES))


def _value(settings: Mapping[str, Magnitude], name: str) -> float:
    value = settings[name]
    if isinstance(value, tuple):
        raise TypeError(f"setting {name!r} is one number, not {value!r}")
    return float(value)


def _revolved(profile: list[tuple[float, float]], segments: int, what: str) -> Drawing:
    """A closed mesh of a profile (r, z) from the axis's bottom point to the top's rim, turned in ``segments`` facets;
    the top capped by a fan from its centre."""
    length = profile[-1][1]
    points: list[tuple[float, float, float]] = [(0.0, 0.0, profile[0][1]), (0.0, 0.0, length)]
    ring = profile[1:]
    n = len(ring)
    for k in range(segments):
        angle = 2.0 * math.pi * k / segments
        points += [(r * math.cos(angle), r * math.sin(angle), z) for r, z in ring]

    def at(k: int, m: int) -> int:
        return 2 + (k % segments) * n + m

    counts: list[int] = []
    indices: list[int] = []
    for k in range(segments):
        indices += [0, at(k + 1, 0), at(k, 0)]
        indices += [1, at(k, n - 1), at(k + 1, n - 1)]
        counts += [3, 3]
        for m in range(n - 1):
            indices += [at(k, m), at(k + 1, m), at(k + 1, m + 1), at(k, m + 1)]
            counts.append(4)
    return Drawing(tuple(points), tuple(counts), tuple(indices), f"{what} turned in {segments} facets")


class _Tip(Solid):
    """What the two tip kinds share: one region, ``body``; a cylinder's volume and inertia."""

    regions = ("body",)
    validity = ("rigid: the tip does not bend or wear; only its signed distance reaches the contact",
                "axisymmetric about its own z axis: a bevel or a side port is not drawn by these kinds")
    differentiability = GradientStatus.PIECEWISE_KINK
    gradient = GradientRecord()
    references = ()
    builds = {"": ()}

    def _cylinder(self, settings: Mapping[str, Magnitude]) -> tuple[float, float]:
        return _value(settings, "radius"), _value(settings, "length")

    @override
    def volume(self, settings: Mapping[str, Magnitude]) -> float:
        r, length = self._cylinder(settings)
        return math.pi * r * r * length

    @override
    def inertia_per_mass(self, settings: Mapping[str, Magnitude]) -> Tensor:
        r, length = self._cylinder(settings)
        along = r * r / 2
        across = r * r / 4 + length * length / 3  # about the end's axis point, not the centroid (as headed_tip's)
        return (across, 0.0, 0.0), (0.0, across, 0.0), (0.0, 0.0, along)

    @override
    def enclosing_radius(self, settings: Mapping[str, Magnitude]) -> float:
        r, length = self._cylinder(settings)
        return math.hypot(r, length)

    @override
    def radial_reach(self, settings: Mapping[str, Magnitude]) -> float:
        return _value(settings, "radius")


@register_geometry
@dataclass(frozen=True, kw_only=True)
class PunchTip(_Tip):
    """A rigid cylinder of ``radius``, ``length`` long, along its frame's z, its end's edge rounded by ``fillet`` and its
    lowest point at the frame's origin: a flat punch (small fillet) or a sphere-tipped needle (``fillet = radius``)."""

    name = "softtissue.punch_tip"
    version = "0.1.0"
    setting_roles = dict.fromkeys(("radius", "fillet", "length"), PHYSICAL)
    region_shapes = {"body": "softtissue.punch_tip"}
    radius: Annotated[Quantity, Length]
    fillet: Annotated[Quantity, Length]
    length: Annotated[Quantity, Length]

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found = [f"{n} is {settings[n]!r} m, not a length above zero" for n in ("radius", "fillet", "length")
                 if not (isinstance(settings[n], (int, float)) and math.isfinite(settings[n]) and settings[n] > 0)]
        if not found and _value(settings, "fillet") > _value(settings, "radius"):
            found.append("the fillet is larger than the radius (a sphere-tipped needle has fillet = radius)")
        if not found and _value(settings, "length") <= _value(settings, "fillet"):
            found.append("the tip is not longer than its rounded end")
        return found

    @override
    def region_parameters(self, region: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        if region != "body":
            raise KeyError(region)
        return {"radius": _value(settings, "radius"), "fillet": _value(settings, "fillet")}

    def drawing(self, settings: Mapping[str, Magnitude], segments: int) -> Drawing:
        r, f, length = (_value(settings, n) for n in ("radius", "fillet", "length"))
        arc = [(r - f + f * math.sin(a), f - f * math.cos(a)) for a in
               (0.5 * math.pi * k / max(2, segments // 8) for k in range(1, max(2, segments // 8) + 1))]
        start = [(r - f, 0.0)] if r - f > 1e-12 * r else []
        return _revolved([(0.0, 0.0), *start, *arc, (r, length)], segments, "the punch tip, its rounded end")


@register_geometry
@dataclass(frozen=True, kw_only=True)
class ConeTip(_Tip):
    """A rigid cone of ``half_angle`` along its frame's z, its apex rounded by ``tip_radius`` at the frame's origin, up
    to a shaft of ``radius``, ``length`` long: a sharp cone, or a needle with a conical point."""

    name = "softtissue.cone_tip"
    version = "0.1.0"
    setting_roles = dict.fromkeys(("radius", "tip_radius", "half_angle", "length"), PHYSICAL)
    region_shapes = {"body": "softtissue.cone_tip"}
    radius: Annotated[Quantity, Length]
    tip_radius: Annotated[Quantity, Length]
    #: An angle: MIME's `Angle` annotation is not a setting kind (the setting would be dropped), so it is a dimensionless
    #: quantity, written with `deg(...)` or `rad(...)`.
    half_angle: Annotated[Quantity, DIMENSIONLESS]
    length: Annotated[Quantity, Length]

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found = [f"{n} is {settings[n]!r}, not a value above zero" for n in ("radius", "tip_radius", "half_angle", "length")
                 if not (isinstance(settings[n], (int, float)) and math.isfinite(settings[n]) and settings[n] > 0)]
        if not found and not _value(settings, "half_angle") < 0.5 * math.pi:
            found.append("the half-angle is not below a right angle")
        if not found and _value(settings, "tip_radius") >= _value(settings, "radius"):
            found.append("the apex's rounding is not smaller than the shaft")
        return found

    @override
    def region_parameters(self, region: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        if region != "body":
            raise KeyError(region)
        return {n: _value(settings, n) for n in ("radius", "tip_radius", "half_angle")}

    def drawing(self, settings: Mapping[str, Magnitude], segments: int) -> Drawing:
        r, rho, a, length = (_value(settings, n) for n in ("radius", "tip_radius", "half_angle", "length"))
        # the rounded apex's tangent point, then the flank to the shaft (the engine's cone: apex at z = rho)
        z_cone = r / math.tan(a) + rho - rho / math.sin(a)  # where the flank meets the shaft, from the lowest point
        touch = (rho * math.cos(a), rho - rho * math.sin(a))
        return _revolved([(0.0, 0.0), touch, (r, z_cone), (r, max(length, z_cone * 1.01))], segments,
                         "the cone tip, its apex rounded")
