"""What the plugin's tip kinds share: each is a lab geometry kind with one region, ``body``, evaluated by a registered
shape that is one of the engine's axisymmetric indenter profiles (``engine.contact``) turned about the tip's z axis, so
the same tip presses an axisymmetric block or a 3-D one through the one contact law (``softtissue.tip_contact``).

Each kind's frame is at its lowest point on the axis, the tip pointing down -z (a design setting does not move the
working surface in the body's frame: a solid has no anchor point of its own, a seam). The punch's and the cone's volume
and inertia are a cylinder's of ``radius`` and ``length``, the point left out (the experiment declares the body's mass
and inertia)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, override

import jax
import jax.numpy as jnp

from mime.core import switch
from mime.gradients import GradientStatus
from mime.registry import Drawing, GradientRecord, Solid, Tensor
from mime.units import Magnitude

from ..engine import contact as ct

AXIS = switch.Declaration(
    owner="softtissue.tips.axis", indicator="the point's distance from the tip's axis, m", orders={"distance": 0, "normal": -1},
    route="none needed: on the axis the radial direction has no limit; the distance is finite and its derivative "
          "along the axis is defined")
EDGES = switch.Declaration(
    owner="softtissue.tips.edges", indicator="which of the tip's surfaces is nearest, m", orders={"distance": 0, "normal": -1},
    route="none needed: the signed distance is continuous with a kink where the nearest surface changes (the medial "
          "set), and the contact reads it only inside the tip")


def radial_distance(point: Any) -> Any:
    """A point's distance from the z axis, with a finite derivative on the axis."""
    s = point[..., 0] * point[..., 0] + point[..., 1] * point[..., 1]
    return jnp.where(s > 0, jnp.sqrt(jnp.where(s > 0, s, 1.0)), 0.0)


class ProfileShape:
    """A registered shape from one of the engine's indenter kinds: the profile at the point's distance from the z axis
    and its z, the lowest point at the origin."""

    parameter_names: tuple[str, ...] = ()
    takes_many = True

    def indenter(self, parameters: Mapping[str, Any]) -> ct.Indenter:
        raise NotImplementedError

    def signed_distance(self, parameters: Mapping[str, Any], point: Any) -> Any:
        point = jnp.asarray(point)
        return ct.signed_distance(self.indenter(parameters), radial_distance(point), point[..., 2],
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


def value(settings: Mapping[str, Magnitude], name: str) -> float:
    """One setting's number (SI; an angle in radians)."""
    held = settings[name]
    if isinstance(held, tuple):
        raise TypeError(f"setting {name!r} is one number, not {held!r}")
    return float(held)


def above_zero_problems(settings: Mapping[str, Magnitude], names: tuple[str, ...]) -> list[str]:
    """The settings among ``names`` that are not a finite number above zero."""
    return [f"{n} is {settings[n]!r}, not a value above zero" for n in names if not _above_zero(settings[n])]


def _above_zero(held: Magnitude) -> bool:
    return isinstance(held, (int, float)) and math.isfinite(held) and held > 0


def revolved_drawing(profile: list[tuple[float, float]], segments: int, what: str) -> Drawing:
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


class CylinderTip(Solid):
    """What the punch and cone kinds share: one region, ``body``; a cylinder's volume and inertia of ``radius`` and
    ``length``."""

    regions = ("body",)
    validity = ("rigid: the tip does not bend or wear; only its signed distance reaches the contact",
                "axisymmetric about its own z axis: a bevel or a side port is not drawn by these kinds")
    differentiability = GradientStatus.PIECEWISE_KINK
    gradient = GradientRecord()
    references = ()
    builds = {"": ()}

    def _cylinder(self, settings: Mapping[str, Magnitude]) -> tuple[float, float]:
        return value(settings, "radius"), value(settings, "length")

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
        return value(settings, "radius")
