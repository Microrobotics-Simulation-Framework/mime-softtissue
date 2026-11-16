"""``softtissue.headed_tip``: a lab geometry kind, the axisymmetric stand-in for a barbed tip (a head of
``head_radius`` and ``head_length`` on a neck of ``neck_radius``, the head's edges rounded by ``fillet``, ``length``
long in all), and the shape its one region is evaluated with, registered under the same name.

Its frame is at the head's lowest point on the axis, the tip pointing down -z: not its centroid, as a solid's frame is
by MIME's contract, so that a design setting (the head's radius) does not move the working surface in the body's frame
(a seam: a solid has no anchor point of its own). Its volume and inertia are a cylinder pair's, the fillets left out
(the experiment declares the body's mass and inertia)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, override

import jax
import jax.numpy as jnp

from mime.core import register_shape, switch
from mime.gradients import GradientStatus
from mime.registry import PHYSICAL, GradientRecord, Solid, Tensor, register_geometry
from mime.units import Length, Magnitude, Quantity

from ..engine import contact as ct

PARAMETERS = ("neck_radius", "head_radius", "head_length", "fillet")

AXIS = switch.Declaration(
    owner="softtissue.headed_tip.axis", indicator="the point's distance from the tip's axis, m",
    orders={"distance": 0, "normal": -1},
    route="none needed: on the axis the radial direction has no limit; the distance is finite and its derivative "
          "along the axis is defined")
EDGES = switch.Declaration(
    owner="softtissue.headed_tip.edges", indicator="which of the head's and the neck's surfaces is nearest, m",
    orders={"distance": 0, "normal": -1},
    route="none needed: the signed distance is continuous with a kink where the nearest surface changes (the medial "
          "set), and the contact reads it only inside the tip")


def _profile(parameters: Mapping[str, Any], R: Any, Z: Any) -> Any:
    tip = ct.Indenter("headed", radius=parameters["neck_radius"], fillet=parameters["fillet"], top=0.0,
                      head_radius=parameters["head_radius"], head_length=parameters["head_length"])
    return ct.signed_distance(tip, R, Z, jnp.zeros((), dtype=jnp.asarray(R).dtype))


def _radius(point: Any) -> Any:
    s = point[..., 0] * point[..., 0] + point[..., 1] * point[..., 1]
    return jnp.where(s > 0, jnp.sqrt(jnp.where(s > 0, s, 1.0)), 0.0)


class HeadedTipShape:
    """The headed tip's region in its own frame: the engine's axisymmetric profile (``engine.contact``, kind
    ``headed``) at the point's distance from the z axis and its z."""

    parameter_names: tuple[str, ...] = PARAMETERS
    takes_many = True

    def signed_distance(self, parameters: Mapping[str, Any], point: Any) -> Any:
        point = jnp.asarray(point)
        return _profile(parameters, _radius(point), point[..., 2])

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


SHAPE = register_shape("softtissue.headed_tip", HeadedTipShape(), switches=(AXIS, EDGES))


def _is_positive_length(value: Magnitude) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _length(settings: Mapping[str, Magnitude], name: str) -> float:
    value = settings[name]
    if isinstance(value, tuple):
        raise TypeError(f"setting {name!r} is one length, not {value!r}")
    return float(value)


@register_geometry
@dataclass(frozen=True, kw_only=True)
class HeadedTip(Solid):
    """A rigid tip with a head on a narrower neck (the axisymmetric stand-in for a barb), along its frame's z, its
    lowest point at the frame's origin. One region, ``body``, evaluated by the shape ``softtissue.headed_tip``."""

    name = "softtissue.headed_tip"
    version = "0.1.0"
    setting_roles = dict.fromkeys((*PARAMETERS, "length"), PHYSICAL)
    differentiability = GradientStatus.PIECEWISE_KINK
    gradient = GradientRecord()
    references = ()
    builds = {"": ()}
    regions = ("body",)
    region_shapes = {"body": "softtissue.headed_tip"}
    neck_radius: Annotated[Quantity, Length]
    head_radius: Annotated[Quantity, Length]
    head_length: Annotated[Quantity, Length]
    fillet: Annotated[Quantity, Length]
    length: Annotated[Quantity, Length]

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found = [f"{n} is {settings[n]!r} m, not a length above zero" for n in (*PARAMETERS, "length")
                 if not _is_positive_length(settings[n])]
        if found:
            return found
        neck, head, head_length, fillet, length = (_length(settings, n) for n in (*PARAMETERS, "length"))
        if neck > head:
            found.append(f"the neck ({neck} m) is wider than the head ({head} m)")
        if 2 * fillet > min(head_length, head):
            found.append(f"the fillet ({fillet} m) is more than half the head's length or radius")
        if length <= head_length:
            found.append(f"the tip's length ({length} m) is not beyond its head's ({head_length} m)")
        return found

    def _parts(self, settings: Mapping[str, Magnitude]) -> tuple[float, float, float, float]:
        neck, head, head_length = (_length(settings, n) for n in ("neck_radius", "head_radius", "head_length"))
        length = _length(settings, "length")
        return math.pi * head * head * head_length, math.pi * neck * neck * (length - head_length), head, neck

    @override
    def volume(self, settings: Mapping[str, Magnitude]) -> float:
        head, neck, _, _ = self._parts(settings)
        return head + neck

    @override
    def inertia_per_mass(self, settings: Mapping[str, Magnitude]) -> Tensor:
        """The two cylinders' about their common axis and across it, about the frame's origin's axis, divided by the
        mass (an approximation: the fillets and the offset of the centroid from the origin are left out)."""
        v_head, v_neck, r_head, r_neck = self._parts(settings)
        length = _length(settings, "length")
        along = (v_head * r_head ** 2 / 2 + v_neck * r_neck ** 2 / 2) / (v_head + v_neck)
        across = along / 2 + length ** 2 / 12
        return (across, 0.0, 0.0), (0.0, across, 0.0), (0.0, 0.0, along)

    @override
    def enclosing_radius(self, settings: Mapping[str, Magnitude]) -> float:
        return math.hypot(_length(settings, "head_radius"), _length(settings, "length"))

    @override
    def radial_reach(self, settings: Mapping[str, Magnitude]) -> float:
        return _length(settings, "head_radius")

    @override
    def region_parameters(self, region: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        if region != "body":
            raise KeyError(region)
        return {name: _length(settings, name) for name in PARAMETERS}
