"""``softtissue.cone_tip``: a rigid cone of ``half_angle`` whose apex is rounded by ``tip_radius``, up to a shaft of
``radius``: a **sharp cone** (a wide shaft) or a **needle with a conical point** (the needle's radius as the shaft); and
its registered shape."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, override

from mime.core import register_shape
from mime.registry import PHYSICAL, Drawing, register_geometry
from mime.units import Angle, Length, Magnitude, Quantity

from ..engine import contact as ct
from ._tip import AXIS, EDGES, CylinderTip, ProfileShape, above_zero_problems, revolved_drawing, value


class ConeShape(ProfileShape):
    parameter_names: tuple[str, ...] = ("radius", "tip_radius", "half_angle")

    def indenter(self, parameters: Mapping[str, Any]) -> ct.Indenter:
        return ct.Indenter("cone", radius=parameters["radius"], fillet=parameters["tip_radius"], top=0.0,
                           half_angle=parameters["half_angle"])


SHAPE = register_shape("softtissue.cone_tip", ConeShape(), switches=(AXIS, EDGES))


@register_geometry
@dataclass(frozen=True, kw_only=True)
class ConeTip(CylinderTip):
    """A rigid cone of ``half_angle`` along its frame's z, its apex rounded by ``tip_radius`` at the frame's origin, up
    to a shaft of ``radius``, ``length`` long: a sharp cone, or a needle with a conical point. ``half_angle`` is an
    angle (``deg(20.0)``, ``rad(0.35)``), read in radians."""

    name = "softtissue.cone_tip"
    #: 0.2.0: ``half_angle`` is an angle (MIME's ``Angle``), where 0.1.0 took it as a dimensionless number of radians.
    version = "0.2.0"
    setting_roles = dict.fromkeys(("radius", "tip_radius", "half_angle", "length"), PHYSICAL)
    region_shapes = {"body": "softtissue.cone_tip"}
    radius: Annotated[Quantity, Length]
    tip_radius: Annotated[Quantity, Length]
    half_angle: Annotated[Quantity, Angle]
    length: Annotated[Quantity, Length]

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found = above_zero_problems(settings, ("radius", "tip_radius", "half_angle", "length"))
        if not found and not value(settings, "half_angle") < 0.5 * math.pi:
            found.append("the half-angle is not below a right angle")
        if not found and value(settings, "tip_radius") >= value(settings, "radius"):
            found.append("the apex's rounding is not smaller than the shaft")
        return found

    @override
    def region_parameters(self, region: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        if region != "body":
            raise KeyError(region)
        return {n: value(settings, n) for n in ("radius", "tip_radius", "half_angle")}

    def drawing(self, settings: Mapping[str, Magnitude], segments: int) -> Drawing:
        r, rho, a, length = (value(settings, n) for n in ("radius", "tip_radius", "half_angle", "length"))
        # the rounded apex's tangent point, then the flank to the shaft (the engine's cone: apex at z = rho)
        z_cone = r / math.tan(a) + rho - rho / math.sin(a)  # where the flank meets the shaft, from the lowest point
        touch = (rho * math.cos(a), rho - rho * math.sin(a))
        return revolved_drawing([(0.0, 0.0), touch, (r, z_cone), (r, max(length, z_cone * 1.01))], segments,
                                "the cone tip, its apex rounded")
