"""``softtissue.punch_tip``: a rigid cylinder of ``radius`` whose end's edge is rounded by ``fillet``: a **flat punch**
(a small fillet) or a **sphere-tipped needle** (``fillet = radius``, a capsule's end); and its registered shape."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, override

from mime.core import register_shape
from mime.registry import PHYSICAL, Drawing, register_geometry
from mime.units import Length, Magnitude, Quantity

from ..engine import contact as ct
from ._tip import AXIS, EDGES, CylinderTip, ProfileShape, above_zero_problems, revolved_drawing, value


class PunchShape(ProfileShape):
    parameter_names = ("radius", "fillet")

    def indenter(self, parameters: Mapping[str, Any]) -> ct.Indenter:
        return ct.Indenter("punch", radius=parameters["radius"], fillet=parameters["fillet"], top=0.0)


SHAPE = register_shape("softtissue.punch_tip", PunchShape(), switches=(AXIS, EDGES))


@register_geometry
@dataclass(frozen=True, kw_only=True)
class PunchTip(CylinderTip):
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
        found = above_zero_problems(settings, ("radius", "fillet", "length"))
        if not found and value(settings, "fillet") > value(settings, "radius"):
            found.append("the fillet is larger than the radius (a sphere-tipped needle has fillet = radius)")
        if not found and value(settings, "length") <= value(settings, "fillet"):
            found.append("the tip is not longer than its rounded end")
        return found

    @override
    def region_parameters(self, region: str, settings: Mapping[str, Magnitude]) -> dict[str, Magnitude]:
        if region != "body":
            raise KeyError(region)
        return {"radius": value(settings, "radius"), "fillet": value(settings, "fillet")}

    def drawing(self, settings: Mapping[str, Magnitude], segments: int) -> Drawing:
        r, f, length = (value(settings, n) for n in ("radius", "fillet", "length"))
        arc = [(r - f + f * math.sin(a), f - f * math.cos(a)) for a in
               (0.5 * math.pi * k / max(2, segments // 8) for k in range(1, max(2, segments // 8) + 1))]
        start = [(r - f, 0.0)] if r - f > 1e-12 * r else []
        return revolved_drawing([(0.0, 0.0), *start, *arc, (r, length)], segments, "the punch tip, its rounded end")
