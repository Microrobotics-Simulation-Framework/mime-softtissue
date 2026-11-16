"""The new tips' signed distances (``engine.contact``: the cone and the headed tip, the barb's stand-in) against their
parametric surfaces: a point drawn on a face lies on the zero set, and the point moved out along the face's normal by
``t`` (less than the gap to any other face) is at distance ``t``. Draws stay at the tips' own scales (Hypothesis
shrinks towards extremes; a face point is drawn from its face's span, never from the medial set).

Planted errors (``PLANTS``) the property must catch every run (``test_tip_plant_rates.py`` measures the rates): the
cone's half-angle read as its complement; the head's length halved in the distance; the apex's rounding left out."""
from __future__ import annotations

import contextlib
import math

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mime_softtissue.engine import contact as ct

TOP = 5e-3
SCALE = 1e-3  # m: the tips' size; tolerances are this times float64's rounding, a hundredfold
TOL = 100 * float(np.finfo(np.float64).eps) * SCALE


def cone(half_angle=math.radians(20.0), radius=0.5e-3, fillet=0.05e-3):
    return ct.Indenter("cone", radius=radius, fillet=fillet, top=TOP, half_angle=half_angle)


def headed(neck=0.4e-3, head=0.6e-3, length=0.2e-3, fillet=0.05e-3):
    return ct.Indenter("headed", radius=neck, fillet=fillet, top=TOP, head_radius=head, head_length=length)


def cone_face(ind, s, depth):
    """A point of the cone's straight side at a fraction ``s`` of its span (from where the apex's rounding ends to
    the shaft), and the side's outward normal."""
    a, rho = ind.half_angle, ind.fillet
    apex = TOP - depth + rho  # the sharp cone's apex, offset by the rounding
    z_low = -rho * math.sin(a)  # where the apex's rounding meets the offset side (height above the sharp apex)
    z_high = (ind.radius - rho / math.cos(a)) / math.tan(a)  # where the offset side reaches the shaft's radius
    z = z_low + s * (z_high - z_low)
    R = z * math.tan(a) + rho / math.cos(a)
    return R, apex + z, (math.cos(a), -math.sin(a))


def headed_faces(ind, s, depth):
    """A point of each of the headed tip's four flat faces at a fraction ``s`` of its span, with its outward normal:
    the head's bottom, the head's side, the shoulder (the head's top), the neck's side."""
    tip, rho = TOP - depth, ind.fillet
    rh, rn, lh = ind.head_radius, ind.radius, ind.head_length
    return [
        (s * (rh - rho), tip, (0.0, -1.0)),
        (rh, tip + rho + s * (lh - 2 * rho), (1.0, 0.0)),
        (rn + 0.05e-3 + s * (rh - rho - rn - 0.05e-3), tip + lh, (0.0, 1.0)),  # clear of the neck by more than t
        (rn, tip + lh + 0.05e-3 + s * 1.0e-3, (1.0, 0.0)),
    ]


def phi(ind, R, Z, depth):
    return float(ct.signed_distance(ind, jnp.asarray(R), jnp.asarray(Z), jnp.asarray(depth)))


def property_holds(ind_cone, ind_head, s, t, depth) -> list[str]:
    """What fails of the property at one draw (an empty list: it holds)."""
    found = []
    R, Z, n = cone_face(ind_cone, s, depth)
    for off, want in ((0.0, 0.0), (t, t)):
        got = phi(ind_cone, R + off * n[0], Z + off * n[1], depth)
        if abs(got - want) > TOL:
            found.append(f"cone at s {s:.3f}, offset {off:.2e}: phi {got:.6e}, not {want:.6e}")
    for k, (R, Z, n) in enumerate(headed_faces(ind_head, s, depth)):
        for off, want in ((0.0, 0.0), (t, t)):
            got = phi(ind_head, R + off * n[0], Z + off * n[1], depth)
            if abs(got - want) > TOL:
                found.append(f"headed face {k} at s {s:.3f}, offset {off:.2e}: phi {got:.6e}, not {want:.6e}")
    return found


DRAWS = dict(s=st.floats(0.0, 1.0), t=st.floats(1e-6, 0.04e-3), depth=st.floats(0.0, 1.5e-3))


@settings(max_examples=100, deadline=None)
@given(**DRAWS)
def test_the_tips_signed_distances_follow_their_faces(s, t, depth):
    found = property_holds(cone(), headed(), s, t, depth)
    assert not found, found


# --- plants (monkeypatches of the distance's module functions, looked up at call time) -----------------------------
@contextlib.contextmanager
def _patched(name, fn):
    old = getattr(ct, name)
    setattr(ct, name, fn)
    try:
        yield
    finally:
        setattr(ct, name, old)


def complement_angle():
    real = ct._cone
    return _patched("_cone", lambda R, z, a: real(R, z, math.pi / 2 - a))


def head_length_halved():
    real = ct._rounded_box
    return _patched("_rounded_box", lambda R, Z, hr, z0, hz, r: real(R, Z, hr, z0 - hz / 2, hz / 2, r))


PLANTS = {"complement_angle": complement_angle, "head_length_halved": head_length_halved}


def test_closed_form_tube_signed_distance_unchanged():
    """Round 3's tube and punch are untouched by the new kinds: two points each, by hand."""
    tube = ct.Indenter("tube", radius=0.6e-3, fillet=0.0, top=TOP, inner=0.4e-3)
    assert abs(phi(tube, 0.5e-3, TOP - 1e-3 + 0.1e-3, 1e-3) + 0.1e-3) < TOL  # the rounded end's centre, rho inside
    punch = ct.Indenter("punch", radius=0.5e-3, fillet=0.05e-3, top=TOP)
    assert abs(phi(punch, 0.0, TOP - 1e-3 - 0.1e-3, 1e-3) - 0.1e-3) < TOL
