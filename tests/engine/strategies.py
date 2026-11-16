"""What a soft solid can have: deformation gradients of physical stretch, moduli of gels and tissues, nonnegative
histories. Drawn in the dtype the engine computes in."""
import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st

from mime_softtissue.engine.material import Fibre, Material, shear_and_bulk

DTYPE = jnp.zeros(()).dtype
STRETCH = (0.6, 2.8)  # principal stretches: compression to a 2.8 stretch, the fibrin strip's at fracture [read]


def _floats(lo, hi):
    return st.floats(lo, hi, allow_nan=False, allow_infinity=False, allow_subnormal=False, width=64)


def rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=DTYPE)


@st.composite
def deformation_gradients(draw, stretch=STRETCH):
    """F = R U with U symmetric positive definite (principal stretches in STRETCH along a drawn direction) and R a
    rotation: every F of positive determinant is one."""
    l1, l2 = draw(_floats(*stretch)), draw(_floats(*stretch))
    psi, phi = draw(_floats(-np.pi, np.pi)), draw(_floats(-np.pi, np.pi))
    V = rotation(psi)
    U = V @ np.diag([l1, l2]) @ V.T
    return rotation(phi) @ U


@st.composite
def materials(draw, fibre: bool = False):
    E = draw(_floats(1e3, 1e7))
    nu = draw(_floats(0.0, 0.45))
    mu, kappa = shear_and_bulk(E, nu)
    gc = draw(_floats(0.1, 100.0))
    length = draw(_floats(1e-4, 1e-3))
    fib = Fibre(k1=jnp.asarray(draw(_floats(1e2, 1e5))), k2=jnp.asarray(draw(_floats(0.1, 5.0))),
                angle=jnp.asarray(draw(_floats(0.0, np.pi / 2)))) if fibre else None
    return Material(mu=jnp.asarray(mu), kappa=jnp.asarray(kappa), gc=jnp.asarray(gc), l=jnp.asarray(length),
                    k_res=jnp.asarray(1e-6), stretch_limit=jnp.asarray(2.8), fibre=fib)
