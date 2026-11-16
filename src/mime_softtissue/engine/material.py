"""The material law: a free-energy density of a compressible neo-Hookean solid, strain-stiffened by Yeoh's terms, with
optional fibre families, written in the deformation's invariants so that it does not know the dimension of the build that
feeds it (plane strain, incompressible plane stress, axisymmetric; a 3-D kinematics supplies the same invariants).

    psi(J, I1, I4) = g(d) psi_plus + psi_minus,                        g(d) = (1 - d)^2 + k_res
    psi_iso   = (mu / 2) x + c2 x^2 + c3 x^3,                          x = J^(-2/3) I1 - 3 >= 0
    psi_vol   = (kappa / 4) (J^2 - 1 - 2 ln J)
    psi_fibre = sum over families of (k1 / 2 k2) (exp(k2 <I4 - 1>^2) - 1)
    psi_plus  = psi_iso + psi_fibre + [J >= 1] psi_vol,                psi_minus = [J < 1] psi_vol

Damage degrades the isochoric, fibre and dilatational parts, never volumetric compression. With c2 = c3 = 0 the law is
neo-Hookean; the Yeoh terms are polynomials in x, so every derivative is smooth at the undeformed state (an Ogden law's
are not).

Viscoelasticity (optional, ``Material.prony``): Prony terms on the isochoric response, each a Maxwell branch with an
internal strain Gamma_i (symmetric, at every quadrature point, carried by the solver's state as the history is):

    psi_i = mu_i |E^ - Gamma_i|^2,     mu_i = beta_i mu,    E^ = kappa (Cb - Cb^-1) / 4,    Cb = J^(-2/3) C
    kappa^2 = (1 + (2 c2 / mu) x + (2 c3 / mu) x^2) (x + delta) / (x* + delta),   x* = |Cb - Cb^-1|^2 / 8,
    delta = 1e-6, so that mu_i |E^|^2 = beta_i psi_iso(x) at every deformation (to delta): a step strain's
    overstress is beta_i times the isochoric response, decaying as exp(-t / tau_i) (Fung's quasi-linear form).
    (E^ has the direction of (Cb - Cb^-1) / 4, the mean of the Green and Almansi strains, and the size that makes
    the branch the Yeoh law's; x / x* -> 1 at small strain.)
    phi_i = eta_i |dGamma_i/dt|^2,     eta_i = mu_i tau_i   (the dissipation potential; tau_i the relaxation time)

Each step is an incremental variational update: the solver minimises the elastic energy plus dt phi_i, and the
internal strains are updated locally in closed form (backward Euler, ``r_i = dt / tau_i``):

    Gamma_i <- Gamma_i + r_i / (1 + r_i) (E^ - Gamma_i),   the step's reduced potential mu_i / (1 + r_i) |E^ - Gamma_i|^2

so the displacement solve stays an energy minimisation and every derivative is smooth (polynomials in C and C^-1; no
eigenvalues). The branches are degraded by g(d) with the isochoric part (``energy_parts`` adds them to psi_plus). Only
the stored elastic energy drives the fracture (the long-term part's psi_plus and the branches' stored psi_i after the
step's update), never the dissipation: the standard choice. The deformation enters as the 3-D right Cauchy-Green tensor
C in Voigt order (11, 22, 33, 12, 13, 23), which every kinematics can supply (plane strain C33 = 1, incompressible plane
stress C33 = det(C_2D)^-1, axisymmetric C33 = F_tt^2, 3-D the full tensor); every function broadcasts over leading axes.

Gent's stiffening (optional, ``Material.gent_jm``): the neo-Hookean term (mu / 2) x is replaced by

    psi_gent = -(mu Jm / 2) ln(1 - x / Jm)          for x <= x* = Jm (1 - GENT_CUT),
    psi_gent(x*) + psi_gent'(x*) (x - x*)            beyond (continued linearly: C1, finite everywhere),

the same shear modulus mu at small strain, stiffening without bound as x nears Jm. The continuation keeps the energy
finite, so a damaged band can still open past the limit (k_res times a locking law would hold a crack shut), and a
Newton trial step past the limit reads a finite energy; an undamaged point reaches it only at a stress 1 / GENT_CUT
times the neo-Hookean one at the same x. Invariant-based like the rest (no eigenvalues); Yeoh's c2, c3 still add.

Gent's term is refused (``MaterialRefused``) with two options it would make wrong: Prony branches (``branch_strain``
sizes them to the Yeoh law's slope, not Gent's; code ``softtissue.gent-viscous``), and the 3-D engine's
``residual="neo"`` (it strips Yeoh's terms past full damage, not Gent's excess; ``softtissue.gent-neo-residual``).
No MIME import."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


class Fibre(NamedTuple):
    """Fibre families at +-``angle`` (radians, in the reference plane) to the x axis, taking no compression."""

    k1: Any
    k2: Any
    angle: Any


class Prony(NamedTuple):
    """Prony terms on the isochoric response: ``beta`` (n,) each branch's shear modulus over ``mu`` (the long-term
    one), ``tau`` (n,) its relaxation time (s). A step strain's isochoric stress relaxes from about (1 + sum beta) to 1
    times the long-term one, the i-th excess as exp(-t / tau_i)."""

    beta: Any
    tau: Any


class ViscousStep(NamedTuple):
    """What a load step's energy reads of the viscous branches: the internal strains at its start, ``gamma``
    (..., n, 6) at the quadrature points, and the step's time ``dt`` (s). ``dt`` = 0 reads the stored energy."""

    gamma: Any
    dt: Any


class Material(NamedTuple):
    """The numbers a step reads (a pytree: derivatives reach every field). ``gc`` and ``l`` are the fracture
    model's toughness and length scale; ``stretch_limit`` is read only by the element-deletion control. ``c2``, ``c3``:
    Yeoh's stiffening (Pa; C1 = mu / 2). ``prony``: the viscous branches (None: elastic). ``gent_jm``: Gent's limit
    on x (None: no Gent term)."""

    mu: Any
    kappa: Any
    gc: Any
    l: Any
    k_res: Any
    stretch_limit: Any
    fibre: Fibre | None = None
    c2: Any = 0.0
    c3: Any = 0.0
    prony: Prony | None = None
    gent_jm: Any = None


class MaterialRefused(ValueError):
    """A combination of the law's options that it would compute wrongly, refused instead; ``code`` names which."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{message} ({code})")


def refuse_gent_with_prony(mat: Material) -> None:
    """Refuses Gent's term with Prony branches: the branches are sized to the Yeoh law's slope (``branch_strain``),
    which does not follow Gent's 1 / (1 - x / Jm)."""
    if mat.gent_jm is not None and mat.prony is not None:
        raise MaterialRefused("softtissue.gent-viscous",
                              "Gent's stiffening (gent_jm) with Prony branches: the branches follow the Yeoh law's slope, "
                              "not Gent's, so the viscous stress would be wrong")


#: Where Gent's logarithm hands over to its linear continuation: 1 - x* / Jm (the stress there is 1 / GENT_CUT times
#: the neo-Hookean one at the same x).
GENT_CUT = 1e-3


def gent_energy(x: jax.Array, mu: Any, jm: Any) -> jax.Array:
    """Gent's isochoric energy at ``x`` = J^(-2/3) I1 - 3, continued linearly past x* = jm (1 - GENT_CUT)."""
    x_star = jm * (1.0 - GENT_CUT)
    inside = -0.5 * mu * jm * jnp.log1p(-jnp.minimum(x, x_star) / jm)
    return inside + (0.5 * mu / GENT_CUT) * jnp.maximum(x - x_star, 0.0)


def shear_and_bulk(young: Any, poisson: Any) -> tuple[Any, Any]:
    """Shear and bulk moduli (3-D) from Young's modulus and Poisson's ratio."""
    return young / (2 * (1 + poisson)), young / (3 * (1 - 2 * poisson))


def energy_parts(J: jax.Array, I1: jax.Array, fibre_invariants: tuple[jax.Array, ...],
                 mat: Material) -> tuple[jax.Array, jax.Array]:
    """(psi_plus, psi_minus) at the invariants ``J``, ``I1`` and one ``I4`` a fibre family; infinite where J <= 0."""
    ok = J > 0
    Js = jnp.where(ok, J, 1.0)
    x = Js ** (-2.0 / 3.0) * I1 - 3.0
    base = 0.5 * mat.mu * x if mat.gent_jm is None else gent_energy(x, mat.mu, mat.gent_jm)
    iso = base + mat.c2 * x ** 2 + mat.c3 * x ** 3
    vol = 0.25 * mat.kappa * (Js ** 2 - 1.0 - 2.0 * jnp.log(Js))
    fibre = jnp.zeros((), dtype=J.dtype)
    if mat.fibre is not None:
        for I4 in fibre_invariants:
            stretch = jnp.maximum(I4 - 1.0, 0.0)
            fibre = fibre + mat.fibre.k1 / (2 * mat.fibre.k2) * (jnp.exp(mat.fibre.k2 * stretch ** 2) - 1.0)
    plus = iso + fibre + jnp.where(Js >= 1.0, vol, 0.0)
    minus = jnp.where(Js < 1.0, vol, 0.0)
    bad = jnp.asarray(jnp.inf, dtype=J.dtype)
    return jnp.where(ok, plus, bad), jnp.where(ok, minus, bad)


def degradation(d: Any, k_res: Any) -> Any:
    """g(d), the share of the undamaged stiffness left at damage ``d``."""
    return (1.0 - d) ** 2 + k_res


# --- viscoelastic branches ------------------------------------------------------------------------------------------
VOIGT = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
_SHEAR = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)
#: Regularises the ratio x / x* (both vanish quadratically at the identity): its effect on the branch energy is below
#: 1e-6 relative at stretches over 1.01, and it keeps every derivative finite (float32 included).
BRANCH_DELTA = 1e-6  # the weights of a Voigt vector's squared norm (off-diagonals count twice)


def voigt(C: jax.Array) -> jax.Array:
    """A symmetric (..., 3, 3) tensor as (..., 6) in the order (11, 22, 33, 12, 13, 23)."""
    return jnp.stack([C[..., i, j] for i, j in VOIGT], axis=-1)


def squared_norm(a: jax.Array) -> jax.Array:
    """|A|^2 = A : A of a symmetric tensor in Voigt order (..., 6)."""
    return jnp.sum(jnp.asarray(_SHEAR, dtype=a.dtype) * a * a, axis=-1)


def _determinant(c: jax.Array) -> jax.Array:
    c11, c22, c33, c12, c13, c23 = (c[..., k] for k in range(6))
    return c11 * (c22 * c33 - c23 * c23) - c12 * (c12 * c33 - c23 * c13) + c13 * (c12 * c23 - c22 * c13)


def _inverse(c: jax.Array, det: jax.Array) -> jax.Array:
    """The inverse of a symmetric tensor in Voigt order, by its cofactors (a polynomial over ``det``)."""
    c11, c22, c33, c12, c13, c23 = (c[..., k] for k in range(6))
    cof = jnp.stack([c22 * c33 - c23 * c23, c11 * c33 - c13 * c13, c11 * c22 - c12 * c12,
                     c13 * c23 - c12 * c33, c12 * c23 - c13 * c22, c12 * c13 - c11 * c23], axis=-1)
    return cof / det[..., None]


def branch_strain(c: jax.Array, mat: Material) -> jax.Array:
    """E^ = kappa (Cb - Cb^-1) / 4 from C in Voigt order (..., 6), sized so that mu |E^|^2 / 2 is the Yeoh isochoric
    energy (``BRANCH_DELTA`` regularises x / x* at the identity); zero where det C <= 0 (the law's energy is infinite
    there anyway)."""
    refuse_gent_with_prony(mat)
    det = _determinant(c)
    ok = det > 0
    det_s = jnp.where(ok, det, 1.0)
    eye = jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0, 0.0), dtype=c.dtype)
    cb = jnp.where(ok[..., None], c * det_s[..., None] ** (-1.0 / 3.0), eye)
    x = cb[..., 0] + cb[..., 1] + cb[..., 2] - 3.0
    inv = _inverse(cb, _determinant(cb))
    diff = cb - inv
    x_star = squared_norm(diff) / 8.0
    shape = 1.0 + (2.0 * mat.c2 / mat.mu) * x + (2.0 * mat.c3 / mat.mu) * x * x
    kappa2 = shape * (jnp.maximum(x, 0.0) + BRANCH_DELTA) / (x_star + BRANCH_DELTA)
    kappa = jnp.sqrt(jnp.maximum(kappa2, 1e-12))
    return (0.25 * kappa)[..., None] * diff


def _ratios(mat: Material, dt: Any, dtype: Any) -> tuple[jax.Array, jax.Array]:
    assert mat.prony is not None
    beta = jnp.asarray(mat.prony.beta, dtype=dtype)
    r = jnp.asarray(dt, dtype=dtype) / jnp.asarray(mat.prony.tau, dtype=dtype)
    return beta, r


def viscous_energies(c: jax.Array, gamma: jax.Array, mat: Material, dt: Any) -> tuple[jax.Array, jax.Array]:
    """The branches' (incremental potential, stored energy) of a step of ``dt`` from the internal strains ``gamma``
    (..., n, 6) to the deformation ``c`` (..., 6): sum mu_i / (1 + r_i) |E^ - Gamma_i|^2 (what the step minimises,
    the dissipation potential included) and sum mu_i / (1 + r_i)^2 |E^ - Gamma_i|^2 (stored after the update).
    ``dt`` = 0: both are the stored energy at ``gamma``."""
    beta, r = _ratios(mat, dt, c.dtype)
    a = squared_norm(branch_strain(c, mat)[..., None, :] - gamma)  # (..., n)
    k = mat.mu * beta / (1.0 + r)
    return jnp.sum(k * a, axis=-1), jnp.sum(k / (1.0 + r) * a, axis=-1)


def updated_strains(c: jax.Array, gamma: jax.Array, mat: Material, dt: Any) -> jax.Array:
    """The internal strains after a step of ``dt`` to ``c``: Gamma + r / (1 + r) (E^ - Gamma)."""
    _, r = _ratios(mat, dt, c.dtype)
    e = branch_strain(c, mat)[..., None, :]
    return gamma + (r / (1.0 + r))[:, None] * (e - gamma)


def branch_dissipation(c_start: jax.Array, c_end: jax.Array, gamma_start: jax.Array, gamma_end: jax.Array,
                       mat: Material) -> jax.Array:
    """The branches' dissipation over a step (..., ): the driving force 2 mu_i (E^ - Gamma_i) times the internal
    strain's increment, by the trapezoid rule over the step (exact for the quadratic branch along a straight path
    in (E^, Gamma)), sum over i."""
    assert mat.prony is not None
    beta = jnp.asarray(mat.prony.beta, dtype=c_end.dtype)
    a0 = branch_strain(c_start, mat)[..., None, :] - gamma_start
    a1 = branch_strain(c_end, mat)[..., None, :] - gamma_end
    inc = gamma_end - gamma_start
    w = jnp.asarray(_SHEAR, dtype=c_end.dtype)
    return jnp.sum(mat.mu * beta * jnp.sum(w * inc * (a0 + a1), axis=-1), axis=-1)


def initial_strains(shape: tuple[int, ...], mat: Material, dtype: Any = jnp.float64) -> jax.Array:
    """Zero internal strains (the undeformed, relaxed state) for ``shape`` quadrature points: shape + (n, 6)."""
    assert mat.prony is not None
    refuse_gent_with_prony(mat)
    return jnp.zeros(tuple(shape) + (len(mat.prony.tau), 6), dtype=dtype)
