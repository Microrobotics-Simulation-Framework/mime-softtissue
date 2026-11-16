"""The 3-D puncture engine (``engine.puncture3d``, ``kinematics.solid3d``, ``solvers.multigrid3d``): the 3-D Yeoh energy
against its closed form, the mean-dilatation element against the pointwise one where they must agree, and the engine
against the 3-D spikes' own solver (3D-COST's ``all2`` at h 0.5 mm, its recorded forces), with the energy balance."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime_softtissue.engine import puncture3d as P
from mime_softtissue.engine.kinematics import solid3d as S
from mime_softtissue.engine.material import Material, shear_and_bulk

jax.config.update("jax_enable_x64", True)

# 3D-COST's all2 at h 0.5 mm (cost3d.py --h 0.5 --ratio 1.5 --cap 16 --fine-xy 0.8 --fine-z 1.0 --inexact 1e-3
# --adaptive --max-step 0.04 --mp), N, at depths 0.02 ... 0.44 mm [measured by the spike, its c3_h0.5.json]
ALL2_H05 = {0.02: 0.0003619710258814965, 0.06: 0.0010604341954791553, 0.1: 0.0016691391472632008,
            0.14: 0.0021507016101468533, 0.18: 0.002508780488223449, 0.22: 0.002771533324487269,
            0.26: 0.002992976229459417, 0.3: 0.003221431638281736, 0.34: 0.0033786737150867134,
            0.38: 0.003477424190379325, 0.4: 0.00350895314068492, 0.44: 0.003545264640331876}


def _homogeneous(F: np.ndarray, grid: S.Grid3) -> jax.Array:
    X = np.stack(np.meshgrid(grid.x, grid.y, grid.z, indexing="ij"), axis=-1)
    return jnp.asarray(X @ (F - np.eye(3)).T)


@pytest.mark.parametrize("volumetric", ["mean", "gauss"])
@pytest.mark.parametrize("stretch", [1.3, 2.0, 3.5])
def test_yeoh_energy_is_the_closed_form_under_incompressible_uniaxial_stretch(volumetric: str, stretch: float) -> None:
    grid = S.uniform_grid((2, 3, 2), (1e-3, 2e-3, 1.5e-3))
    c1, c2 = 361.0, 438.0
    mat = Material(mu=2 * c1, kappa=50 * 2 * c1, gc=1.0, l=1e-4, k_res=0.0, stretch_limit=10.0, c2=c2)
    F = np.diag([stretch, stretch ** -0.5, stretch ** -0.5])
    u = _homogeneous(F, grid)
    g = jnp.ones((2, 2, 2) + grid.shape)
    x = stretch ** 2 + 2 / stretch - 3
    exact = (c1 * x + c2 * x ** 2) * 1e-3 * 2e-3 * 1.5e-3
    assert float(S.solid_energy(u, g, grid, mat, volumetric)) == pytest.approx(exact, rel=1e-12)
    # the control: the law without its stiffening term is not the closed form
    assert float(S.solid_energy(u, g, grid, mat._replace(c2=0.0), volumetric)) < 0.9 * exact


def test_mean_dilatation_equals_pointwise_where_J_is_uniform_and_differs_where_it_is_not() -> None:
    grid = S.uniform_grid((2, 2, 2), (1e-3, 1e-3, 1e-3))
    mu, kappa = shear_and_bulk(50e3, 0.45)
    mat = Material(mu=mu, kappa=kappa, gc=1.0, l=1e-4, k_res=0.0, stretch_limit=10.0, c2=0.3 * mu)
    F = np.array([[1.2, 0.1, 0.0], [0.05, 0.9, 0.02], [0.0, 0.1, 1.1]])
    u = _homogeneous(F, grid)
    g = jnp.full((2, 2, 2) + grid.shape, 0.7)
    a, b = (float(S.solid_energy(u, g, grid, mat, v)) for v in ("mean", "gauss"))
    assert a == pytest.approx(b, rel=1e-12)
    # a non-uniform J (a bulge at the centre node): the two elements disagree, so the comparison above can fail
    u2 = u.at[1, 1, 1, 2].add(2e-4)
    a2, b2 = (float(S.solid_energy(u2, g, grid, mat, v)) for v in ("mean", "gauss"))
    assert abs(a2 - b2) > 1e-6 * abs(b2)


def test_the_engine_reproduces_3d_costs_all2_and_balances_its_energy() -> None:
    """The engine at 3D-COST's all2 settings (neo-Hookean gel, Gauss-point volumetric, no Anderson, no limit) gives the
    spike's forces at its first three depths to 1e-10, and its energy balance closes within 2 % (the whole 0.44 mm run
    matches at all twelve depths to 7e-13 [measured, the laptop], but takes four minutes)."""
    mu, kappa = shear_and_bulk(50e3, 0.4)
    mat = Material(mu=mu, kappa=kappa, gc=1.0, l=0.1e-3, k_res=1e-6, stretch_limit=10.0)
    st = P.Settings(volumetric="gauss", anderson=0, stagger_cap=0, newton_cap=0, first_step=0.04e-3, min_step=0.01e-3,
                    damage_bound="clip")
    run = P.run_puncture(P.Block(h=0.5e-3), mat, P.blunt_punch(0.5e-3, 0.05e-3), 0.10e-3, st, frames=2,
                         output_depths=(0.02e-3, 0.10e-3))
    got = {round(float(z) * 1e3, 4): float(f) for z, f in zip(run.rows["depth_m"], run.rows["force_N"])}
    assert sorted(got) == [0.02, 0.06, 0.1]
    for z, f in got.items():
        assert f == pytest.approx(ALL2_H05[z], rel=1e-10), z
    assert run.energy_balance() < 0.02
    # the surface: every boundary node once, its frames at the run's depths
    assert len(run.surface.points) == len(run.surface.depth) and run.surface.depth[-1] == pytest.approx(0.10e-3)
    pts, faces, dmg, display = P.mirrored(run.surface)
    assert pts.shape[1] == 4 * run.surface.rest.shape[0] and display.sum() == 3 * (~display).sum()


def test_the_damage_bounds_give_a_minimiser_where_the_clip_does_not() -> None:
    """One element driven hard (the node under a sharp tip): the unconstrained damage solve leaves [0, 1], so its clip is
    feasible but not stationary inside (0, 1) (the energy's gradient there is not zero); the active set's field is feasible,
    stationary inside, and has the lower energy. Where nothing leaves [0, 1], the two are the same field."""
    mu, kappa = shear_and_bulk(50e3, 0.4)
    mat = Material(mu=mu, kappa=kappa, gc=1.0, l=0.1e-3, k_res=1e-6, stretch_limit=10.0)
    blk = P.Block(half_width=1e-3, height=1e-3, h=0.25e-3, fine_xy=0.5e-3, fine_z=0.5e-3)
    tip = P.sharp_cone(0.35, 0.05e-3)
    bound = P.Puncture3D(blk, mat, tip, P.Settings(mixed=False))
    clip = P.Puncture3D(blk, mat, tip, P.Settings(mixed=False, damage_bound="clip"))
    shape = (2, 2, 2) + bound.grid.shape
    gc, l = bound.mat.gc, bound.mat.l
    energy = lambda d, H: float(S.damage_energy(d, H, bound.grid, gc, l))
    hard = jnp.zeros(shape).at[:, :, :, 0, 0, -1].set(1e4 * gc / l)
    d_b, _ = jax.jit(bound.solve_damage)(hard)
    d_c, _ = jax.jit(clip.solve_damage)(hard)
    assert float(jnp.max(d_c)) == 1.0 and float(jnp.max(d_b)) <= 1.0
    assert energy(d_b, hard) < energy(d_c, hard)

    def stationarity(d):  # the largest |dE/dd| over nodes strictly inside (0, 1), against the largest overall
        g = jax.grad(lambda x: S.damage_energy(x, hard, bound.grid, gc, l))(d)
        inside = (d > 1e-9) & (d < 1 - 1e-9)
        return float(jnp.max(jnp.where(inside, jnp.abs(g), 0.0)) / jnp.max(jnp.abs(g)))

    assert stationarity(d_b) < 1e-6 < 1e-3 < stationarity(d_c)
    mild = jnp.full(shape, 0.1 * gc / l)
    assert jnp.allclose(jax.jit(bound.solve_damage)(mild)[0], jax.jit(clip.solve_damage)(mild)[0], rtol=0, atol=0)
