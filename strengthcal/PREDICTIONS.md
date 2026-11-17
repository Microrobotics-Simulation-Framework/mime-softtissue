# STRENGTH-CAL predictions (spike branch spike-strength-cal; never merged to cleanup)

Spike G2: what the strength-surface phase field (`engine.fracture.strength`) costs at a realistic regularisation length
(l 20 to 50 um, h = l/2), and whether one strength calibrated on flat punches predicts spherical-tip punctures. The source
paper's numbers, figures and digitised values are not kept in this repository; comparisons against them are in the spike's
report. Here: predictions (model-side numbers) and our own computed results. Labels: [estimate] unless said.

Driver: `strengthcal/run.py` (the axisymmetric engine called directly; NUCLEATE's needle construction; neo-Hookean, nu 0.49,
volumetric term at the element's mean dilatation; Gc 27.6 J/m^2; s_hs/s_ts 1.5; block 50 mm; grid uniform at h within
`--fine-r` of the axis and `--fine-z` of the top, graded by 1.25 to cells of at most `--coarse`). `--no-lag` drops the
lagged contact pressure that the GelBlock construction carries with zero friction (it changes no fixed point, only the
staggered iteration count): a declared deviation, its effect measured by the `-lag` arms.

Validity of the formulation [computed, `coefficients`' formulas, this engine's law]: the strength surface is faithful while
l < 3 Gc / (16 W_ts), and beta2 (eq. 16) turns positive past about 0.6 of that length. For this neo-Hookean law at nu 0.49
the ceiling on s_ts is about 5-6 G' at l 50 um, 6-8 G' at l 20 um (G' 3.66 and 6.2 kPa alike, in units of G'), and about
8-10 G' at l 10 um: W_ts grows as s_ts^2 at large stretch (stretch 14 at 10 G', 48 at 20 G').

## Round 1: cost at l 50, 25, 20 um, and the strength a flat punch needs (written 2026-11-17 06:47 CET, `date`, before any arm ran)

Arms: sphere R 0.5 mm, G' 3.66 kPa, s_ts 18.3 kPa (5 G', inside the validity ceiling at all three l), fine zone r <= 1.0 mm,
top 2.0 mm, coarse 1.0 mm, steps of 0.05 mm to 6.5 mm, stopped at a 50 % drop: `c50`, `c25`, `c20` (no lag), `c50-lag`,
`c20-lag` (the GelBlock's lag); elastic twins `e50`, `e25`, `e20` (h 0.025, 0.0125, 0.01; strength fields recorded); flat
punch, G' 6.2 kPa, R 0.359 mm, elastic, to 7 mm (19.5 R): fillet 0.02 mm at h 0.0125 and 0.025, fillet 0.07 mm at h 0.0125.

- **P1.1** Mean s/step before nucleation on a 4-vCPU runner, no lag: l 50: 10-40 s; l 25: 40-150 s; l 20: 70-300 s.
- **P1.2** The lag costs 2-5 times the no-lag step before nucleation; forces agree within 0.1 % before nucleation.
- **P1.3** Peak memory: l 50 under 1.5 GB; l 25 1-3 GB; l 20 1.5-4 GB.
- **P1.4** s_ts 5 G' punctures (a drop of at least 30 % with crack area) before 6.5 mm at all three l, at Fc/(mu R^2) 3-20,
  and Fc falls as l falls (l 50 to 20: by 10-40 %).
- **P1.5** The elastic twins show no drop above 2 % to 6.5 mm at h <= 0.025.
- **P1.6** Flat punch, elastic neo-Hookean: P/(G' R^2) at d = 13 R is 60-120.
- **P1.7** The largest equivalent tensile strength seq (the s_ts at which a Gauss point sits on the surface, Biot,
  s_hs/s_ts 1.5) in the flat-punch gel at d = 13 R is 15-60 G': above the ceiling at l 20 um, so a strength that makes the
  flat punch puncture as deep as d = 13 R cannot be represented faithfully at l 20 um with this law.
- **P1.8** That seq differs by more than 20 % between fillet 0.02 and 0.07 mm at the same depth, and by more than 10 %
  between h 0.025 and 0.0125 at fillet 0.02 (the corner sets it).

### Round 1 status at 07:22 CET (`date`; first peek of run 44027009704, cancelled at 07:25), read before round 1b was written
- **The first load step never converged at l 25 and 20 um** (h 12.5 and 10 um): 0.05 mm of sudden penetration is 4-5 cells
  under a stiff nodal penalty; no row in 28 min. At l 50 um it converged after 3 staggered passes (67 Newton).
- **The elastic twins and flat-punch arms are invalid:** without the lag an elastic step is one Newton solve, capped at 40;
  the first step capped and every later one diverged (J below 0, forces of newtons). No reading is taken from them.
- **c50 (no lag) against c50-lag:** forces equal at every step to 1.30 mm (the lag changes no fixed point, P1.2's second half
  held); the lag costs 1.4-2.5 times the step (P1.2 held). No-lag median 9.7 s a step (P1.1 held at l 50), RSS 1.23 GB.
- **Nucleation at 1.20-1.30 mm, on the axis element under the apex** (the strength fields' maximum there from 0.85 mm; J there
  dips 0.85 to 0.50 and back to 0.84 between 1.15 and 1.30 mm), at 11.2 mN; then c50-lag went NaN at 1.35 mm (590 Newton in one
  step) and c50 did not finish the 1.35 mm step in over 20 min.

## Round 1b: the same costs with a ramped start and step cutting (written 2026-11-17 07:27 CET, `date`, before any arm ran)

Driver: `--ramp 4` (the first step is 1/16 of the step, doubling back), `--cut 5` (a step whose staggered loop or a Newton
solve reaches its cap, or whose force is not finite, is retried from the last accepted state at half the increment, down to
1/32 of the step after the ramp), `--max-newton 60` and `--max-staggered 40` for the strength arms, `--max-newton 100` for
the elastic ones; no lag throughout. Arms as round 1 (sphere R 0.5, s_ts 5 G' = 18.3 kPa at G' 3.66; elastic twins; flat
punch G' 6.2, R 0.359, elastic, fillet 0.02 at h 0.025, 0.0125, 0.01 and fillet 0.07 at h 0.0125).

- **P1b.1** The ramped first step converges at every l. Median s/step (no lag, 4-vCPU runner) before nucleation: l 50: 5-15 s;
  l 25: 20-60 s; l 20: 40-120 s.
- **P1b.2** Peak RSS: l 50 1.1-1.5 GB; l 25 2-3 GB; l 20 3-4.5 GB.
- **P1b.3** The strength arms nucleate at 1.1-1.4 mm on the axis element under the apex at every l; with the cut, the arm either
  passes to a drop of at least 30 % within 0.5 mm or ends at the cut's floor or NaN (lean: ends, 0.6).
- **P1b.4** The elastic twins reach 6.5 mm with finite forces; the axis element under the apex dips below J 0.6 between 1.0 and
  1.4 mm on every grid, i.e. the dip does not shrink with h (lean 0.6).
- **P1b.5** Flat punch: P1.6-P1.8 restated for these arms.

### A local control read before round 1c (07:36 CET, `date`): the axis element under the apex snaps under mean dilatation
Elastic sphere R 0.5, local, h 0.1 and 0.05 (engine-direct, ramp, no lag), both volumetric treatments, to 1.8 mm:
- **mean:** the first element on the axis under the apex snaps: J there falls from 0.74-0.77 to 0.32-0.42 within one step, at
  0.95 mm (h 0.1) and 1.10 mm (h 0.05) (and 1.20 mm at h 0.025 in round 1's c50); the strength fields' maximum sits on it from
  then (seq 280-3100 kPa by 1.8 mm). The snap moves with h: a discretisation artefact of the element on the axis.
- **gauss:** no snap; J falls smoothly (0.92 to 0.58 at h 0.05 by 1.8 mm); the strength fields' maximum lies off the axis under
  the apex (r 0.175-0.25, 0.2-0.3 mm below the top) to about 1.45 mm, then on the axis 0.5-0.7 mm below; seq agrees between
  h 0.1 and 0.05 to 1-3 % at 0.85-1.3 mm (17 / 28 / 87 kPa at 0.85 / 1.0 / 1.3 mm).
- So round 1b's strength arms (mean) will nucleate on the artefact; their cost still reads, their forces do not.

## Round 1c: full integration, and a stiffening placeholder (written 2026-11-17 07:38 CET, `date`, before any arm ran)
Arms (gauss, no lag, ramp 4, cut 5): strength s_ts 18.3 kPa at l 50 and 20 um (`c50g`, `c20g`); elastic twins h 0.025 and
0.01 (`e50g`, `e20g`); flat punch elastic h 0.0125 fillet 0.02 (`fe-f02-h125g`); and a placeholder Gent law, Jm 9.3 (GENT's
committed fit, set at another modulus: a placeholder, not a calibration), elastic sphere h 0.025 and flat punch h 0.0125
(`e50g-gent`, `fe-f02-h125g-gent`).
- **P1c.1** gauss elastic: J on the axis element under the apex stays above 0.55 to 2 mm at h 0.025 and 0.01; no force drop
  above 2 % to 6.5 mm.
- **P1c.2** s_ts 18.3 kPa nucleates at 0.80-1.0 mm, Fc 7.5-9.5 mN (Fc/(mu R^2) 8-10.5) at both l, off the first axis element.
- **P1c.3** The Gent placeholder carries 5-40 % more force than neo-Hookean at 3-6 mm, and its seq at a given force is lower
  than neo-Hookean's at the same force.
