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
