# HERTZ-CHECK predictions and lines (written before any arm's result was read)

Written by the HERTZ-CHECK agent (Opus). Time from `date` at the end of this file. A local dev run (dev_m49_h02: tip3d
settings, kappa 50 mu, h 0.2, L 20) was started about a minute before this file was stamped (dev1.pid), for timing; its log had not been read when this file was written.

## Operating point [read before writing]
- The plot in question (notebook cell 34) is the 1 mm flat punch (radius 0.5 mm, fillet 0.05 mm) into
  `softtissue_gel_placeholder`: E 50 kPa, **nu 0.4** (kappa/mu 4.67), Gc 1 J/m^2, l 0.1 mm; quarter block 5 x 5 mm,
  fine_xy 0.8, fine_z 1.0; contact at the nodes in every series (both presets keep contact_at="nodes");
  through MIME: fixed 0.04 mm steps; outside MIME (3D-COST): all2 arithmetic with adaptive steps, and a uniform-step baseline.
- So nu (0.45 against kappa 50 mu) and the contact discretisation (nodes against cells) are NOT differences between that
  plot's series; they are differences in the clot-cone comparison. The punch series differ by: the volumetric treatment
  (mean against Gauss), the damage bound (active set against clip), Anderson + caps + verify, and fixed against adaptive steps.
- h 0.2 and 0.1 against l 0.1 mm: the phase field is under-resolved at both grids (h/l 2 and 1).

## The control: Hertz
Sphere-tipped needle R 5 mm, depths 0.01 to 0.05 R (0.05 to 0.25 mm; contact radius a 0.5 to 1.12 mm: a/h 2.5 to 5.6 at
h 0.2, the punch's a/h range), quarter block L = 20 mm (half-width = height), fine_xy 1.5, fine_z 1.0, ratio 1.5, cap 16 h;
neo-Hookean, Gc 1e9 (elastic). Reference: F_H = 4/3 E* sqrt(R) delta^1.5, E* = 2 mu / (1 - nu) (the brief's "16/9 E*" is
16/9 E for nu = 0.5; 16/3 mu is right), times the finite block's factor (measured: L 10/20/40 at h 0.2, extrapolated in 1/L;
cross-checked against Dimitriadis et al. 2002's bonded-layer series) and Sneddon's exact-sphere factor.

**Line (stated first):** a configuration is right at the elastic level if its mesh-extrapolated force at delta/R 0.01-0.05 lies
within max(its discretisation error |F_finest - F_inf|, 2 %) of the corrected Hertz force. The 2 % is the allowance for the
small-strain theory against a neo-Hookean solid at delta/R <= 0.05 [estimate].

## Predictions (leans)
- P1 tip3d (mean dilatation), any nu: within 3 % of Hertz at h 0.1, converging at order 1 to 2; sign uncertain (Q1 is stiff,
  nodal contact quantises the contact radius and reads low).
- P2 all2 (Gauss-point volumetric): nu 0.4: +2 to +8 % at h 0.2, converging from above; nu 0.45: +10 to +30 %;
  kappa 50 mu: +40 to +100 % at h 0.2, converging slowly from above (TIP3D measured +106 % on the clot cone).
- P3 nu: at the same mu, Hertz says nu 0.45 reads 7.3 % below kappa 50 mu (E* ratio 3.636/3.922) [derived]; the
  measured gap matches within 2 points. This, not the steps, is most of the 7-12 % "through MIME below the engine" on the cone.
- P4 contact in cells (contact_at="gauss"): reads LOW at small depth on a coarse grid (the top Gauss points sit 0.21 h below
  the surface), by tens of % at delta/R 0.01, h 0.2; converges at order about 1 from below.
- P5 step control (fixed against adaptive): no effect on an elastic indentation (< 1e-5 relative): the Hertz control CANNOT
  discriminate it. Said as such, not as a null result.
- P6 plants: E-for-mu (x 2.98 at kappa 50 mu, x 2.8 at nu 0.4) caught at every grid and preset; dropped symmetry factor
  (quarter: / 4) caught everywhere. Detection rate 100 % predicted (the line's allowance is a few %).
- P7 the punch's 16 % peak gap (tip3d against all2 at h 0.2, engine called directly with MIME's fixed steps): lean: the damage
  bound (active set) gives most of the peak drop, the volumetric treatment the 3-5 % before the peak; Anderson/caps/verify
  under 2 %. Low confidence.
- P8 verdict lean: tip3d's pre-peak forces are the nearer to the truth at nu 0.4 (all2's are high by a few %); neither
  peak is shown right by Hertz (a fracture quantity), and all2's agreement with 3D-COST is agreement of the same arithmetic.
Mon 16 Nov 18:37:11 CET 2026

## Addendum, round 2 (written before round 2's arms; round 1's first nine arms had been read)
Read so far [measured, h 0.2, raw F/F_H, L 20 block not yet corrected]: tip3d nu 0.45 0.99-1.04, all2 nu 0.45 1.04-1.07, all2
nu 0.4 1.02-1.07; the plants read x 2.87-2.98 (E for mu) and / 4.0 (symmetry). Round 2 adds the third grid (h 0.1) at
kappa 50 mu for both presets (the locking case) and the punch's one-at-a-time arms at the plotted grid h 0.1. Leans unchanged
(P2: all2 at kappa 50 mu high and converging from above; P7: the damage bound carries most of the peak gap).
Mon 16 Nov 18:49:36 CET 2026

## Addendum, round 3: the punch's peak under refinement, for each volumetric treatment (written before round 3's arms)
Asked by F0 (resumed 18:51): does the peak converge to the same value for mean dilatation (tip3d) and full integration (all2)?
Read before writing [read: SPIKE_3DCOST, SPIKE_3DSTAGGER, SPIKE_3DPUNCTURE rows; TUTORIAL3D]. Full integration's peak (mN, depth):
h 0.2 11.28 (0.28, the plugin, 0.04 mm steps), h 0.1 10.62 (0.30), h 0.0707 11.03 (0.28; uniform 0.02 mm), h 0.05 11.28 (0.27,
adaptive), and the axisymmetric build h 0.1 / 0.0707 / 0.05 / 0.0354 / 0.025: 10.26 / 10.99 / 11.99 / 11.51 / 11.67 (depth 0.28
to 0.22): not monotone, spread 4 % over the three finest. tip3d through MIME: 8.87 (0.20) at h 0.2 and 0.1 (0.4 % apart).
Arms: tip3d (mean dilatation) on the punch at h 0.0707 (0.04 mm steps to 0.34 mm) and h 0.05 (to 0.32 mm; may not land by
21:30); 0.02 mm steps at h 0.2 and 0.1 to 0.28 mm (the step's effect on the sampled peak).
- P9 tip3d's peak at h 0.0707: 8.6 to 10.5 mN at 0.18 to 0.26 mm; lean 0.55 that it rises more than 2 % above 8.87.
- P10 the gap to full integration at h 0.0707 (11.03) stays above 8 %: lean 0.65. So by h/l 0.7 the two treatments do NOT yet
  agree on the peak; both must meet as h/l -> 0 (the cell mean tends to the point value), so a persisting gap says the grids
  are not converged, not which one is right.
- P11 0.02 mm steps move tip3d's sampled peak by less than 2 % at h 0.2 and 0.1 (lean 0.7).
- P12 tip3d at h 0.05 (if it lands): within 5 % of its h 0.0707 peak (lean 0.5).
Mon 16 Nov 18:55:11 CET 2026

## Addendum, round 4: what in the volumetric treatment carries the peak gap (written before round 4's arms)
Read before writing [measured, round 1, h 0.2]: mean dilatation on the punch peaks at 8.83 mN at 0.20 mm (d_max 0.39) and falls
to 1.6 mN at 0.28 (d_max 1.0); full integration peaks at 11.28 at 0.28 (d_max 0.66) and declines gradually (10.58 at 0.44,
d_max 0.99); 5.7 % apart at 0.04 mm. Hertz at kappa 50 mu, h 0.2: full integration 9 to 16 % above mean dilatation. Operating
point: the damage degrades shear but never volumetric compression, so a damaged zone's kappa / (g mu) is 4.67 / g (39 at
d 0.66, 467 at d 0.9): the intact gel's nu 0.4 does not bound the locking there. Also read: the engine scales the input Gc by
1 / (1 + h / 2l) (Gc x 0.5 at h 0.2, x 0.74 at h 0.0707), in every series (the spikes too).
Arms (h 0.2): the history field's source switched alone (mean dilatation with a pointwise source; full integration with the
element-mean source; mean with mean, the identity control); Hertz at kappa 500 mu and 5000 mu for both treatments (the locking
at a damaged zone's ratio); the punch at kappa 2/3 mu (nu 0) for both.
- P13 the elastic part (one volumetric constraint an element against eight) carries more than half of the h 0.2 peak gap; the
  history source alone moves tip3d's peak by less than 5 % (lean 0.55).
- P14 Hertz, h 0.2: full integration above mean dilatation by more than 30 % at kappa 500 mu (lean 0.7) and more than 60 % at
  5000 mu (lean 0.6); mean dilatation within 5 % of Hertz at both (lean 0.6).
- P15 the punch at nu 0: the treatments' peaks within 5 % (lean 0.65).
- P16 the identity control (hist=mean on tip3d) equals tip3d to 1e-10 at every depth (lean 0.95).
Mon 16 Nov 18:59:11 CET 2026
