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
