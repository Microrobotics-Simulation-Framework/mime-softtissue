# TENSION (spike G3): predictions and pass lines, committed before each round

A tension-only strength surface (`strength.Tension`: the largest principal Biot stress reaching s_ts) in the
phase field, Gent at G' with a loading-fitted Jm, the edge contact (4 points, 1000 E/h), full integration, Gc the
notched value. The puncture data are digitised from a published figure and stay out of this repository: lines
below name them ("the data's Pc at R 4"); the numbers live in the private work folder.

## Round 0: choices, before any prediction run (2026-11-17 09:47 CET, `date`)

- **Stress measure: Biot** (first Piola-Kirchhoff principal values, d W / d lambda_i). Reasons: (1) a uniaxial
  tensile test reports the nominal stress, force over the undeformed area, which is the Biot stress on that path, so
  the calibrated s_ts is the number an independent tensile test would measure; (2) the existing surface and its
  coefficients (the uniaxial point exact at any l) are in Biot. The sign of a principal value is the same in Biot
  and Cauchy (lambda / J > 0), so "compression never nucleates" holds in either; the Cauchy value is recorded too.
- **Calibration radius: R 4 mm, not R 1.** In the data's two regimes the strength is identifiable only where the
  force is stress-limited (above R_T); R 1 sits on the energy-limited branch, where Pc is set by the toughness and a
  strength fitted there is ill-conditioned. Calibrating on the plateau makes the energy branch (R 0.5, R 1) a
  prediction from the notched Gc, which was measured independently of the puncture data. Held out: R 0.5, 1, 2
  (and 0.25 if affordable). The same ladder also gives what calibrating at R 1 would do (reported as secondary).
- **Calibration rule:** run a ladder of s_ts at every radius; s_ts* is the interpolation (log Pc against log s_ts,
  linear between the bracketing rungs) at the data's Pc at R 4; the held-out predictions are the same interpolation
  at s_ts*, then confirmed by runs at s_ts* itself (the confirmed numbers are the result).
- **Puncture reading:** the first force drop of at least 30 % (NUCLEATE's rule); also recorded: the first
  crack-confirmed event (>= 0.01 mm^2 of new crack over three steps) and the first 2 % drop.
- **Pass line, per held-out radius:** Pc within a factor 1.35 of the data's point (|ln ratio| <= 0.30); dc within
  a factor 1.35 (secondary).
- **Pass line, regimes:** (i) energy branch: Pc(R 1) / Pc(R 0.5) in [1.6, 2.8]; (ii) plateau: Pc(R 4) / Pc(R 2) in
  [3.0, 5.3]; (iii) the model's Pc / (G' R^2) at R 1 at least 2 times its value at R 2, and at R 0.5 at least 3 times
  its value at R 4; (iv) R_T, the crossing of the energy line (mean Pc / R of R 0.5 and 1) with the plateau (mean
  Pc / R^2 of R 2 and 4), within a factor 1.5 of the same computation on the data.
- **The test passes** if all three held-out radii pass on Pc and (i)-(iv) hold.

## Round 1: the elastic fields (no damage), before its arms (2026-11-17 09:47 CET, `date`)

Elastic twins, Gent Jm 3.7 (and 3.5, 4.5) at G' 3.66 kPa, nu 0.49, R 1 (h 0.05; h 0.025; fine zone 6 and 9 mm
deep) and R 0.5 scale-equivalent (h 0.025): the largest principal Biot stress s1 each step and where.

- P1.1 At d/R <= 1 the largest s1 sits on the free surface just outside the contact (a ring, r 0.5-1.2 R, I1 >= 0).
- P1.2 s1_max / G' rises slowly with depth after d/R 1: below 10 at d/R 3 (where the data's R 4 punctures). Even
  so, a sharp strength (no length) cannot give two regimes: the elastic problem depends on d/R and h/R only, so a
  strength reached at one d/R gives Pc / (G' R^2) the same at every R. Any energy branch must come from Gc and l.
- P1.3 Scale invariance: s1_max at R 0.5 h 0.025 within 3 % of R 1 h 0.05 at equal d/R.
- P1.4 Grid: s1_max at h 0.025 within 10 % of h 0.05 (a surface stress concentration converges slowly); fine
  zone 9 mm against 6 mm: force within 1 %, s1_max within 3 %.
- P1.5 Jm 3.5 against 4.5 moves s1_max at d/R 3 by under 25 %.

## First prediction of the test (before any phase-field needle run)

- P0.1 The tension-only surface removes the compressed apex as a nucleation site; nucleation is at the surface
  ring (I1 >= 0) at shallow depth for a low strength, and the calibrated s_ts is a few G' (2-10 G').
- P0.2 Calibrated at R 4, the held-out R 2 passes (both on the plateau, Pc ~ R^2 from scale invariance).
- P0.3 The energy branch is weaker than the data's: R 0.5 and R 1 come out below the data, R 0.5 by more than
  the 1.35 factor (Pc / (G' R^2) rising from R 4 to R 0.5 by 1.5-3 times, not the data's 7). Test fails on R 0.5.
  Probability that all three held-out radii pass: about 15 %.
