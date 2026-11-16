# NUCLEATE predictions (spike branch spike-nucleate; never merged to cleanup)

Source paper: Fakhouri, Hutchens and Crosby, "Puncture mechanics of soft solids", Soft Matter 11 (2015), doi 10.1039/C5SM00230C.
The paper's numbers, figures and digitised values are not kept in this repository; comparisons against them are in the spike's
report. Here: predictions (model-side numbers) and our own computed results. Labels: [estimate] unless said.

## Round 0: the mu check (predictions committed in a3d304f, 21:06 CET, before digitising)

Result [computed from the paper's Fig. 5(d)/(e) and 7(a)]: the spherical-tip gel's G', as the paper's own figures normalise it,
is **3.66 kPa** (15 punctures, R 0.12 to 4.75 mm, within +/- 2.4 %), not 7.2 kPa. Control: the same computation on the
flat-punch panels (Fig. 5(a)/(b)) returns their legend's G' to 0.3 to 3 % (10 points, 3 gels), so sigma_c = Pc/R^2 as the
caption says (Pc/(pi R^2) would give 0.32 of the legend). P0.1, P0.2, P0.3, P0.5 held; P0.4 failed (Fig. 7(a)'s punctures are
Fig. 5(d)'s small-R points).

## Round 1: the cheap arm (written 2026-11-16 about 21:17 CET (the commit time; an earlier draft of this line carried a guessed 21:24), before any arm ran)

Neo-Hookean AT2 through MIME's `Study` (`examples/needle_sweep.py`), l 0.5 mm, nu 0.45, step 0.05 mm, as the tutorial and GENT
ran it; R 0.5 and 1 mm in the 25 mm block (fine 5 mm), R 2 and 4 mm in a 50 mm block (fine 10 mm), because the arms with a
larger toughness puncture deeper. Fc: the first local maximum the force later falls 2 % below (GENT's rule); no drop = a lower
bound. Records: `nuc_mu3p66_g142` (mu 3.66 kPa, Gc 142), `nuc_mu3p66_g290` (290, a bracket), `nuc_mu3p66_g25` (today's Gc at
the corrected mu), `nuc_mu7p2_g142` (Gc 142 at Fregonese's mu, the mu sensitivity).

- **P1.0 (control):** `softtissue.puncture_gel` (mu 7.2 kPa, Gc 25) at h 0.2 reproduces the tutorial's sweep: Fc 20.52 / 50.98 /
  176.3 / 640.8 mN at R 0.5 / 1 / 2 / 4, each step's force to 1e-6 relative or better (1e-9 at R 1, GENT's measured repro). A
  difference above 1e-6 fails the control. Its 50 mm-block twin at R 4: within 5 % of the 25 mm block.
- **P1.1** g142 at mu 3.66, h 0.2: Fc/R at R 0.5 is 40 to 80 N/m; at R 1, 50 to 100 N/m.
- **P1.2** g290 at mu 3.66, h 0.2: Fc/R at R 0.5 is 55 to 110 N/m.
- **P1.3** No transition: over R 0.5 to 4 the model's Fc/R rises with R (Fc about R^1.4 to R^1.8 throughout), with no stretch of
  Fc proportional to R followed by Fc proportional to R^2.
- **P1.4** At R 4, Fc/(mu R^2) is 14 to 24 for g142 and 20 to 34 for g290 (the AT2 strength there scales as sqrt(E Gc / l), with l
  a placeholder).
- **P1.5** The model's critical depth at R 4 (g142, mu 3.66) is 12 to 22 mm: a neo-Hookean at mu 3.66 is too soft at depth to
  reach a given force as shallowly as a stiffening gel.
- **P1.6** g25 at mu 3.66, h 0.2: Fc/(mu R^2) at R 0.5 is 12 to 18; the dimensional Fc is 0.55 to 0.7 of the mu 7.2 numbers.
- **P1.7** g142 at mu 7.2, h 0.2: Fc/R at R 0.5 is 60 to 110 N/m.
- **P1.8** h 0.1 against h 0.2 at R 0.5: Fc falls by 15 to 35 % (as the Gc 25 arm did); at R 1 within 10 %.
- **Expected verdict:** the cheap arm misses the energy-limited level at small R by 3 to 7 times and shows no transition; any
  agreement at R 4 comes through the placeholder l, so it is not a test there.

### Round 1 status at 21:45 CET (`date`), read before round 2 was written
Control held: the Gc 25 arm equals GENT's MIME run of the tutorial's R 1 member at every one of 41 steps (0 difference);
R 0.5 / 2 match the tutorial's peaks. **Found:** at h 0.2 and R 0.5 every arm's first drop (1.35 to 1.40 mm, a 20 % fall)
is a contact artefact, not a puncture: the elastic run (no damage at all) drops 20 % at 1.40 to 1.45 mm too, and the Gc 142
and 290 arms drop there with no crack (area under 0.001 mm^2); their force then rises, with a sawtooth of period about h, to
95 and 190 mN by 8 and 10 mm. The h 0.1 elastic run has a 3.7 % drop at 1.30 mm. So GENT's committed rule (first 2 % drop)
reads an artefact wherever no crack forms first. **Added rule (set at 21:40, after the h 0.2 R 0.5 arms were read; the h
0.1 arms not yet):** Fc (crack-confirmed) = the first local maximum that the force later falls 2 % below AND over whose
next three steps the crack area grows by at least 0.01 mm^2 (14 times the largest growth at the artefact drops); both
readings are reported.

## Round 2a: the strength-surface model, engine-direct (written 2026-11-16 21:45 CET, `date`, before any arm ran)

`nucleate/needle.py` with `--sts` (uniaxial tensile strength, Biot, kPa) and `--ratio 1.5` (hydrostatic / tensile, a
declared placeholder: Kamarei et al.'s PDMS ballpark), Gc 27 J/m^2 (propagation), mu 3.66 kPa, nu 0.45, l 0.5 mm, h 0.2,
block 50 mm (fine 5 for R 0.5 and 1, step 0.05; fine 8 for R 2 and 4, step 0.1); s_ts 5, 10, 20, 43, 100, 250 kPa at
R 0.5, 1, 2, 4, and 1100 kPa at R 0.5 and 4; the l test: l 0.25 at h 0.1 (h/l 0.4, as l 0.5 at h 0.2) for s_ts 10 and
43 at R 0.5 and 2. Local controls so far: option off is bit-equal to AT2 (40 steps of the R 1 needle, every force and
iteration count); the bar's peak nominal stress is 0.9989 of the declared 10 kPa (l 0.5).

- **P2.1** No single s_ts puts Fc within 30 % of the data at all four radii (lean 0.8).
- **P2.2** For each s_ts the model's Fc rises with R with a log-log slope of 1.5 to 2.2 throughout 0.5 to 4 mm: no
  stretch of Fc proportional to R (lean 0.65).
- **P2.3** Fc rises with s_ts at every R; at s_ts >= 100 kPa no puncture by the run's depth at R <= 1 (lean 0.6).
- **P2.4** l 0.5 to 0.25 (h/l fixed) moves Fc by under 10 % at R 0.5 and R 2 where it punctures (lean 0.6); AT2 moved
  38 % at R 2 (l 0.5 to 0.1, GENT).
- **P2.5** At a puncture the model's depth is 1.5 to 3 times the data's (the neo-Hookean at mu 3.66 is too soft at
  depth: Fregonese's own alpha = 2 law through the data's (Pc, dc) needs mu 4.9 to 8.8 kPa) (lean 0.7).
- **P2.6** The first damage (d > 0.5) sits in the surface skin around the tip, not on the axis below it (lean 0.5).

### Round 2a status at 22:15 CET (`date`), read before round 2b was written
8 of 30 arms in. s_ts 5 kPa punctures early (R 0.5: 9.8 mN at 1.25 mm; R 1: 27.6 mN at 1.65 mm), damage first on the axis
0.3 mm under the apex. s_ts 10, 20, 43 at R 0.5 and 20 at R 1: no damage at all to 20 and 22 mm (the force 541 and 909
mN, identical across s_ts: elastic). s_ts 10 and 20 at R 2: a damaged spot at about 11.5 mm that stops growing; the force
keeps rising (a lower bound). The uniform-state surface check (`nucleate/surface.py`, local): the model matches the
Drucker-Prager surface to 1 % on tensile paths at s_ts 5 kPa (l 0.5 and 0.1 mm, l under the material length 3 Gc / 16
W_ts); at s_ts 20 its equibiaxial onset is 0.74 to 0.77 of the surface's and it never nucleates in uniaxial compression
(l 2 to 10 material lengths). So the sweep above about 10 kPa tests a distorted surface.

## Round 2b (written 2026-11-16 22:16 CET, `date`, before any arm ran)
Elastic maps (no damage; s_eq = the uniaxial strength at which a point's Biot stress sits on the surface, s_hs / s_ts
1.5): R 0.5, 1, 2, 4 at h 0.2 and 0.1. Strength sweep's finer steps: s_ts 6, 7, 8, 9 kPa at the four radii, h 0.2. At
s_ts 5: h 0.1 at the four radii, and l 0.25 at h 0.1 (h/l 0.4) at R 0.5, 1, 2.
- **P3.1** The elastic map's largest s_eq at R 0.5 rises with depth to a maximum between 5 and 10 kPa, then stays below
  it to 20 mm: a knife-edge in s_ts (lean 0.6); its site is on the axis under the apex.
- **P3.2** No s_ts in 6 to 9 kPa puts Fc at R 0.5 within 30 % of the data: either a puncture under 30 mN or none by 20 mm
  (lean 0.7).
- **P3.3** At s_ts 5, l 0.5 to 0.25 (h/l 0.4) moves Fc by under 10 % at R 0.5, 1 and 2 (lean 0.6).
- **P3.4** At s_ts 5, h 0.2 to 0.1 moves Fc by under 15 % at every R (lean 0.6).
