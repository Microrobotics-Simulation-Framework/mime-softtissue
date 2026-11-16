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

## Round 1: the cheap arm (written 2026-11-16 21:24 CET, `date`, before any arm ran)

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
