# NUCLEATE predictions (spike, branch spike-nucleate; private, never merged to cleanup or the public seed)

Source of every digitised number here: Fakhouri, Hutchens and Crosby, "Puncture mechanics of soft solids", Soft Matter 11 (2015),
doi 10.1039/C5SM00230C (accepted manuscript). Numbers only; no figure or page image is stored in this repository.
Labels: [estimate] unless said; [read] = the paper's text; [arithmetic] = mine.

## Round 0: the mu check (written 2026-11-16 21:06 CET, `date`, before any figure was rendered or digitised)

What I know before digitising [read, text only]: the 25 w% gel's sigma_0 is about 1.1 +/- 0.18 MPa "from Figure 5(b)" (flat
punches) and flat-punch sigma_c is about 145 G'; Pc/R = 290 N/m (Fig. 7a, 25 w%); Fregonese's Exp. 1 (their Fig. 5, digitised in
`studies/data/fregonese2021_fig5.json`), converted back with mu 7.2 kPa and Gc 25, gives Pc/R of 164 to 355 N/m for R 0.04 to
3.9 mm and Pc about 1.30 N at R 3.9 mm [arithmetic].

- **P0.1** Fig. 5(d)'s Pc at R about 4 mm reads 1.3 to 1.5 N.
- **P0.2 (the identity check, can fail):** G' = (Pc/R^2) / (sigma_c/G') at matching R is constant across at least 6 points to
  within +/-15 % (one gel, one G', both panels from the same punctures). A spread over 25 % means the panels are not one gel and
  one G', or my digitisation is off; I would then check by overlay before believing either.
- **P0.3** That G' is 2.8 to 3.8 kPa, not 7.2 kPa (lean 0.7). 1.1 MPa / 145 = 7.6 kPa [arithmetic], so 7.2 kPa is probably the
  25 w% flat-punch gel's G', and the spherical gel would then be a softer one.
- **P0.4** Fig. 7(a)'s points (25 w%) are NOT the same punctures as Fig. 5(d)'s small-R points (different R values or Pc at the
  same R differing by more than 20 %) (lean 0.55). If they are the same, Fig. 5(d,e) is the 25 w% gel and P0.3 contradicts
  sigma_0 = 1.1 MPa with sigma_c/G' = 145.
- **P0.5** Fig. 7(b)'s Gc histogram centres at 20 to 35 J/m^2 and its unnotched (Gamma_0 tensile) at 100 to 180 J/m^2.
- **If G' is about 3.2 kPa:** in the master-curve plane every data point moves left by 7.2/3.2 = 2.25 in x and up by 2.25 in y;
  the neo-Hookean model (y about x^-0.35 locally) at mu 3.2 kPa gives a dimensional Fc about (3.2/7.2)^0.65 = 0.59 of today's,
  so the dimensional miss at R 0.5 grows from about 6 to about 10 times [arithmetic]. GENT's Jm and the paper's alpha = 9 were
  both chosen with mu 7.2 kPa, so both would need restating.
