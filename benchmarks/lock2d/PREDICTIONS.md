# LOCK2D predictions (committed before each round's arms run; times from `date`)

Mean dilatation ("mean") against full integration ("gauss") in the axisymmetric engine. Ratios are mean / gauss unless said.
Fc = the first local maximum the force later falls 2 % below (GENT's rule); y = Fc / (mu R^2).

## Round 0: the controls (local, cores 2-3), written 20:48 CET

- **C-a identity** (switch off against f7279ba): Hertz R 4 h 0.2 nu 0.49 (16 steps) and the neo-Hookean needle member R 1
  h 0.2 to 2 mm through MIME (41 steps): every force equal bit for bit (difference exactly 0).
- **C-b homogeneous**: element energy (any damage), internal force (uniform damage) and history source agree to <= 1e-12
  of kappa V; the cylinder's uniform stretch through the solver: both treatments equal to <= 1e-10. Hold.
- **C-c Hertz** (R 4, block 50, mu 7200, elastic, delta/R 0.0125-0.2; kappa/mu 9.67 (nu 0.45), 49.7 (0.49), 500, 5000):
  - mean: F/Hertz within max(its discretisation error, 2 %) at every kappa, at h 0.1: 0.97-1.02, approached from below
    (as tip3d in 3-D) at h 0.2.
  - gauss (the control that must fail): kappa 500 mu over-stiff by x1.2-2.0 at h 0.2; **kappa 5000 mu x1.8-5 at h 0.2 and
    still >= x1.4 at h 0.1: fails the line**. At nu 0.45 and 0.49 gauss holds the line (GENT measured 0.99-1.02 at 0.49).
  - Size of the axisymmetric locking against 3-D's (x1.6, x3.5 at h 0.2): within a factor 2 either way.
- **C-d plant** (the mean without the 2 pi R weight): the element oracle (Pappus volume) catches it in 100 % of 10 000
  random draws (physical scales, the axis column included), each by >= 1e-6 against a <= 1e-12 floor; the correct mean
  passes 100 %. On the Hertz line the plant is caught at kappa 5000 mu only, or not at all (it changes one constraint's
  weights, not the constraint count): F changes < 2 % at kappa <= 50 mu.

## Round 1: the arms (runners), written 20:48 CET

- **neo-Hookean nu 0.45, h 0.1** (gauss 8.79 / 7.24 at R 0.5 / 1): mean lowers Fc by 5-25 % (y 6.6-8.4 / 5.4-6.9);
  direction confidence about 75 %. The miss below Exp. 1's bars (data/model 8.0 / 4.8) grows by the same factor; it does
  not close.
- **Gent Jm 9.312 nu 0.49** (gauss y 65.4 / 28.6 at h 0.1, 16.6 / 10.9 at h 0.2 for R 0.5 / 1 / 2 / 4): mean lowers every
  point: R 0.5 by 10-40 % (40-60), R 1 by 10-35 % (18-26), R 2 by 5-30 % (11.5-15.5), R 4 by 0-15 % (9.3-10.9).
  Inside Exp. 1's bars with mean: 0 or 1 of 4 (gauss: 2 of 4). Half or more of GENT's nu 0.45-vs-0.49 gap at R 0.5 and 1
  is locking (mean at 0.49 lands within 20 % of gauss at 0.45 there).
- **Axisymmetric 1 mm punch** (gel placeholder nu 0.4, through MIME, block 5, fine 1.0, step 0.04 mm): gauss peak above
  mean at h 0.1 and 0.0707 by 8-25 % (3-D: +19 %); gauss peak 10.0-10.8 mN at h 0.1; mean 8.3-9.5 mN. Before the peak the
  gap is smaller (F at 0.08 mm within 2-7 %). The mean peak rises 1-6 % from h 0.1 to 0.0707 (3-D tip3d +4.7 %) and keeps
  rising to 0.05 and 0.0354 (not converged by h/l 0.35); its peak / F(0.04 mm) stays within 2 % across those grids.
- Runtime: mean costs within x0.7-1.5 of gauss per arm.
