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

## Round 2 (runners), written 21:20 CET, after round 1's punch arms and the local Hertz controls at h 0.2 and 0.1

Seen before writing: Hertz gauss at kappa 5000 mu x1.56-1.95 (h 0.2) and x1.24-1.45 (h 0.1), at 500 mu x1.10-1.26 and
x1.03-1.10; mean 0.977-1.008 at every kappa on both grids. The axisymmetric punch: gauss over mean at the peak +16 / +12 /
+8 / +2.5 % at h 0.1 / 0.0707 / 0.05 / 0.0354; the mean peak 9.01 / 10.01 / 11.27 / 11.75 / 11.29 mN (h 0.1 to 0.025).

- **Hertz at h 0.05** (both treatments, the four kappas): mean within 2 % of Hertz at delta/R 0.04-0.19 and within 1 % of
  its h 0.1 value at every kappa; gauss at 5000 mu still x1.08-1.30 (the excess falling by a third to a half a halving):
  outside max(its three-grid discretisation error, 2 %), the control failing as it must; gauss at 500 mu x1.01-1.05.
- **The punch, finer and with half the step** (step 0.02 mm at h 0.0354 and 0.025, both treatments; h 0.0177 at 0.04 mm,
  both): the gap at the peak <= 4 % at h 0.025 and 0.0177 (it closes); the mean peak at h 0.0177 in 11.0-12.2 mN; halving
  the step moves the sampled peak by <= 2 %; peak / F(0.04) within 3 % across h 0.05 to 0.0177 for mean.

## Round 3 (runners): the punch at HERTZ-CHECK's axisymmetric settings, written 21:3x CET (the commit time rules)

(Round 2's header said 21:20; its commit is 21:17:59.) HERTZ-CHECK's apparent limit, 11.5-11.7 mN, is its axisymmetric
build at h 0.0354 / 0.025: full integration (spike 3D-PUNCTURE's arm 2: steps 0.02 mm, no viscosity, Newton cap 100,
block 5, fine 1.0). Its ladder: 10.26 / 10.99 / 11.99 / 11.51 / 11.67 mN at h 0.1 / 0.0707 / 0.05 / 0.0354 / 0.025.
Informed by round 1 (the same punch through MIME, 0.04 mm steps, viscosity 0.1 Gc/l: gauss 10.46 / 11.18 / 12.19 / 12.04,
mean 9.01 / 10.01 / 11.27 / 11.75 / 11.29).

- **Settings identity:** gauss on this engine reproduces that ladder within 0.5 % at every depth (the same arithmetic;
  bit for bit only if nothing in the engine's path moved since).
- **mean**, h 0.1 / 0.0707 / 0.05 / 0.0354 / 0.025: 8.6-9.4 / 9.6-10.4 / 10.8-11.6 / 11.0-11.7 / 11.0-11.7 mN; the gap to
  gauss 12-18 / 8-14 / 4-10 / 0-4 / 0-4 %. At h 0.0177 both within 11.0-11.9 mN and within 3 % of each other.
- **Judgement it would support:** the converged peak about 11.2-11.7 mN; full integration's limit inflated by locking by at
  most about 4 % at h/l <= 0.35. If mean at h 0.025 and 0.0177 reads below 10.9 mN, I am wrong and the limit is inflated.

## Round 4 (runners): the needle points at the finer grids, written after round 1's first three needle arms

(Round 3's header said "21:3x"; its commit is 21:24:50.) Seen before writing: mean against gauss, Fc / (mu R^2): neo-Hookean
R 0.5 h 0.1 8.79 against 8.79 (+0.0 %); Gent R 2 h 0.2 10.69 against 16.61 (-36 %); Gent R 4 h 0.2 11.50 against 10.88
(+6 %). GENT's engine-direct full-integration twins at h 0.05 (its run, downloaded, not yet in its report): y 89.9 / 29.5 /
12.25 at R 0.5 / 1 / 2 (R 4 running).

- **MIME's Study, mean, h 0.1:** neo-Hookean R 2 and R 4 within 5 % of gauss (6.27, 5.72); Gent R 2 within 0.75-1.25 x
  of gauss's 14.61 (direction not predicted: at h 0.2 the pair is grid-dominated); Gent R 4 within 12 % of gauss's twin
  (pending on GENT's run; if it never lands, no comparison is claimed).
- **Engine-direct, mean, h 0.05** (GENT's pf-line arguments plus `--volumetric mean`): each within 15 % of its
  full-integration twin; with mean, 1 or 2 of R 0.5 / 1 / 2 inside Exp. 1's bars (gauss: 1 of 3).

## Round 5 (runners): the noise floor of Fc, and the nu question at h 0.2; written after round 1's six needle arms

Seen: mean against gauss, y = Fc / (mu R^2): neo-Hookean h 0.1 R 0.5 / 1: 8.79 / 7.03 against 8.79 / 7.24; Gent h 0.1
R 0.5 / 1: 58.4 / 30.4 against 65.4 / 28.6; Gent h 0.2 R 2 / 4: 10.69 / 11.50 against 16.61 / 10.88. The signs differ, so
the question is whether these are locking or the puncture's own sensitivity.

- **A perturbation that physics ignores** (the block 24 mm instead of 25; gauss): it moves Gent's Fc by 0-10 % at R 0.5 and
  1 (h 0.1) and by 0-30 % at R 2 (h 0.2). If it moves them by as much as the treatment does, the treatment's needle effect
  is not attributed to locking. The same perturbation on mean at R 2 (h 0.2) moves it by 0-30 %.
- **nu at h 0.2:** mean, Gent nu 0.49, R 0.5 and 1: within 0.6-1.4 x of gauss's 24.3 / 24.7. GENT's nu 0.45 (15.8 / 12.5)
  is not reproduced by removing the locking: mean at 0.49 lands nearer gauss at 0.49 than gauss at 0.45 at R 1.
