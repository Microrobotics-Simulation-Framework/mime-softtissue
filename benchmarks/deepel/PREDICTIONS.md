# DEEP-ELASTIC predictions (committed before each round's arms run; times from `date`)

Elastic (damage off) deep indentation of a neo-Hookean block (mu 3660 Pa, nu 0.49, mean dilatation) by a sphere-tipped
needle of radius R, engine-direct (`axi_deep.py`), frictionless, 0.05 mm steps. A "drop" is any step whose force falls
more than 0.5 % below the largest force so far: in an elastic run every drop is an artefact. The loading ratio is P over
eq. 5 of Fakhouri, Hutchens and Crosby (Soft Matter 2015), P = k' E d^2 + k'' E R d at its stated constants, E = 3 mu.

Note before any arm: the elastic neo-Hookean problem and eq. 5 are both free of any length but R once the block is large,
so P / (E R^2) is a function of d / R and h / R alone. R 1 at h is R 0.5 at h / 2 over a shorter range of d / R: it checks
the numerics (scaling, block), not a law.

## Round 0 (local, cores 0-1, h 0.2, R 0.5, to 6.5 mm), seen before writing (06:40-06:55 CET)

- Nodal penalty (100 E / h, the GelBlock's): first drop at 1.3 mm (d/R 2.6), then at 1.4, 2.9, 5.2 mm; the tip passes
  between surface nodes (the deformed top edge inside the tip by 0.12 mm at the first drop, 0.39 mm = 0.78 R by 6 mm) and
  20 nodes below the top row end inside the tip. At 1000 E / h: the same (first drop 1.3 mm, 13 nodes inside by 3.5 mm).
- Edge penalty (4 Gauss points an edge of the top face, exact signed distance): at 100 E / h the tip pokes through at
  2.6 mm and the gel lets go (force 0); with 8 points the same. At 1000 E / h: no drop to 4.7 mm (d/R 9.4), then drops.

## Round 1 (runners), written 06:57 CET

Arms: nodal (100 E / h) and edge (4 points, 1000 E / h) at R 0.5 (h 0.1, 0.05, 0.025; block 60 mm; to 6.5 mm) and R 1
(h 0.1, 0.05; block 80; to 8.5 mm); edge at R 0.5 h 0.05 with block 30, with 100 E / h, and with 8 points.

- **P1.1 Nodal has artefacts on every grid.** First drop at d/R 3-5 (h/R 0.2), 4-8 (0.1), 6-11 (0.05). Before each first
  drop the deformed top surface is inside the tip by more than 0.1 R between nodes (the cause: the tip passes between
  nodes); nodes below the top row end inside the tip after it. Newton's cap is not the cause (0 capped solves before the
  first drop).
- **P1.2 Edge, 1000 E / h:** no drop to d/R 13 at h/R 0.1 and 0.05 (confidence 65 %); at h/R 0.2 a drop between d/R 8 and
  13 (50 %). Its surface penetration stays under 0.02 R where no drop occurred.
- **P1.3 Smallest J** above 0.1 on every arm until a drop; no inversion.
- **P1.4 Loading ratio (k' at its stated value), artefact-free stretches:** 0.40-0.60 at d/R 2, 0.35-0.55 at d/R 4-8,
  0.30-0.55 at d/R 12; outside the band (k' at its stated +- 1 sigma) at every d/R >= 2. Finer grids lower P (coarse Q1
  is stiff in the wrapped surface): h/R 0.1 to 0.05 lowers P at d/R 8 by 2-10 %.
- **P1.5 Scaling:** R 1 at h equals R 0.5 at h / 2 in P / (E R^2) to 2 % where both are artefact-free.
- **P1.6 Block:** 30 mm against 60 mm moves P by under 1 % at every depth.
- **P1.7 Penalty:** 100 E / h fails earlier than 1000 E / h at h 0.05 (a drop before d/R 13); where both are artefact-free
  they differ by under 3 %. 8 points against 4: under 1 % where both are artefact-free.
- Runtime: h 0.025 at R 0.5 within the runner's wall (2-5 h); R 1 at h 0.025 not run (an estimated 10-15 h).

## Round 2 (runners): Gent's Jm against eq. 5 at R 0.5, written 07:03 CET (commit 07:04)

Launched before round 1's fine-grid neo-Hookean result, on the local h 0.2 numbers (seen before writing): neo-Hookean
edge 0.53-0.55 of eq. 5 from d/R 1 to 8 (flat), far outside the band; Gent at h 0.2: Jm 3 rises 0.56 (d/R 1), 0.86 (4),
1.16 (8), 1.38 (12); Jm 10 rises 0.54, 0.66, 0.80, 0.91; no drop to d/R 13 for either (Gent bounds the stretch: largest
1.9 and 2.8). Control (a) passed locally (06:59): edge_points 0 against 8110f3f, the nodal elastic deep run and the
phase-field needle with the lagged pressure (axi_run.py, mean), 20 steps each, every entry bit for bit.
If round 1's fine neo-Hookean lands inside the band at depth, these arms are discarded.

Arms: edge contact (4 points, 1000 E / h), R 0.5, h 0.05, block 60, to 6.5 mm; Jm 2, 3, 4, 5, 7, 10, 15.

- **P2.1 Shape:** for every Jm the ratio to eq. 5 rises with depth (Gent's stiffening is steeper than d^2): no single Jm
  stays inside the band (k' +- 1 sigma, about +-4 % in P at depth) over d/R 2-12. Confidence 80 %.
- **P2.2 The least-squares Jm** over d/R 2-12 (relative error, k' at its stated value): 3-7 at h 0.05; it falls by
  under a factor 1.5 from h 0.2's. Its ratio spans about 0.75-1.25 over that range.
- **P2.3 No drop** to d/R 13 for Jm <= 10 at h 0.05 (the stretch bounded); Jm 15 may drop past d/R 10 (40 %).
- **P2.4 Grid:** h 0.05 lowers P against h 0.2 by 5-20 % at d/R 8 for every Jm.

## Round 3 (runners): a finer axis column, and the flat punch (the premise of comparing a sphere with eq. 5), written 07:08 CET

Seen before writing: control (b) Hertz (R 4, block 50, nu 0.49, mean): edge 1000 E/h F/Hertz 0.985-1.009 (h 0.2) and
0.986-1.001 (h 0.1), within 1 % to delta/R 0.1 on both (0.998-1.009, 0.994-1.001); nodal 0.982-0.998 (h 0.2). Local h 0.2
edge failure at d/R 9.6: the axis column's surface element (reference R 0-0.2 mm) wraps the whole hemisphere, its axis node
0.56 R below the apex (a sling). No round 1 or 2 arm finished yet.

Arms: (A) neo-Hookean edge, columns refined near the axis only (cheap: the solve costs the columns linearly, the rows
cubed): R 0.5, h 0.05 with 0.0125 within 0.75 mm; h 0.025 with 0.00625 within 0.5 mm. (F) a flat punch (edge radius
0.2 R), edge contact, R 0.5: neo-Hookean at h 0.05 and 0.025; Gent Jm 4 and 7 at h 0.05.

- **P3.1** The axis refinement delays the first drop: no drop to d/R 13 at h 0.05 / 0.0125 (60 %); it changes P by under
  3 % where both it and the uniform grid are clean.
- **P3.2 The premise:** the paper reads spheres as loading like flat punches for d > R. In the model (neo-Hookean) the
  flat punch sits above the sphere by 20-60 % at d/R 2-4 and the two approach to within 15 % by d/R 12; the flat punch's
  ratio to eq. 5 is 0.85-1.05 at d/R 1 (eq. 5's linear term is the incompressible flat punch's small-strain law).
- **P3.3** Gent narrows the sphere/flat gap at depth against neo-Hookean (both near the same stiffened core).
- **P3.4** The flat punch's edge drops (the gel wrapping a 0.2 R edge) before d/R 13 at h 0.05 (50 %).

## Round 4 (runners): Gent at the fitted range, its grid, block, held-out radius and the nodal penalty, written 08:08 CET (my 08:12 was a guess; commit time)

Seen before writing (rounds 1-3 so far): neo-Hookean edge drops at d/R 4.8 (h/R 0.1, R 0.5) and 3.7 (h/R 0.1, R 1), 8.4
(h/R 0.2); the axis column ahead of the tip is squashed to axial stretch 0.03-0.07, hoop 4-7, and snaps (lmax 4.4 -> 6.1 in
two steps) before the first small drop; nodal drops at d/R 2.8 (h 0.1). Scaling held: R 0.5 h 0.05 and R 1 h 0.1 give
P/(mu R^2) 9.9 / 19.3 / 41.4 against 9.9 / 19.3 / 41.6 at d/R 2 / 4 / 8. Gent Jm 2-15 at h 0.05: no drop to d/R 13; the ratio
to eq. 5 rises with depth for every Jm (Jm putting P on eq. 5: 2.4 at d/R 5, 4.5 at 8, 7.2 at 12); least squares Jm 3.7
over d/R 2-12 (ratio 0.63-1.26), 3.6 over 1-12, 4.35 over 4-12. Flat punch (edge 0.2 R) Gent Jm 4 / 7: above the sphere
by 57 / 27 / 16 / 5 % at d/R 1 / 2 / 4 / 12 (Jm 4).

Arms (edge 1000 E/h unless said; Jm 4): R 0.5 at h 0.1 and 0.025 (the ladder with h 0.05); R 1 at h 0.1 and 0.05 (held
out: the same h/R as R 0.5's h 0.05 and 0.025); R 0.5 h 0.05 with block 30; the nodal penalty (100 E/h) at R 0.5 h 0.05
and h 0.025.

- **P4.1 Ladder:** h 0.05 -> 0.025 moves P by under 3 % at every d/R <= 12; h 0.1 sits 2-8 % above h 0.05 at depth.
- **P4.2 Held out:** R 1 at h 0.1 reproduces R 0.5 at h 0.05 in P / (E R^2) to 1 % at every d/R <= 8.5 (scale-free; it
  can only fail through the numerics). The fitted Jm transfers to R 1 by construction.
- **P4.3 Block 30:** under 1.5 % from block 60 to d/R 12 (Gent's larger force reaches the walls more than neo-Hookean's).
- **P4.4 Nodal with Gent:** no drop above 0.5 % to d/R 13 at h 0.05 (55 %); its P within 3 % of the edge's where both clean.
- **P4.5** Every arm drop-free to d/R 13 except possibly the nodal ones.

## Round 5 (runners): Gent's Jm fitted on eq. 5's own geometry (the flat punch), written 08:18 CET (commit time; the 08:22 first written here was a guess)

Seen before writing: flat punch (edge 0.2 R) ratios at d/R 1 / 2 / 4 / 8 / 12: Jm 4 0.82 / 0.79 / 0.90 / 1.12 / 1.29; Jm 7
0.80 / 0.74 / 0.79 / 0.94 / 1.07; neo-Hookean 0.78 / 0.64 / 0.52 then drops. Gent Jm 4 at h 0.1 within 0.9 % of h 0.05.
Eq. 5 was fitted to flat punches, so the flat punch is its own geometry; the sphere fit (Jm 3.7) may absorb the sphere's
shortfall at d/R <= 4.

Arms: flat punch, edge 0.2 R, R 0.5, h 0.05, edge contact 1000 E/h, Gent Jm 3, 5, 10.

- **P5.1** The flat-punch least-squares Jm over d/R 2-12 lies in 4.5-7, above the sphere's 3.7 (the sphere's fit absorbs
  geometry); its ratio still rises with depth (spread at least 0.75-1.2).
- **P5.2** All three drop-free to d/R 13.

## Round 6 (runners): is block 60 large enough with Gent? written 08:33 CET (`date` 08:33:36; the 08:35 first written here was wrong)

Seen before writing: Gent Jm 4, R 0.5, h 0.05: block 30 against 60 +0.6 / +1.6 / +3.5 / +5.9 / +6.6 % at d/R 1 / 4 / 8 /
12 / 13 (P4.3's 1.5 % failed). The neo-Hookean's at block 30: +0.5 to +2.1 % to d/R 4.5. Axis-refined neo-Hookean
(0.0125 near the axis at h 0.05): first drop earlier, d/R 3.2 (P3.1 failed).

Arms: Gent Jm 4, R 0.5, h 0.05, edge, blocks 120 and 240 mm.

- **P6.1** The wall's share falls about as R / L (L the tip-to-bottom distance): block 60 sits 2-4 % above block 120 at
  d/R 12, and 120 within 1.5 % of 240. The fitted Jm moves up by 3-10 % with the larger block.
