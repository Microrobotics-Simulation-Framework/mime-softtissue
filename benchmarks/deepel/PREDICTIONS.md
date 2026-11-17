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
