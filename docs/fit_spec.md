# Data spec for the material fit (`python -m mime_softtissue.fit`)

One manifest (TOML) names the files; each file is one test of one specimen, a CSV with a header row. Column names are
fixed (any order; other columns are ignored; lines starting with `#` are comments). Strains are fractions (0.01 is
1 %), stresses and moduli in Pa, frequencies in Hz. Every value is positive (compression is counted positive).

## Files

| test | required columns | optional columns |
|---|---|---|
| rheology, strain (amplitude) sweep | `strain` (shear strain amplitude), `storage_modulus_Pa` | `loss_modulus_Pa`, `frequency_Hz` |
| rheology, frequency sweep | `frequency_Hz`, `storage_modulus_Pa` | `loss_modulus_Pa`, `strain` (the amplitude; else give it in the manifest) |
| unconfined compression | `strain` (engineering: change in height over the initial height), `stress_Pa` (nominal: force over the initial cross-section) | |

Points whose value is under 1 % of the file's largest are dropped (a relative residual means nothing there); the
record says how many.

## Manifest

```toml
[fit]
model = "neo_hookean"      # or "fibre": neo-Hookean plus the plugin's fibre pair (k1, k2)
fibre_angle_deg = 45.0     # the fibre pair's angle, fixed, not fitted

[[rheology]]
file = "clot1_strain_sweep.csv"
sweep = "strain"           # or "frequency"
geometry = "parallel_plate"  # or "cone_plate": the instrument's geometry (parallel plates read a torque-averaged strain)
name = "clot 1, strain sweep"
# strain = 0.01            # a frequency sweep's amplitude if its file has no strain column

[[compression]]
file = "clot1_compression.csv"
name = "clot 1, compression"
```

Run: `python -m mime_softtissue.fit manifest.toml --out results/ [--model fibre] [--bootstrap 200]`. It writes
`fit.json` (parameters, relative standard errors, 95 % intervals, which parameters the data identify, each data set's
noise and misfit flags) and `residuals.png` (data and fit above, relative residuals and their trend below).

## What is fitted, and what each parameter needs

The plugin's law (`engine/material.py`): psi = (mu/2)(J^-2/3 I1 - 3) + (kappa/4)(J^2 - 1 - 2 ln J) [+ fibres
(k1/2k2)(exp(k2 <I4 - 1>^2) - 1), two families at +-angle]. The rheometer is modelled as simple shear at a fixed gap
(fibres in the plane of shear); compression as uniaxial stress between frictionless platens (fibres across the load).

| parameter | needs |
|---|---|
| mu | G' at small strain (any sweep), or the initial slope of compression together with nu |
| kappa (nu) | shear and compression together (G and E give nu); compression alone pins it only loosely; shear alone not at all |
| k1 | data past the linear range: G' rising with amplitude, or compression past about 20 % |
| k2 | the curvature of that rise: data well into it |

## What it cannot identify

- Viscoelasticity: G'' is read and its ratio to G' reported, never fitted; the law is elastic.
- Rate dependence: a frequency sweep is fitted by one G'; a trend with frequency is flagged as a misfit, not modelled.
- Poroelastic drainage and fluid loss in compression (the test's rate is not used).
- Fracture (toughness, length scale): needs a notched test or a puncture.
- The fibre pair is planar; a gel is isotropic. Fitted k1, k2 are a stiffening term in the stated orientations, not
  properties of fibres.

## Flags

A data set is flagged when its residuals carry a trend the law cannot follow (a cubic in log strain or log frequency,
F-test p < 1e-3 and at least 3 % in size), or, with several sets, when its mean residual is off zero (p < 1e-3, at
least 3 %): the sets disagree under one law (for example shear and compression moduli that no single mu and nu fit).
A flag means the law, not the data, is in question; the parameters are then a best compromise, not a material's.
