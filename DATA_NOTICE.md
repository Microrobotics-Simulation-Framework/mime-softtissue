# Data notice

Files in this repository that hold values taken from other people's work. Everything here is stated from the files' own
metadata; where a file records no licence, this notice says so rather than guessing. The plugin's code is Apache-2.0.

## Digitised from published figures

These are readings of published figures, facts cited with their sources. They are not covered by this repository's Apache-2.0
licence, and the licence of each source paper is not recorded in the files.

- `src/mime_softtissue/studies/data/fregonese2021_fig5.json`
  - Source: Fregonese S., Bacca M., "Piercing soft solids: a mechanical theory for needle insertion", J. Mech. Phys.
    Solids 154 (2021) 104497, doi:10.1016/j.jmps.2021.104497; also on arXiv:2412.09809 (posted 13 Dec 2024).
  - Obtained: digitised from the raster of Fig. 5 (a bitmap embedded in the PDF), by fitting marker glyphs and curve strokes to
    the pixels; about 0.005 to 0.02 decades of uncertainty. No figure pixels are stored. Fig. 5's second data set
    (Exp. 2) is the data of Rattan and Crosby 2019, as the file names it; Exp. 1 is the gel data of Fakhouri, Hutchens and Crosby 2015.
  - Licence: the file records that the arXiv abstract page lists the arXiv non-exclusive distribution licence and that no
    Creative Commons licence link was seen. The licences of the journal paper and of the Rattan and Crosby data are not recorded.
- `src/mime_softtissue/studies/data/lohr2025_digitised.json`
  - Source: Lohr M.J., Bechtel G.N., Dortdivanlioglu B., Rausch M.K., Int. J. Fract. (2025), doi:10.1007/s10704-024-00820-4,
    read from the author manuscript at https://par.nsf.gov/servlets/purl/10575340.
  - Obtained: digitised from the vector paths of Figs. 1A, 1B and 5A (calibrated on the axis strokes; by-eye points for some
    curves, as the file's labels say), with a few numbers the paper states.
  - Licence: not recorded in the file; the paper's and the manuscript's licences are not recorded.

## Raw data and values read from papers

- `src/mime_softtissue/records/data/fibrin_plasma_tutwiler2020.json`
  - Source: Tutwiler et al. 2020, Sci. Adv. 6:eabc0496, and Tutwiler et al. 2021, Acta Biomater. 131:355.
  - Obtained: single values read from the papers' text and tables (each value is labelled `read`, `derived` or `placeholder`
    in the file); no data series.
  - Licence: the record is labelled CC-BY-4.0 as the plugin authors' record; the two papers' licences are not recorded.
- `src/mime_softtissue/records/data/puncture_gel_fakhouri2015.json`
  and `src/mime_softtissue/records/data/puncture_gel_gent.json`
  - Source: gel constants of Fakhouri, Hutchens and Crosby 2015, Soft Matter 11, 4723, as stated on p. 9 of Fregonese and
    Bacca (J. Mech. Phys. Solids 154 (2021) 104497, doi:10.1016/j.jmps.2021.104497; arXiv:2412.09809).
  - Obtained: two numbers read from that page (mu = 7.2 kPa, Gc = 25 J/m^2); other values in the record are labelled
    placeholders or numerical choices.
  - Licence: the record is labelled CC-BY-4.0 as the plugin authors' record; the source papers' licences are not recorded.
- `src/mime_softtissue/records/data/whole_blood_clot_lohr2025.json`
  - Source: constants calibrated by this plugin on the raw data of Lohr et al. 2025 (doi:10.1007/s10704-024-00820-4), data
    sets doi:10.18738/T8/97ZAZN and doi:10.18738/T8/DUJWSR, both CC0 1.0 (the Lohr raw data are not stored in this branch).
  - Obtained: fitted by the plugin (jackknife uncertainties in the file's labels), not copied.
  - Licence: the record is labelled CC-BY-4.0 as the plugin authors' record; the underlying raw data are CC0 1.0.
- `src/mime_softtissue/studies/data/rausch_visco_na0wxy.json`
  - Source: Texas Data Repository, doi:10.18738/T8/NA0WXY, "Hyper-viscoelastic damage modeling of blood clot under large
    deformation - Experimental Data" (Rausch lab).
  - Obtained: raw download, reduced to box means (0.1 s, then 1 s after the first 10 s of each hold); no offset removed and no
    other filtering.
  - Licence: CC0 1.0, as the file records it (noted on download, 2026-11-16).
- `src/mime_softtissue/records/data/visco/calibration.json`
  - Source: calibration on the Rausch lab data above (doi:10.18738/T8/NA0WXY, CC0 1.0); its relaxation times are declared
    from Sugerman et al. 2024's published values, and its Yeoh constants from the pure-shear replication (363, 434 Pa).
  - Obtained: computed by the plugin (fitted or declared, as the file's labels say); not third-party values copied.
  - Licence: labelled CC-BY-4.0 in `records.json` as the plugin authors' result.

## Computed by this plugin

Results the plugin computed (runs, fields, curves) are listed in `src/mime_softtissue/records/records.json` with their
licences and checksums; they hold no third-party values beyond the inputs named above.

Anything here is removed on request: open an issue.
