"""Fitting the plugin's material law to a lab's rheology and unconfined compression (pure JAX, no MIME import):
``homogeneous`` (the law at the tests' homogeneous deformations), ``core`` (least squares, uncertainty, misfit,
identifiability), ``data`` (the CSV spec, ``docs/fit_spec.md``), ``synthetic`` (data from known parameters, for the
controls), ``report`` (the record and the residual plot). Command line: ``python -m mime_softtissue.fit``."""
