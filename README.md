# mime-softtissue

A plugin for [MIME](https://github.com/Microrobotics-Simulation-Framework/MIME), namespace `softtissue`: a large-strain
solid that can tear (plane strain and axisymmetric, AT2 phase-field fracture, pure JAX, differentiable in forward mode), a
rigid tip pressed into it through contact, a fitting harness for rheology and compression data, and clot records. It is
checked against one published benchmark (the edge-notched fibrin strip, Tutwiler et al. 2020) and demonstrated on
placeholder numbers. **It is the model, not a measurement of fibrin**: the grip spacing is assumed, the Poisson ratio is a
placeholder, and no force curve of the paper was read. Laid out by `mime scaffold plugin`; code Apache-2.0; records in
`src/mime_softtissue/records/` (the fibrin and gel records ship in the wheel, the lab's is a placeholder that is referenced
and never shipped).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Microrobotics-Simulation-Framework/mime-softtissue/blob/cleanup/notebooks/tutorial_puncture_3d.ipynb) **Puncture a clot in 3-D**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Microrobotics-Simulation-Framework/mime-softtissue/blob/cleanup/notebooks/replicate_pure_shear.ipynb) **Tearing a blood-clot sheet: a simulation-based measurement of its toughness**

## Use

Install this plugin, with its test tools; MIME comes with it:

```bash
pip install -e ".[test]"
```

Each block below that is tagged `bash ci` is what CI runs on every push (the `default` job reads this file and runs
them), so this section cannot go stale.

**1. See what it adds.** The plugin registers under the namespace `softtissue`, and the strip experiment runs (a 40-step
pull, one to two minutes on a laptop CPU):

```bash ci
mime registry | grep softtissue
python examples/tutwiler_strip.py
```

`mime registry --json` carries every entry's settings, roles, validity lines and gradient status, and is very large. To
read one namespace's:

```bash
mime registry --json | jq '.entries[] | select(.name | startswith("softtissue."))'
```

**2. A benchmark folder.** `benchmarks/build_strip.py` saves the strip as a MIME experiment with its `mime.toml`
(`benchmarks/mime.toml`, the record's manifest); `mime check` lists what is still to write in it (your creator's ORCID) and
`mime verify` loads it, rebuilds it and reruns its first steps bit for bit:

```bash ci
python benchmarks/build_strip.py
mime check benchmarks/tutwiler_strip
mime verify benchmarks/tutwiler_strip
```

**3. Its records.** `src/mime_softtissue/records/` holds the plugin's records (the `mime.assets` entry point): the fibrin record is
open (CC-BY-4.0 unless you chose another licence), so its bytes ship in the wheel; `mime assets audit` on the built sdist
(CI does it) fails if a file that may not be redistributed is in it. `mime assets list` lists MIME's own records too, with no
filter by namespace.

**4. A gradient, a tip, a fit.** `python examples/notch_gradient.py` takes the peak grip force's derivative in the notch length
through `Study.gradient` (forward mode; the notch is the body's initial state, because a dynamics' design setting is not a study
axis today). `python examples/tip_retrieval.py` presses a headed tip into a gel block as a MIME rigid body. The fitting
harness: `python examples/fit_synthetic/make_example.py OUT && python -m mime_softtissue.fit OUT/manifest.toml --out RESULT`
(the column spec is `docs/fit_spec.md`).

**5. Cite it.** `CITATION.cff`, and Tutwiler et al. 2020 for the benchmark. Third-party data, its sources and licences: [`DATA_NOTICE.md`](DATA_NOTICE.md).

## Names

The namespace `softtissue` is filled in six places from one value: the `mime.entries` and `mime.assets` entry points in
`pyproject.toml`, `src/mime_softtissue/_vocabulary.py`, the entry's name, the record's name and the instance in
`examples/`. Three separators, as MIME names things today:

- an entry, with a dot: `softtissue.notched_strip`;
- a record, with an underscore (the first records) or a dot (the later ones): `softtissue_fibrin_plasma`,
  `softtissue.whole_blood_clot`; dotted record names wait for MIME's one-`asset()` change;
- an instance of an entry on a thing, with a colon: `softtissue.grip_force:strip`.

A script imports an entry from its kind, as it does MIME's (`from mime.dynamics import RigidInertial`):

```python
from mime_softtissue.dynamics import GelBlock, NotchedStrip, ReplayedTear
from mime_softtissue.interactions import Servo, TipContact
from mime_softtissue.observations import CrackSize, GripForce, TipForce
from mime_softtissue.scene import ConeTip, HeadedTip, PunchTip
```

## Layout

```
src/mime_softtissue/
  engine/        pure JAX, no `mime` import (a test holds it): material.py (the energy law in invariants), kinematics/
                 (plane_strain, axisymmetric), fracture/ (phase_field, deletion), solvers/, contact.py, specimen.py
                 (a specimen is data), solver.py loadpath.py (the staggered step, the load steps), puncture3d.py (the 3-D
                 block's solver), readouts/
  dynamics/      the entries by kind, as MIME's packages hold MIME's, one file an entry, each kind's __init__ exporting
  interactions/  them: dynamics (gel_block, notched_strip, replayed_tear), interactions (tip_contact, servo),
  observations/  observations (tip_force, crack_size, grip_force), scene (the tips as lab geometry kinds: cone_tip,
  scene/         punch_tip, headed_tip)
  entries.py     the `mime.entries` entry point: imports every kind, which registers its entries
  _vocabulary.py the namespace and the interfaces the entries publish and accept (as `mime.vocabulary`)
  _material_record.py  a material record's values as the engine's Material (interim: OpenMaterica's, when it resumes)
  _display.py    the block's surface for display: the axisymmetric section revolved, the 3-D part mirrored
  fit/           the fitting harness (pure JAX on the engine's law; `python -m mime_softtissue.fit`; interim: OpenMaterica's)
  studies/       the studies that compare the model with published data, and the loaders of their shipped results
  records/       records.json, and the data it names in data/ (the `mime.assets` entry point): material records, and
                 every result a notebook shows (the MIME runs' results, the engine's 3-D runs, the pure-shear study's)
tests/
  conftest.py test_registry.py test_nodes.py test_validity.py test_engine_imports.py test_relative_imports.py
  engine/        closed forms, properties (Hypothesis), the energy balance, the tips' shapes and pull-out
  fit/           the fitting harness against closed forms
  plants.py test_plants.py   planted errors for every control, and their detection
  reference/ reproduction_cases.py test_reproduction.py   numbers from the prototype's untouched code, and the comparison
  verification/  the strip benchmark (slow lane)
  assert_types.py  types an editor completes from, held by pyright
benchmarks/      build_strip.py and mime.toml
examples/        what a user runs
scripts/         the measurements behind the reports (puncture, retrieval, Sneddon, probes): not part of the wheel
docs/            the fitting harness's column spec
```

## Tests

```bash
JAX_PLATFORMS=cpu taskset -c 0 pytest          # the default lane: -m "not slow"
JAX_PLATFORMS=cpu taskset -c 0 pytest -m slow  # the slow lane: the benchmark in tests/verification/
```

**Pin to one core and the CPU** (`taskset -c 0`, `JAX_PLATFORMS=cpu`). An unpinned JAX run on a shared machine
oversubscribes the cores: one measurement took 51 s unpinned for what takes 11 s pinned.

CI (`.github/workflows/ci.yml`) has a `smoke` job (a draft pull request: the tests that need no solve, about two minutes), a
`default` job (a pull request that is not a draft, main, every week: the tests on both cores with `pytest -n 2`, JAX's
compilation cache kept between runs, Python 3.12; 3.14 too on main, every week and on a pull request labelled `ready`), a
`slow` job that runs on a pull request labelled `slow` and every week, a weekly job against MIME's release branch (it tells
you the week MIME moves under you), and a `typing` job. A push to a pull request cancels its earlier run; a change that only
touches `docs/`, notebooks, the licence or `CITATION.cff` runs nothing. Run the suite in float32 and in float64
(`JAX_ENABLE_X64=1`) before you trust a number. A property test that flakes is
read, not loosened: see the comments of `tests/engine/test_properties.py`.

## Editor support

`pyrightconfig.json` is strict over `src/mime_softtissue` and `tests/assert_types.py`, and names this folder's `.venv`, so that an
editor's language server finds MIME, MADDENING and JAX there (autocomplete is only as good as the types it reads):

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test,typing]"
```

If `.venv` does not exist, pyright says so on stderr and uses the environment it is given. CI creates `.venv` itself and runs
`pyright --pythonpath "$(which python)"`. A `.venv` that exists but does not hold MIME overrides `--pythonpath` (measured),
so install into it or delete it.
