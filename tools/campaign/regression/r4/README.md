# Regression schedule diagnosis

These tools diagnose the existing `cdcbmi_public_units` results using only the
original training rows. They do not declare or execute another evaluation cell.
The package sources and the sealed test files are untouched.

`prepare_training.py` reuses the original R cohort and split functions, verifies
the UCI source and all frozen training/site checksums, and writes no test CSV.
`emulate.py` uses analytic linear gradients and stock optimizers to reproduce
the released finite schedule. It checks individual updates against Opacus before
running any emulation. Its diagnostic random seeds are public and its outputs
are **not private releases**. Noise/no-noise pairs share initialization and
sampling; noise has a separate stream. The Gaussian multiplier comes directly
from the unchanged runner, including its replace-one adjacency conversion.

The inner holdout takes 2,400 positions from each original site using NumPy
`default_rng(20260922 + site).permutation(12000)`. The remaining 9,600 positions
are inner training. All three outer split seeds are used independently, with
initialization seeds 101, 202, 303, 404 and 505. The original 12,000-row geometry
is also emulated and scored on its training rows only. No outer test metric is
recomputed. The controls are explanatory counterfactuals; they are not proposals
to change initialization, clipping or the privacy contract.

On the regression pod, with the lightweight branch tooling at
`/workspace/cells/r4-client` and unchanged tag packages at `/workspace/cells`:

```sh
export R_LIBS=/workspace/cells/Rlib
export DSFLOWER_VENV_ROOT=/workspace/cells/venvs
export DSFLOWER_CLIENT_VENV_ROOT=/workspace/cells/client
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 /workspace/cells/r4-client/tools/campaign/regression/r4/prepare_training.py
/workspace/cells/venvs/pytorch/bin/python \
  /workspace/cells/r4-client/tools/campaign/regression/r4/emulate.py \
  --runner /workspace/cells/dsFlower/inst/flower_app \
  --protocol /workspace/cells/r4-client/tools/campaign/regression/cdcbmi_public_units_protocol.json \
  --mode baseline --output /workspace/cells/r4/baseline.json
```

The `candidates` and `controls` modes use the same inputs and separate output
JSONs. All candidate schedules keep five rounds, clipping and public bounds.
The candidate sweep is non-private, training-only diagnosis.

`stage_inner.py` stages the first split's identical inner partition for
`run_federated.R`. That driver permits exactly one default-schedule epsilon-8
training and retains its released model. It refuses to repeat a started run.
`verify_predictions.py /workspace/cells` independently loads its released
coefficients and checks both training and inner-validation predictions against
the package prediction API. Original campaign run guards and records are not
modified. Do not invoke the historical cell drivers for this diagnosis.

`resume_provision.sh` records how the interrupted installation was completed
from the already downloaded v0.5.0 source archives and verified wheel cache.
It is a continuation script, not a general empty-machine provisioner; the
track's original `provision.sh` and `install_dependencies.R` provide the initial
system and R dependency setup.
