# Regression schedule diagnosis and corrected cell

The original diagnostic tools below use only training rows. The separately
authorized corrected-cell implementation is documented at the end of this
file. Package sources remain unchanged; only the corrected-cell guarded
scoring callback opens sealed test files.

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

## Corrected-cell implementation

`stage_selection.py` and `select_federated.R` reuse the diagnosis inner partition
for each original training split and confirm the top two candidates under the
real epsilon-8 contract. `select.sh` ran all six guarded fits in the foreground.
`finish_selection.py` requires all six results and absence of sealed test CSVs,
then selects by mean validation RMSE. `selection.json` retains every replicate,
node policy, position/source-id hashes, and the selected schedule.

The corrected cell was declared in both track READMEs and committed before
execution; `declaration_commit.txt` identifies that commit. On the same pod:

```sh
bash /workspace/cells/r4-client/tools/campaign/regression/r4/run_corrected.sh
```

Do not rerun this command after execution: immutable budget and scoring guards
refuse repeat runs. Each epsilon/seed trains real pooled DP at one site and
real federated DP at three sites. `comparators.py fit` freezes OLS and the
verified noiseless finite-schedule twin before scoring. Only the guarded final
callback reconstructs and opens the exact original test CSV. All five arms
are scored in BMI units; the three noiseless reference fits/scores are reused
across budgets. There are no tuning branches after test scoring.

`selected_accounting.py` records complete-horizon calibration without data.
`execution_audit.py` records guards, timings and model byte hashes without
rescoring. `package_evidence.py` validates saved metrics, mean/sample SD, gaps,
node settings, split identities, tooling hashes and comparator reuse, then
updates the four-cell track summary. Private runtime files, raw rows and
prediction tables remain on the pod; only aggregate evidence is published.
