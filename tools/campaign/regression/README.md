# Subject-private Parkinsons regression

The frozen design is in `protocol.json`. It uses the unchanged dsFlower and
dsFlowerClient 0.5.0 packages, `pytorch_linear_regression` registry defaults,
three DSLite PSOCK workers, loopback SuperNodes/SuperLink, five FedAvg rounds,
and three subject-split seeds at each epsilon 1, 8, 4. `campaign_lib.R` copies
the existing campaign federation plumbing; its changes supply regression target
bounds, retain node-reported privacy metadata, and score continuous predictions.
No package, canonical runner, shared campaign file, or privacy mechanism changes.

## Dataset and public bounds

[UCI Parkinsons Telemonitoring (189)](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)
contains 5,875 recordings of 42 people, licensed CC BY 4.0. Download source:
`https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip`.
The downloader records the archive, raw CSV and documentation SHA-256 hashes.

Citation: Tsanas A, Little MA, McSharry PE, Ramig LO (2010). Accurate
telemonitoring of Parkinson's disease progression by noninvasive speech tests.
*IEEE Transactions on Biomedical Engineering* 57(4):884–893.
[doi:10.1109/TBME.2009.2036000](https://doi.org/10.1109/TBME.2009.2036000).
Dataset DOI: [10.24432/C5ZS3N](https://doi.org/10.24432/C5ZS3N).

Target: `total_UPDRS`; features: age, sex, and the sixteen voice measures in
`protocol.json`. Subject identifier, motor_UPDRS and test_time are excluded from
features. `subject#` is the node-pinned patient column. Bounds were declared
before downloading/inspecting recordings. They are fixed admissible clipping
ranges, not empirical extrema or assertions of universal physical maxima.

| Variables | Lower | Upper |
|---|---:|---:|
| age | 18 | 100 |
| sex | 0 | 1 |
| Jitter(%), Jitter:RAP, Jitter:PPQ5 | 0 | 0.1 |
| Jitter(Abs), seconds | 0 | 0.001 |
| Jitter:DDP | 0 | 0.3 |
| Shimmer, Shimmer:APQ3, Shimmer:APQ5, Shimmer:APQ11 | 0 | 0.5 |
| Shimmer(dB) | 0 | 5 |
| Shimmer:DDA | 0 | 1.5 |
| NHR | 0 | 2 |
| HNR, dB | -20 | 60 |
| RPDE, DFA, PPE | 0 | 1 |
| total_UPDRS | 0 | 176 |

The public UCI dictionary defines the variables and binary sex coding. Adult
age and nonnegative acoustic perturbation measures inform the round-number
engineering caps. [Praat's shimmer definitions](https://praat.org/manual/Voice_3__Shimmer.html)
give the DDA = 3 × APQ3 relation; [HNR documentation](https://praat.org/manual/Harmonicity.html)
specifies dB units. Bounds for normalized nonlinear descriptors are a declared
[0,1] clipping policy. The [UPDRS I–III scale](https://www.ncbi.nlm.nih.gov/books/NBK539553/table/cl5.tab14/)
has total range 0–176. Values outside these declared domains are clipped using
the released package's transform; no bounds are estimated from train or test.

## Frozen split, central twin, and diagnostic

For seeds 20260820, 20260821, 20260822, R samples 8 of the 42 sorted subject IDs
for test, shuffles the remaining 34, and deals them round-robin to the three
sites (12/11/11 subjects). No subject crosses a split or site. Test values are
not used for selection, exploration or scoring before the final callback.

The canonical patient-mode runner clips and transforms features, then averages
features and continuous outcomes within each subject. The central twin fits
OLS with an intercept to the same 34 training subject means, using NumPy's
default least-squares rank determination. Both predict individual recordings
from held-out subjects. Metrics are recording-weighted RMSE, MAE and R² in
original UPDRS units. The trivial baseline predicts the recording-weighted
training target mean. No prediction clipping is added.

The acceptance diagnostic is mean federated-DP RMSE below mean trivial RMSE
and mean federated-DP R² above zero at epsilon 8. Per-seed decisions are also
reported. No tuning or inner validation is performed. A utility failure is
retained; ridge is reserved solely for an execution failure of the linear
contract. Three split seeds do not control node secrets or the package's
cryptographic DP randomness, and the repeated trainings have no combined
epsilon guarantee. The 5,875 recordings provide only 42 privacy units.

## Reproduce

Rsync the release sources into `/workspace/cells/dsFlower` and
`/workspace/cells/dsFlowerClient`, then run the following in the foreground:

```sh
bash /workspace/cells/dsFlowerClient/tools/campaign/regression/provision.sh
python3 /workspace/cells/dsFlowerClient/tools/campaign/regression/prepare_data.py --root /workspace/cells
export R_LIBS=/workspace/cells/Rlib
export DSFLOWER_VENV_ROOT=/workspace/cells/venvs
export DSFLOWER_CLIENT_VENV_ROOT=/workspace/cells/client
export DSFLOWER_NODE_SECRET_FILE=/workspace/cells/smoke/parent-node-secret
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
Rscript /workspace/cells/dsFlowerClient/tools/campaign/regression/preflight.R /workspace/cells
bash /workspace/cells/dsFlowerClient/tools/campaign/regression/run.sh /workspace/cells
```

Provisioning installs R from CRAN's Ubuntu jammy repository, system libraries,
uv, current R dependencies from Posit's jammy binary repository, and both
packages through their unmodified configure scripts. This avoids incompatible
Ubuntu R 4.1 package binaries under R 4.6 and a full libarrow source build.
CPU Torch uses `/workspace/cells/venvs/pytorch`; the client environment is
`/workspace/cells/client/venv`; R packages are in `Rlib/`. Download checksums are
in `data_cache/CHECKSUMS.sha256`. All emitted evidence is under
`inst/extdata/campaign/regression/`; local model artifacts and node state remain
in `/workspace/cells/runs/`. Never publish the node secret files.

The driver refuses to overwrite or restart a started cell. Use a fresh work
root for an independent replication; do not delete run guards to seek a better
score. `summarize.py` checks split disjointness, identical central/trivial
baselines across epsilons, node privacy settings, completed rounds, finite
metrics, and the reported means/SDs before creating `summary.json`.
