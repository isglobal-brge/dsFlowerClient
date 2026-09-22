# Regression track: Parkinsons boundary and CDC BMI

## CDC BMI alternative declared before execution — 2026-09-22

The only alternative is `pytorch_linear_regression` on **cdc45k**, the exact
45,000-respondent cohort from the thesis logistic n-scaling arm. Its existing
preparation is reusable: R seed 20260819 samples proportionately within the
binary diabetes strata and sorts the selected source rows. The original
prepared logistic cohort SHA-256 is
`5f521899386021fd6670d166f4f5ef9a4dcaa80e11d8df10942e60bd18724d3b`.
The diabetes label is used solely for cohort/split stratification and is
excluded from model inputs. Each source row is one respondent and one privacy
unit; there is no patient pooling.

Target: continuous BMI, with public, predeclared clipping bounds **[12,98]**.
This is an admissible clipping policy, not an empirical range or a claim that
the BRFSS codebook defines 98 as a universal maximum. The released neural
regression contract clips targets and retains original units; it does not
require target standardization. Features use the contract's bounded affine
transform to [-1,1], with these public bounds:

| Features | Bounds |
|---|---|
| HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex | [0,1] |
| GenHlth | [1,5] |
| MentHlth, PhysHlth | [0,30] |
| Age | [1,13] |
| Education | [1,6] |
| Income | [1,8] |

The existing logistic 80/20 split and stratified round-robin site assignment
are reused for seeds **20260820, 20260821, 20260822**: 36,000 training rows,
9,000 held-out rows, and three sites of 12,000 training respondents.
Protocol: five FedAvg rounds; epsilon **1,4,8** in that execution order;
delta **1e-6**; default row privacy; clipping norm **1**; unmodified 0.5.0
packages and registry defaults (learning rate .01, batch size 32, one local
epoch, SGD, no scheduler, no L1/L2 penalties). No model overrides or tuning.

OLS with an intercept is fitted on every training row of the same split,
using the same public feature transform and clipped training targets.
The trivial baseline predicts the raw training target mean. Held-out metrics
are RMSE, MAE, and R² in BMI units; gap = federated-DP RMSE minus OLS RMSE.
At epsilon 8, RMSE below the trivial RMSE and R² > 0 are annotations, not
acceptance gates. Each replicate is scored once after training; held-out values
are routed during preparation but never inspected or used for choices.
Completed scores are immutable. No additional alternative will be run.

If a cdc45k replicate exceeds about 15 minutes on the provisioned 32-vCPU pod,
the pre-authorized cdc9k size fallback may be selected solely on elapsed time,
before inspecting scores; any started attempt will be retained and documented.
The selected cohort and any fallback are recorded in the execution evidence.

Source: [UCI CDC Diabetes Health Indicators, ID 891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators),
[source CSV](https://archive.ics.uci.edu/static/public/891/data.csv),
dataset DOI [10.24432/C53919](https://doi.org/10.24432/C53919), thesis citation
key **`uci_cdc_diabetes_health_indicators`**. Public coding references:
[CDC BRFSS 2015 documentation](https://www.cdc.gov/brfss/annual_data/annual_2015.html).
Source, cohort, split, protocol and tooling checksums accompany the results.

## Parkinsons correction and boundary interpretation

The original `pilot_parkinsons_*.json`, `protocol.json`, and runner scripts are
retained unchanged. Their 34-subject-mean central OLS is annotated as a
**twin-computation error relative to the requested all-recording OLS reference**.
The added corrected twin fits all training recordings on the same saved
splits and seeds; only its central metrics and corresponding gaps are new.
Federated-DP models and scores are preserved byte for byte.

The source audit also matters: the released patient-mode runner really does
pool each subject's features and continuous target before DP-SGD. Therefore
the requested all-recording OLS is a recording-level reference, rather than an
exact twin of that pooled training representation. The original 34-row OLS
matched the pooled representation but was unsuitable for the requested
recording-level comparison. Neither central calculation isolates DP noise.

The cohort provides only 42 privacy units. After the eight-subject holdout,
the actual training sites contain **12/11/11 subjects**, rather than 14 each.
Subject privacy with only tens of units per site is outside the useful regime
of this measured five-round, unit-clipped, registry-default protocol across
epsilon 1/4/8. This is a documented utility boundary, not evidence of a
dsFlower implementation defect. The measurement does not establish that
every possible optimizer or protocol must fail at those privacy budgets.

## Original Parkinsons declaration (historical; central reference superseded above)

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
The gap compares the requested five-round federated DP-SGD protocol with
converged noiseless OLS; it does not isolate the causal effect of DP noise from
optimization and federation. Subject pooling leaves only 34 OLS training rows.

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

For slow pod downloads, `fetch_wheels.py <wheel-directory>` downloads the 60
Linux CPython 3.11 wheels in `pylock.toml` and verifies every SHA-256. Transfer
that directory to `/workspace/cells/wheels/` before package installation;
`provision.sh` then validates the checksums and instructs uv to install locally
with offline mode and `runtime-constraints.txt`. Both are needed because cached
index entries can otherwise select newer versions absent from the wheel set.
The lock satisfies the release package requirements and pins CPU Torch 2.14.0.
It uses PyTorch's official `download.pytorch.org` mirror because the alternate
`download-r2.pytorch.org` host returned HTTP 403 for the same hashed artifacts.
The executed provisioning transferred the wheel archive in parallel 8 MiB
chunks over SSH because individual pod connections were throttled. No runtime
or model code was changed for this transport workaround.

The driver refuses to overwrite or restart a started cell. Use a fresh work
root for an independent replication; do not delete run guards to seek a better
score. `summarize.py` checks split disjointness, identical central/trivial
baselines across epsilons, node privacy settings, completed rounds, finite
metrics, and the reported means/SDs before creating `summary.json`.
