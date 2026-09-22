# Regression evidence: target scale and public-unit BMI

The track contains three cells: Parkinson subject-level privacy, raw-unit CDC
BMI, and the separately declared public-unit CDC BMI cell. All use unchanged
`pytorch_linear_regression`, dsFlower/dsFlowerClient 0.5.0, three sites, five
FedAvg rounds, three seeds, epsilon 1/4/8, delta 1e-6 and unit clipping.

The [pre-execution diagnosis](REGRESSION_DIAGNOSIS.md) uses only existing scores
and release source. The raw cells show approximately epsilon-independent RMSE,
with excess-error scales around 27.5 UPDRS and 8 BMI units. The previous runs
passed target bounds, but the runner only clips targets while scaling features;
it does not normalize targets or invert their units. Exact old signed bias is
not identifiable from the saved scalar metrics. Parkinson has only 12/11/11
training subjects/site and remains outside the measured useful regime. The raw
BMI gap is optimization-limited and must not be called a privacy cost.

The [third-cell declaration](../../../../tools/campaign/regression/README.md)
was committed before execution as `66dc8222be3ad8b9f33ddf15b15b620586d325a5`,
published unchanged after a required rebase as
`15aa3bdf6c417f6df6a5407b72bb7b73452e5bcf`. The public-unit target is
`y'=(clip(BMI,12,98)-55)/43`; predictions return to BMI as `55+43*y'_hat`.
The client harness transforms training targets and passes [-1,1] target bounds;
the unchanged runner scales features through their original public bounds.
The public coordinate origin is BMI 55; random linear initialization remains
unchanged. All model defaults remain .01 learning rate, batch 32, one local
epoch, SGD, no scheduler or penalties. No extension was needed for the
intercept-motion reachability bound, and no convergence claim follows from it.

The same cdc45k cohort and prepared splits/seeds are reused: 45,000 respondents,
36,000 training rows, 9,000 held-out rows, three sites of 12,000. OLS fits the
same training rows in the same public units; both model predictions are scored
in original BMI units. Trivial predicts the raw training mean. Held-out files
are first opened inside each guarded final scoring callback. RMSE below trivial
and R²>0 at epsilon 8 are annotations only. This is the operational privacy-cost
measurement after addressing target scale; the DP-minus-OLS gap still includes
optimization, clipping and federation effects and is not a noise-only estimate.

Provenance: [UCI CDC Diabetes Health Indicators, ID 891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators),
DOI [10.24432/C53919](https://doi.org/10.24432/C53919), thesis key
**`uci_cdc_diabetes_health_indicators`**. The reused BMI interval [12,98] is a
public declared clipping policy, not empirical cohort bounds or a claim about
universal BRFSS endpoints. Original cohort/split checksums remain unchanged.

## Completed three-cell comparison

| Contract | Dataset/cell | n | Privacy unit | ε | Central RMSE | DP RMSE | Trivial RMSE | DP R² | Gap mean ± SD | Diagnostics |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| pytorch_linear_regression | Parkinson | 5,875 | subject | 1 | 11.309447 | 29.503164 | 10.771196 | -6.879554 | 18.193717 ± 3.082866 | No/No |
| pytorch_linear_regression | Parkinson | 5,875 | subject | 4 | 11.309447 | 29.456259 | 10.771196 | -6.883872 | 18.146812 ± 4.167444 | No/No |
| pytorch_linear_regression | Parkinson | 5,875 | subject | 8 | 11.309447 | 29.549561 | 10.771196 | -6.919012 | 18.240114 ± 3.516164 | No/No |
| pytorch_linear_regression | CDC BMI raw units | 45,000 | respondent row | 1 | 6.195724 | 10.371563 | 6.580037 | -1.484565 | 4.175839 ± 0.070229 | No/No |
| pytorch_linear_regression | CDC BMI raw units | 45,000 | respondent row | 4 | 6.195724 | 10.398831 | 6.580037 | -1.497708 | 4.203107 ± 0.110651 | No/No |
| pytorch_linear_regression | CDC BMI raw units | 45,000 | respondent row | 8 | 6.195724 | 10.414030 | 6.580037 | -1.504846 | 4.218306 ± 0.120771 | No/No |
| pytorch_linear_regression | CDC BMI public units | 45,000 | respondent row | 1 | 6.195724 | 6.752820 | 6.580037 | -0.053274 | 0.557096 ± 0.010386 | No/No |
| pytorch_linear_regression | CDC BMI public units | 45,000 | respondent row | 4 | 6.195724 | 7.096150 | 6.580037 | -0.168634 | 0.900426 ± 0.545750 | No/No |
| pytorch_linear_regression | CDC BMI public units | 45,000 | respondent row | 8 | 6.195724 | 6.866393 | 6.580037 | -0.092664 | 0.670670 ± 0.437716 | No/No |

Diagnostics report the mean RMSE-below-trivial / mean R²-positive annotations.
SD is the sample SD over three split replicates, not a confidence interval.
Parkinson has 42 subjects in total and 12/11/11 training subjects per site.

The public-unit cell completed all nine prescribed executions once. Its mean
DP RMSEs are **6.752820 / 7.096150 / 6.866393** at epsilon 1/4/8, compared with
raw-unit **10.371563 / 10.398831 / 10.414030**. Central OLS remains **6.195724**;
the public transform changes this comparator by less than 1e-9 RMSE. Trivial
remains **6.580037**. At epsilon 8 the paired gap is **0.670670 ± 0.437716**,
and mean DP R² is **-0.092664**: both mean diagnostic annotations are false.
One of three epsilon-8 replicates meets both annotations (seed 20260822);
all three are retained. No setting was selected from these results.

The signed residual mean (prediction minus BMI) ranges from **-0.535610 to
+0.414520 BMI** across all nine new runs; the model reaches the target's mean
location. Exact residual decompositions are saved and checked without rescoring.
Removing the raw-unit scale mismatch substantially reduces error, but mean
utility still trails the trivial predictor and there is no monotonic epsilon
response. The public-unit gap is the requested operational privacy-cost
measurement, with remaining optimization, clipping, initialization variability
and federation effects; these data do not isolate the causal cost of noise.

| Public-unit ε | DP RMSE mean ± SD | DP MAE mean ± SD | DP R² mean ± SD |
|---:|---:|---:|---:|
| 1 | 6.752820 ± 0.058812 | 4.782874 ± 0.049697 | -0.053274 ± 0.001975 |
| 4 | 7.096150 ± 0.501409 | 4.985946 ± 0.337625 | -0.168634 ± 0.178670 |
| 8 | 6.866393 ± 0.423674 | 4.794027 ± 0.258125 | -0.092664 ± 0.146712 |

Runs took 85.56–96.14 seconds each (13.48 minutes summed runtime).
All 45 federation rounds completed with three clients and no failures; cleanup
passed. The pod remains running. Privacy accounting is per training, with no
combined-epsilon claim for the grid; split seeds do not fix cryptographic DP
randomness or model initialization.

`cdcbmi_public_units_pytorch_linear_regression_eps*.json` contain the scores,
residual decompositions, node settings, histories and model/tooling hashes.
`cdcbmi_public_units_execution_audit.json` records the execution-to-publication
commit mapping, 3 started-cell guards, 9 scoring guards, verified saved model
hashes, immutable original files and preflight. `summary.json` lists all three
cells and their interpretations. `summarize_public_units.py` checks all nine
budget artifacts, original preservation, cohort/split identity, bounds/defaults,
privacy, rounds, baselines, metrics, means/SDs/gaps and residual identities.
No held-out values were read outside scoring, no scored cell was rerun, and no
further alternative was run. Source package code and previous harnesses remain
unchanged. Raw rows, model artifacts and node credentials stay on the pod.

## Historical cells and original declarations

The prior “no further alternative” statements below describe their completed
sessions. The third public-unit cell is separately declared and authorized.
Original scored JSONs and the Parkinson central correction remain unchanged.


Both cells use `pytorch_linear_regression`, unchanged dsFlower/dsFlowerClient
0.5.0, three sites, five FedAvg rounds, seeds 20260820/20260821/20260822,
epsilon 1/4/8, delta 1e-6, clipping norm 1, and registry defaults. Metrics are
held-out RMSE and MAE in original target units, and dimensionless R². The gap is paired
federated-DP RMSE minus central OLS RMSE; SD is the sample SD over three splits.

| Dataset | n | Privacy unit | ε | Central RMSE | DP RMSE | Trivial RMSE | DP R² | Gap mean ± SD | Diagnostics |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Parkinsons | 5,875 | subject | 1 | 11.309447 | 29.503164 | 10.771196 | -6.879554 | 18.193717 ± 3.082866 | No/No |
| Parkinsons | 5,875 | subject | 4 | 11.309447 | 29.456259 | 10.771196 | -6.883872 | 18.146812 ± 4.167444 | No/No |
| Parkinsons | 5,875 | subject | 8 | 11.309447 | 29.549561 | 10.771196 | -6.919012 | 18.240114 ± 3.516164 | No/No |
| CDC BMI (cdc45k) | 45,000 | respondent row | 1 | 6.195724 | 10.371563 | 6.580037 | -1.484565 | 4.175839 ± 0.070229 | No/No |
| CDC BMI (cdc45k) | 45,000 | respondent row | 4 | 6.195724 | 10.398831 | 6.580037 | -1.497708 | 4.203107 ± 0.110651 | No/No |
| CDC BMI (cdc45k) | 45,000 | respondent row | 8 | 6.195724 | 10.414030 | 6.580037 | -1.504846 | 4.218306 ± 0.120771 | No/No |

Diagnostics mean “DP RMSE below the training-mean predictor” and “DP R² > 0”.
They are annotations, not acceptance gates. No settings were changed after
scoring, no scored model was retrained, and no further alternative was run.
The three split replicates are descriptive, not independent-cohort confidence
intervals. Privacy accounting is per training; the grid has no combined-epsilon
claim, and split seeds do not seed the node's cryptographic DP randomness.

## Parkinsons: corrected reference and subject-privacy boundary

The original `pilot_parkinsons_*.json` files remain byte-for-byte unchanged,
including federated metrics, model hashes and the original 34-subject-mean OLS.
That central calculation is retained and annotated as a **twin-computation
error relative to the requested all-training-recording OLS reference**.
`parkinsons_corrected_central_twin.json` adds OLS on all training recordings
using the exact saved splits, seeds, public transforms and target bounds.
It records original hashes, split CSV hashes, coefficients and source excerpts.
No federated model was loaded, retrained or rescored for the correction.

Corrected central RMSE is **11.309447 ± 2.925426**, compared with the original
**41.659715 ± 9.844498** and trivial **10.771196 ± 0.431560**. Corrected central
MAE is **9.710291 ± 2.406439** and R² is **−0.181958 ± 0.532994**.

| Seed | Training recordings | Held-out recordings | Corrected central RMSE |
|---|---:|---:|---:|
| 20260820 | 4,751 | 1,124 | 9.951068 |
| 20260821 | 4,793 | 1,082 | 14.667110 |
| 20260822 | 4,770 | 1,105 | 9.310162 |

A source audit corrects a premise of the requested comparison: the released
patient-mode runner actually averages each subject's transformed features and
continuous target before DP-SGD. Thus the new all-recording OLS is the requested
recording-level reference, while the original OLS matched the pooled training
representation. The corrected gap includes that fitting-unit difference and
five-round optimization versus converged OLS; it is not a noise-only effect.

The 5,875 recordings contain only 42 privacy units. Holding out eight subjects
leaves 34 training subjects, distributed **12/11/11**, rather than 14 per site.
Subject privacy with only tens of units per site is outside the useful regime
of this measured unit-clipped, five-round, registry-default protocol across
the epsilon grid. The cell remains a documented utility boundary, not a
dsFlower implementation defect. It does not prove every optimizer or protocol
must fail at these budgets. The corrected OLS is also slightly worse on average
than the mean predictor. No causal claim about DP noise alone is warranted.

Dataset: [UCI Parkinsons Telemonitoring, ID 189](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring),
5,875 recordings from 42 subjects, CC BY 4.0,
DOI [10.24432/C5ZS3N](https://doi.org/10.24432/C5ZS3N).
Target: `total_UPDRS`, bounds [0,176]; predictors: age, sex and the sixteen
voice measures in the unchanged original `protocol.json`.
Citation: Tsanas A, Little MA, McSharry PE, Ramig LO (2010). Accurate
telemonitoring of Parkinson's disease progression by noninvasive speech tests.
*IEEE Transactions on Biomedical Engineering* 57(4):884–893.
[doi:10.1109/TBME.2009.2036000](https://doi.org/10.1109/TBME.2009.2036000).

## CDC BMI: the single predeclared alternative

The selected cohort is **cdc45k**, the exact 45,000-person fixed-seed stratified
cohort used by the thesis logistic n-scaling arm. The original preparation was
reused and its prepared CSV hash matched
`5f521899386021fd6670d166f4f5ef9a4dcaa80e11d8df10942e60bd18724d3b`.
Each source row is one respondent and one privacy unit. Stratification uses
only the original diabetes label; that label and source IDs are excluded from
model inputs. The three original logistic split seeds give 36,000 training
rows, 9,000 held-out rows and three sites of 12,000 training respondents.
All row memberships and disjointness checks are retained.

BMI is the continuous target, with predeclared admissible clipping bounds
**[12,98]**. This is a clipping policy, not an empirical range or a claim that
BRFSS defines 98 as a universal maximum. The released contract clips targets
and retains original units; no target standardization is required. The 20
features and public bounds are:

| Features | Bounds |
|---|---|
| HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex | [0,1] |
| GenHlth | [1,5] |
| MentHlth, PhysHlth | [0,30] |
| Age | [1,13] |
| Education | [1,6] |
| Income | [1,8] |

The central OLS fits every training row after the same clipped affine feature
transform and target clipping. The trivial predictor is the raw training BMI
mean. Held-out targets and predictions remain in original BMI units. Held-out
rows were routed during preparation but never analyzed or used for setting
selection; the training process opens its held-out file only in the scoring
callback. A score guard and immutable started-cell files prevent rescoring.

At epsilon 8, mean federated-DP RMSE is **10.414030**, versus trivial
**6.580037**, and mean R² is **−1.504846**. Both annotations are false for
every replicate and for the means. Central OLS achieves **6.195724** RMSE
and **0.113335** R². This one-row-per-person cohort supplies many more privacy
units, but the measured five-round default protocol still does not produce a
useful BMI predictor under these diagnostics. The gap includes optimization
and federation differences; these runs cannot attribute it to DP noise alone.
The result is retained without tuning, retries or another alternative. All nine
replicates completed in 75.76–94.83 seconds (13.26 minutes summed runtime).

The README declaration preceded preparation and training. Its execution-time
commit is `9796615`; the publication mapping after the required rebase is in
`cdcbmi_execution_audit.json`. The first replicate took 83 seconds, so **cdc9k
was not run**. All runs were foreground on the existing 32-vCPU
`pod-flower-regression` (`n4emgxhiqzy5i4`), with no provisioning or release
changes. The pod is left running. The official CSV was downloaded locally and
transferred to avoid a slow pod download. An integer conversion in new
preparation tooling was corrected before any model execution; it preserves
the exact original cohort sampler.

Dataset: [UCI CDC Diabetes Health Indicators, ID 891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators),
source population 253,680 respondents, CC BY 4.0,
DOI [10.24432/C53919](https://doi.org/10.24432/C53919).
Citation: CDC Diabetes Health Indicators [Dataset]. (2017). UCI Machine
Learning Repository. Thesis citation key: **`uci_cdc_diabetes_health_indicators`**.
Public coding reference: [CDC BRFSS 2015 codebook](https://www.cdc.gov/brfss/annual_data/2015/pdf/CODEBOOK15_LLCP.pdf).
Official [source CSV](https://archive.ics.uci.edu/static/public/891/data.csv)
SHA-256: `9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964`.

## Evidence and reproduction

- `summary.json`: both datasets, all budgets, checked means/SDs, diagnostics and interpretations.
- `pilot_parkinsons_*.json`: immutable original evidence.
- `parkinsons_corrected_central_twin.json`: additive corrected OLS reference and gaps.
- `cdcbmi_pytorch_linear_regression_eps*.json`: all nine new scored replicates, node-reported settings, model/tooling hashes and timings.
- `cdcbmi_provenance.json`, `cdcbmi_CHECKSUMS.sha256`: source/cohort/protocol/tooling provenance and cache checksums.
- `cdcbmi_split_seed*.json`: public row memberships; raw respondent CSVs stay on the pod.
- `cdcbmi_execution_audit.json`: preservation/release checks and declaration commit mapping.
- `runtime.json`: existing provisioned runtime and unscored synthetic preflight.

See [`tools/campaign/regression/README.md`](../../../../tools/campaign/regression/README.md)
and `cdcbmi_protocol.json` for the declaration and foreground commands.
`summarize_regression.py` validates the preserved originals, corrected gaps,
CDC row membership, baseline consistency, privacy/defaults/rounds, hashes and
all summary statistics. `run_cell.R`, `campaign_lib.R`, `central_train.py` and
release source were not edited. No secrets or raw model-training rows are
included in the published evidence. The laptop evidence mirror is
`~/Documents/GitHub/dsflower-cells/evidence/regression/`.
