# Regression evidence: Parkinsons boundary and CDC BMI

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
