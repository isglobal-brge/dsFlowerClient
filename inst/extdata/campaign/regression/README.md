# Parkinsons subject-private regression evidence

The pre-declared epsilon-8 diagnostic **fails**: mean federated-DP RMSE is
29.55 versus 10.77 for the training-mean baseline, and mean R² is −6.92.
All three epsilon-8 replicates fail both criteria. This is a measured boundary
at the frozen registry defaults; no settings were changed after scoring.
The linear contract executed successfully, so its execution-only ridge
fallback was not used.

The protocol uses `pytorch_linear_regression`, three subject-disjoint DSLite
sites, five FedAvg rounds, three split seeds, epsilon 1/4/8, delta 1e-6, and
subject-level clipping norm 1. Training/test contain 34/8 subjects; sites have
12/11/11 training subjects. Metrics weight recordings within held-out subjects.
The unchanged patient-mode runner averages each subject's features and target.
The matched OLS twin therefore fits 19 coefficients to 34 subject means.
Its training matrices are highly collinear (condition numbers 6.89–8.67 million).
Both fitted models perform worse than the trivial baseline; the negative RMSE
gap does not demonstrate useful DP predictions. The gap also includes the
difference between five-round DP-SGD and converged OLS. Three split replicates
are descriptive and do not provide independent-cohort uncertainty estimates.

`pilot_*.json` contain per-replicate metrics, subject membership, site sizes,
public bounds, full node-reported privacy configurations, versions, runner and
model hashes, source checksums, and timings. `summary.json` contains checked
means and sample SDs. `runtime.json` records dependency versions and the
successful unscored synthetic preflight. See the
[reproduction instructions](https://github.com/isglobal-brge/dsFlowerClient/blob/evidence/representative-cells/tools/campaign/regression/README.md)
and frozen `protocol.json` for all choices.

Execution: 2026-09-22, RunPod `n4emgxhiqzy5i4`, dsFlower/dsFlowerClient 0.5.0,
R 4.6.1, CPU Torch 2.14.0. Provisioning took 39m 52s. Older Ubuntu R binaries,
Arrow installation, slow downloads, and cached uv index selection required
environment repairs; verified wheels and strict offline constraints resolved
them without package or mechanism changes.

Dataset: [UCI Parkinsons Telemonitoring, ID 189](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring),
5,875 recordings from 42 subjects, CC BY 4.0,
dataset DOI [10.24432/C5ZS3N](https://doi.org/10.24432/C5ZS3N).
Target: `total_UPDRS`; predictors: age, sex and the sixteen declared voice
measures. Bounds are fixed public clipping declarations, not observed extrema.

Citation: Tsanas A, Little MA, McSharry PE, Ramig LO (2010). Accurate
telemonitoring of Parkinson's disease progression by noninvasive speech tests.
*IEEE Transactions on Biomedical Engineering* 57(4):884–893.
[doi:10.1109/TBME.2009.2036000](https://doi.org/10.1109/TBME.2009.2036000).
