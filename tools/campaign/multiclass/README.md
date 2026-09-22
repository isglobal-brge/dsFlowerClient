# Representative multiclass utility cells

This directory extends the released campaign harness for `pytorch_multiclass`.
`campaign_lib.R` copies the DSLite-per-PSOCK-worker federation, local SuperLink
and SuperNode lifecycle, stratified split, bounds rule and cleanup from
`../campaign_lib.R`; the task-specific changes are three target levels,
multiclass scoring, and capture of each node's privacy configuration.
The shared harness, package code, registry and canonical runner are unchanged.

The original CTG design is frozen in `protocol.json`, before any test scoring.
All v0.5.0 registry defaults apply, including one local epoch (the older binary
pilot's two-epoch override is not inherited). There are three sites, five
rounds and three paired split seeds per epsilon, in execution order 1, 8, 4.
Central training is unregularized `nnet::multinom` with the same clip-and-affine
transform, fitted only on the pooled training split. The trivial prediction
uses training class proportions and therefore predicts its majority class.
No tuning or inner validation is performed. Epsilon 8 is assessed against
strictly better-than-majority mean accuracy and macro-AUC above 0.5; each
replicate is also disclosed. No fallback was specified in that original protocol;
the separately authorized HAR561 follow-up is documented below.

Split seeds control cohort partitioning and central initialization. Node-owned
DP randomness stays cryptographic and sticky, and is not reproducible from the
published seeds. Each independent node training has its own epsilon contract;
this campaign does not assert one composed epsilon for all nine federated fits.
Bounds follow the original public-cohort TRAIN-extrema rule, widened by 10%.
They never use test extrema. This simulation does not treat private empirical
bounds as freely publishable statistics.

## Dataset and citation

[UCI Cardiotocography, id 193](https://archive.ics.uci.edu/dataset/193/cardiotocography),
2,126 records and all 21 standard CTG measurements. The target is `NSP`
(1 normal, 2 suspect, 3 pathologic); `CLASS` is excluded. The official
[CSV](https://archive.ics.uci.edu/static/public/193/data.csv) is used without
subsampling or row exclusions. License: CC BY 4.0. Dataset DOI:
[10.24432/C51S4N](https://doi.org/10.24432/C51S4N).

Ayres-de-Campos D, Bernardes J, Garrido A, Marques-de-Sá J, Pereira-Leite L.
(2000). *SisPorto 2.0: A program for automated analysis of cardiotocograms*.
Journal of Maternal-Fetal Medicine 9:311–318.
DOI: `10.1002/1520-6661(200009/10)9:5<311::AID-MFM12>3.0.CO;2-9`.

CSV SHA-256:
`4648b5bf338d18f0c7e030a5d2cfb2018c831695cdb90129912cf00367c7d751`.

## Reproduction

Use Ubuntu 22.04 with R >= 4.4 from CRAN's jammy apt repository, C/C++/Fortran
toolchains, libcurl/OpenSSL/XML development libraries and `uv`. Under a fresh
`/workspace/cells`, place the v0.5.0 package sources in `dsFlower/` and
`dsFlowerClient/`. Keep the external runtime directories `Rlib/`,
`venvs/pytorch/`, `client/venv/`, and `data_cache/`.

```sh
cd /workspace/cells
mkdir -p Rlib data_cache
Rscript dsFlowerClient/tools/campaign/multiclass/install.R
export R_LIBS=/workspace/cells/Rlib
export DSFLOWER_VENV_ROOT=/workspace/cells/venvs
export DSFLOWER_CLIENT_VENV_ROOT=/workspace/cells/client
R CMD INSTALL --library=Rlib dsFlower
R CMD INSTALL --library=Rlib dsFlowerClient
curl -fL https://archive.ics.uci.edu/static/public/193/data.csv \
  -o data_cache/cardiotocography.csv
(cd data_cache && sha256sum cardiotocography.csv > CHECKSUMS.sha256)
bash dsFlowerClient/tools/campaign/multiclass/run.sh /workspace/cells
```

`check_metrics.R` checks perfect predictions, constant probabilities,
independent pairwise AUC with ties, multiclass log-loss and clipped scaling
without using the real test data. Each cell refuses to overwrite a previous
artifact or started run. Each replicate retains model artifacts, history,
node secrets and logs on the pod; secrets must not be copied into evidence.
JSON metrics retain full numeric precision. All package and runner hashes,
node-reported policies, split/site sizes, bounds and elapsed times are recorded.

The gap compares a fully converged pooled central model with the fixed
five-round DP federation; it includes optimization, federation and privacy
costs, and is not a causal estimate of the privacy mechanism alone. This is a
single public-data cell with three random partitions, not evidence about other
multiclass datasets or non-IID site distributions.

For the original CTG-only campaign, validate and summarize its JSONs without
rescoring (use `summarize_har561.py` below once both datasets are present):

```sh
python3 tools/campaign/multiclass/summarize.py inst/extdata/campaign/multiclass
```

`environment.py` captures installed dependency versions and provisioning
start/end times. During this run the client CPU torch runtime was provisioned
before scoring with `uv pip install --torch-backend cpu` and its remaining
prediction dependencies checked with the unchanged package helper
`dsFlowerClient:::.ensure_client_framework("pytorch")`.

CTG's epsilon-8 accuracy diagnostic failed in all three replicates;
see the [committed results](../../../inst/extdata/campaign/multiclass/README.md).
All nine CTG fits completed with unchanged settings and no reruns. This is a
calibration boundary: ranking was retained, while argmax remained at the majority
class, consistent with the MLP cells' calibration observation. The original CTG
protocol, settings and scored evidence are retained unchanged.

## HAR561 follow-up

[har561_protocol.json](har561_protocol.json) freezes the one user-declared
alternative, authorization `FLOWER_CELLS_MULTICLASS_2026-09-22`.
[UCI HAR, id 240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
provides 561 engineered features for 10,299 windows, 30 subjects and six activities:
the sequence track's collection in tabular form. The official subject-disjoint
split has 7,352 training windows/21 subjects and 2,947 test windows/nine subjects.
Each seed shuffles training subjects into three sites of seven, paired across
epsilon. Public feature bounds are `[-1,1]`.

Anguita D, Ghio A, Oneto L, Parra X, Reyes-Ortiz JL. (2013). *A Public Domain
Dataset for Human Activity Recognition Using Smartphones*. ESANN 2013.
Dataset DOI: [10.24432/C54S4K](https://doi.org/10.24432/C54S4K).
UCI lists CC BY 4.0; the historical archive README separately states noncommercial
use and required citation. Official archive SHA-256:
`c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.
Nested dataset archive SHA-256:
`2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0`.

`har561.R` loads one split; `har561_campaign.R` adds six target levels and
node-pinned patient privacy on `subject`. The unchanged release averages bounded
features within subject, chooses its modal activity, then clips each pooled
subject gradient at norm 1: seven effective units/site, one update/round at
default batch size 32. Central multinomial logistic regression uses all training
windows; the gap therefore includes this preprocessing difference.
The training-only audit found subject-modal counts `[6,0,0,1,5,9]` for classes
1–6: activities 2 and 3 have no pooled training labels. This was known before
scoring; settings were retained unchanged.

Optimization defaults are unchanged; `n_classes=6` supplies the output schema.
Three replicates, five rounds, epsilon 1/8/4 and delta `1e-6` are fixed.
`run_har561.R` trains all three central and nine federated models before opening
test tables once for final scoring. Seeds control sites and central initialization;
federated initialization and DP randomness retain release behavior. No scored cell
is rerun with changed settings.

On the provisioned pod, prepare the archives without opening test tables:

```sh
cd /workspace/cells
curl -fL 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip' \
  -o data_cache/har561_official.zip
python3 - <<'PY'
from pathlib import Path
from zipfile import ZipFile
with ZipFile('data_cache/har561_official.zip') as archive:
    member, = [name for name in archive.namelist() if name.endswith('UCI HAR Dataset.zip')]
    Path('data_cache/har561_inner.zip').write_bytes(archive.read(member))
PY
bash dsFlowerClient/tools/campaign/multiclass/run_har561.sh /workspace/cells
cd dsFlowerClient
python3 tools/campaign/multiclass/summarize_har561.py inst/extdata/campaign/multiclass
```

The runner verifies both hashes, runs synthetic `check_har561.R`, and refuses
existing run markers. `summarize_har561.py` validates saved metrics without
rescoring and adds HAR alongside CTG. HAR results are reported in the
[evidence README](../../../inst/extdata/campaign/multiclass/README.md).

## Pre-declared CDC GenHlth alternative

Authorization `FLOWER_CELLS_MULTICLASS_2026-09-22`. The new and final alternative
is `pytorch_multiclass` on **cdc45k**, the exact fixed-seed 45,000-respondent
cohort used by the logistic n-scaling arm. Its original diabetes-stratified
membership is retained; the new 80/20 split and three sites are stratified by
`GenHlth`. The five ordered health responses (excellent to poor) are treated
as nominal classes. The other 20 indicators are inputs; ID and the diabetes
label are excluded.

[UCI CDC Diabetes Health Indicators, id 891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators),
DOI [10.24432/C53919](https://doi.org/10.24432/C53919), is cited in the thesis as
`uci_cdc_diabetes_health_indicators`. UCI documents one row per respondent.
Raw CSV SHA-256: `9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964`.
The legacy logistic prepared-cohort hash must match
`5f521899386021fd6670d166f4f5ef9a4dcaa80e11d8df10942e60bd18724d3b`.

The frozen CDC protocol declares three sites, five rounds, three seeds
20260820–20260822, epsilon 1/4/8 in that order, delta 1e-6, row privacy and
clipping norm 1. Registry optimization defaults remain unchanged, with only
`n_classes=5` supplying the output schema. Public bounds are binary [0,1],
BMI [0,100], MentHlth/PhysHlth [0,30], Age [1,13], Education [1,6], Income [1,8];
BMI is a declared clipping domain, not an empirical bound. These are copied
from the existing CDC trees campaign, never estimated from held-out rows.

Central multinomial logistic regression uses the same training rows and bounded
transform. Trivial probabilities use training class proportions and predict
their majority. Report macro one-vs-rest AUC, accuracy, log-loss and paired
federated-minus-central macro-AUC gap, with mean and sample SD. Epsilon-8
macro-AUC above 0.5 and accuracy above majority are **annotations, not pass/fail**.

Partition preparation only assigns and seals held-out rows; no test values or
statistics are inspected. All three central and nine federated models train
before the sealed test partitions are opened once for final scoring. Each fit
is scored once. No scored cell is rerun or changed, and no further alternative
is authorized. CTG remains the ranking-retained, argmax-calibration-lost
measurement at 1,701 training rows. HAR remains the subject-privacy utility
boundary with seven units/site, outside the useful regime on this epsilon grid;
this is the utility law, not a defect. Their scored evidence stays unchanged.

The full frozen declaration is [cdcgenhlth_protocol.json](cdcgenhlth_protocol.json).
`prepare_cdcgenhlth.R` copies the original logistic cohort sampling and checks
its prepared-CSV checksum before changing the target. `cdcgenhlth.R` copies
stratified splitting and adds five-class scoring and the public-bound transform.
`cdcgenhlth_campaign.R` copies the federation lifecycle with row privacy and five
output levels. These files do not modify the shared loader or released code.

On the existing pod, run in the foreground:

```sh
cd /workspace/cells
curl -fL https://archive.ics.uci.edu/static/public/891/data.csv \
  -o data_cache/cdc_diabetes_health_indicators.csv
printf '%s  %s\n' \
  9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964 \
  cdc_diabetes_health_indicators.csv > data_cache/cdcgenhlth_CHECKSUMS.sha256
(cd data_cache && sha256sum -c cdcgenhlth_CHECKSUMS.sha256)
bash dsFlowerClient/tools/campaign/multiclass/run_cdcgenhlth.sh /workspace/cells
cd dsFlowerClient
python3 tools/campaign/multiclass/summarize_cdcgenhlth.py inst/extdata/campaign/multiclass
```

The wrapper runs synthetic metric/partition/scaling checks, preparation, and
then the fixed training and scoring matrix. Existing run markers or artifacts
are refused. The final summarizer checks saved evidence without fitting or
scoring. `cdcgenhlth_*.json` records results and environment; `summary.json`
retains CTG and HAR and adds CDC. Epsilon comparisons are per node training,
not a composed nine-fit privacy claim. The gap includes optimization,
federation and privacy costs; it does not isolate a causal privacy penalty.


The scored CDC result is macro-AUC 0.7518 / 0.7517 / 0.7512 at epsilon
1 / 4 / 8, versus central 0.7837. Accuracy is 0.4326 / 0.4330 / 0.4341
versus the 0.3500 majority baseline. All three epsilon-8 replicates have
macro-AUC above 0.5 and accuracy above majority; these are annotations only.
Macro-AUC is nearly flat across the budget grid. The paired gap includes the
fixed five-round optimization, federation and privacy costs. All nine fits
were scored once after all training completed, with no changed settings or
further alternative. See the [full results and sample SD](../../../inst/extdata/campaign/multiclass/README.md).
