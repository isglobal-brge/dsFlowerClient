# Representative multiclass utility cell

This directory extends the released campaign harness for `pytorch_multiclass`.
`campaign_lib.R` copies the DSLite-per-PSOCK-worker federation, local SuperLink
and SuperNode lifecycle, stratified split, bounds rule and cleanup from
`../campaign_lib.R`; the task-specific changes are three target levels,
multiclass scoring, and capture of each node's privacy configuration.
The shared harness, package code, registry and canonical runner are unchanged.

The complete design is frozen in `protocol.json`, before any test scoring.
All v0.5.0 registry defaults apply, including one local epoch (the older binary
pilot's two-epoch override is not inherited). There are three sites, five
rounds and three paired split seeds per epsilon, in execution order 1, 8, 4.
Central training is unregularized `nnet::multinom` with the same clip-and-affine
transform, fitted only on the pooled training split. The trivial prediction
uses training class proportions and therefore predicts its majority class.
No tuning or inner validation is performed. Epsilon 8 is assessed against
strictly better-than-majority mean accuracy and macro-AUC above 0.5; each
replicate is also disclosed. No fallback was specified for this track.

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

After the cells finish, validate and summarize their JSONs without rescoring:

```sh
python3 tools/campaign/multiclass/summarize.py inst/extdata/campaign/multiclass
```

`environment.py` captures installed dependency versions and provisioning
start/end times. During this run the client CPU torch runtime was provisioned
before scoring with `uv pip install --torch-backend cpu` and its remaining
prediction dependencies checked with the unchanged package helper
`dsFlowerClient:::.ensure_client_framework("pytorch")`.

The measured epsilon-8 accuracy diagnostic failed in all three replicates;
see the [committed results](../../../inst/extdata/campaign/multiclass/README.md).
All nine planned replicates completed with unchanged settings and no reruns.
