# dsFlowerClient

`dsFlowerClient` is the researcher-side R package for coordinating
[Flower](https://flower.ai/) federated learning through
[DataSHIELD](https://www.datashield.org/). It pairs with the node-side
[`dsFlower`](https://github.com/isglobal-brge/dsFlower) package installed by each
data custodian.

The client starts a Flower SuperLink, submits a declarative request, checks that
every node has the byte-identical canonical runner, and coordinates staging,
training and cleanup. It cannot choose epsilon, delta, clipping, the accountant
domain or whether arbitrary code is allowed. Those decisions belong to each
data node.

## Installation

```r
remotes::install_github("isglobal-brge/dsFlowerClient")
```

The workstation also needs the Flower CLI:

```sh
python -m pip install "flwr==1.31.0"
```

If `uv` is absent, package provisioning refuses mutable `latest`/`curl|sh`
bootstrap. Either install `uv` through the operating system, or set an audited
release tag and platform-archive digest in `DSFLOWER_UV_VERSION` and
`DSFLOWER_UV_SHA256`. For reproducible Python environments, set
`DSFLOWER_CLIENT_PYTHON_LOCK` to a complete requirements file containing hashes
for all transitive artifacts; installs then use `uv pip install
--require-hashes`. Set `DSFLOWER_CLIENT_REQUIRE_PYTHON_LOCK=true` to reject a
missing lock instead of falling back to the tested direct requirements. Also set
`DSFLOWER_PYTHON_VERSION` to an exact
`major.minor.patch`; the default `3.11` permits compatible patch updates. Without
an exact interpreter and lock, transitive dependencies can still vary and the
resolved environment is not fully reproducible.

Each Opal/Rock server must have a compatible `dsFlower` installation. Tunnel and
runner ABI mismatches fail before stream bytes or a run are submitted; upgrades
that change either ABI require coordinated server/client deployment.

## Quick start

```r
library(dsFlowerClient)
library(DSI)
library(DSOpal)

builder <- DSI::newDSLoginBuilder()
builder$append(server = "site1", url = "https://opal1.example.org",
               user = "researcher", password = "...",
               table = "PROJECT.training_data", driver = "OpalDriver")
builder$append(server = "site2", url = "https://opal2.example.org",
               user = "researcher", password = "...",
               table = "PROJECT.training_data", driver = "OpalDriver")
conns <- DSI::datashield.login(builder$build(), assign = TRUE, symbol = "D")

# Bounds must be public/domain-knowledge constants in the same order as features.
fit <- ds.flower.fit(
  conns,
  symbol = "D",
  target = "outcome",
  features = c("age", "biomarker"),
  model = "pytorch_logreg",
  feature_bounds = list(lower = c(18, 0), upper = c(100, 250)),
  target_levels = c("control", "case"),
  holdout = 0.2
)

# One pooled DP test release from the same all-or-nothing training job.
fit$holdout

DSI::datashield.logout(conns)
```

`features` can be omitted for an assigned tabular `symbol`, in which case only
column names are queried and the target is excluded. Feature values, exact
feature statistics, node logs and node metrics are never requested by this
workflow.

## Privacy is server-authoritative

Each node applies one administrator-pinned epsilon/delta contract to every
training. Its accountant composes that contract across the training's own
rounds. There is no historical database, quota or resource-specific balance;
one training never reduces or blocks the next. Distinct trainings compose in
the standard way when they are analysed together.

When atomic holdout is requested, that same per-training pair is the total job
budget: the node applies a fixed split between composed training and one pooled
test release. It does not create a persistent balance or affect later jobs.
Cross-validation follows the same per-job rule: 80 percent is divided across
the `K` real trainings and the remaining 20 percent is used once for the pooled
OOF release. Higher `K` can therefore reduce DP-SGD utility under the same
contract; the API defaults to three folds.

Guarantees are per node. If the same person appears at multiple observed nodes,
their epsilons and deltas compose across those nodes; only disjoint node
populations receive the parallel-composition bound. Cross-site overlap needs a
shared federation-level person accountant when one global guarantee is required.

The formal adjacency is bounded/replace-one with a fixed number of privacy
units: neighbouring datasets replace one row, or all records belonging to one
configured patient.
This is not an unbounded add/remove membership guarantee for a changing unit
count.

Noise and training randomness are derived from a canonical semantic identity
with HMAC-SHA256 under the node's dedicated 256-bit secret. Equivalent effective
trainings recompute the same streams without a query database; changes to data,
model, mechanism, bounds or round change the identity. The secret is not a
client seed, R RNG state or `datashield.seed`.

The clipping, sensitivity and accounting contracts implement the standard
`(epsilon, delta)` mechanisms under their mathematical model. The shipped
finite-precision sampler is ChaCha20-backed and computationally hardened, but is
not a formally verified discrete-Gaussian implementation; the production claim
is therefore computational/practical DP for the documented valid-input domain.

The server's `flowerPrivacyPolicyDS()` method reports only the public
per-training policy. There is no analyst privacy-configuration API or historical
status to expose.

The Flower Fleet API is carried inside the DataSHIELD channel by a
capability-bound DSI tunnel. Because the inner Flower connection is plaintext
on loopback, `ds.flower.link.up()` validates the outer transport before starting
any local service or node forwarder:

- DSOpal exposes both its URL and retained curl options, so it requires
  `https://` and fails if peer or hostname verification is disabled;
- a recognized DSMolgenisArmadillo connection is accepted automatically when
  it exposes a valid `https://` URL; its frontend or reverse proxy remains
  responsible for certificate and hostname validation;
- DSLite is accepted because it is in-process and has no network transport;
- unknown connectors, or connections whose endpoint cannot be identified, fail
  closed unless the operator has independently verified transport
  confidentiality, integrity, and peer authentication, and attests each exact
  connection name, for example
  `options(dsflower.dsi_tls_attested = c("site1", "site2"))`;
- plaintext or unverifiable loopback endpoints require the explicit
  development-only option
  `options(dsflower.dsi_allow_insecure_loopback = TRUE)`.

Attestation records an operational fact; it does not add encryption. Plaintext
HTTP is rejected by default, but exact sites can be opted in with
`options(dsflower.dsi_allow_insecure_http = c("site1"))` or the equivalent
`allow_insecure_http` argument to `ds.flower.fit()`, `ds.flower.submit()`,
`ds.flower.hook.run()`, or `ds.flower.link.up()`. That exception assumes an
independent trusted network layer; it does not make HTTP confidential or
tamper-resistant. It only relaxes the dsFlowerClient preflight: the DSI
connector must itself support that endpoint (in particular, current `opalr`
rejects remote plaintext Opal connections before dsFlowerClient is called).
Credentials are also exchanged when the DSI connection is created, before this
preflight can inspect it. This client-side gate prevents accidental downgrade;
production DataSHIELD frontends remain authoritative. Link startup is all-or-nothing:
if one node does not publish a live, ready loopback forwarder, the client tears
down every attempted site and aborts. Nodes negotiate bounded exchange chunks;
the relay keeps a bounded buffer per site so TCP backpressure replaces
unbounded R-memory growth.

DSI 1.8 exposes a failed per-node operation as a named `NULL`. The client never
interprets that value as delivery: every mutating path requires an explicit ACK
from the matching node. For uploads and tunnel traffic, an ACK binds the
generation, offset, length and content identity. If an ACK is lost, only the
byte-identical in-flight chunk is retried against the idempotent node store; new
bytes cannot change that chunk's geometry.

## Supported computation contracts

### Declarative models (recommended)

The client emits data-only specifications; the node-installed runner constructs
and trains the model:

- neural and vision specifications use Opacus DP-SGD with per-example or
  server-selected per-patient clipping and noise.

This is the path that reaches `nn.Module`-level granularity because the trusted
runner owns the training loop and sees per-sample gradients. The client never
sends an analyst-controlled training loop for declarative models.

Available registered training contracts can be inspected with
`ds.flower.list_models()`. `ds.flower.model_parameters("pytorch_mlp")` returns
the accepted names, types, aliases, effective defaults and finite choices for one
contract. Hyperparameters are supplied through `model_params`:

```r
fit <- ds.flower.fit(
  conns, symbol = "D", target = "y", features = c("x1", "x2"),
  model = "pytorch_mlp",
  model_params = list(
    hidden_layers = c(64, 32), optimizer = "adamw",
    beta1 = 0.9, beta2 = 0.999,
    scheduler = "cosine", scheduler_min_lr = 1e-4
  ),
  rounds = 10L,
  feature_bounds = list(lower = c(0, 0), upper = c(1, 100))
)
```

The first-party suite covers binary, multiclass, multilabel and ordinal
classification; linear SVM; MLP, CNN, residual, recurrent, temporal and
Transformer specifications; linear, ridge, lasso, elastic-net and robust Huber
regression; Poisson, negative-binomial and Gamma outcomes; frozen-backbone
medical-image classification; and bounded conditional-quantile regression.
Neural contracts expose
SGD, Adam, AdamW and RMSprop plus none/step/exponential/cosine learning-rate
schedules. Parameters that are unknown, incompatible with the selected
optimizer/loss, outside a resource cap, or not implemented fail before staging;
accepted first-party parameters are pinned in the manifest and actually consumed
by the trusted runner.

The image contracts currently cover private federated training of their frozen-
backbone heads. The packaged local predictor and private validation API are
tabular; they do not yet reconstruct image loaders/backbones for inference or
validation. This boundary is explicit rather than silently treating images as a
numeric table.

The first-party native-tree constructors are `xgboost`, `extra_trees`,
`random_forest`, `lightgbm` and `catboost`. They run only in the dedicated
trusted node runtime, require public bounds and complete public cuts, and emit
sanitized data-only ensembles that can be saved, reopened, predicted locally
with the bundled standard-library predictor, or validated privately. Random
Forest resolves `max_features = "auto"` from the public feature count when the
request is built. It is a disjoint-partition DP forest in which each effective
privacy unit belongs to one tree, not upstream bootstrap/bagging Random Forest;
small cohorts can therefore have fewer effective units per tree. LightGBM and
CatBoost are dsFlower-style safe numeric engines,
not the upstream binaries or their model formats. A node advertises an engine
only after a fresh executable end-to-end probe; this is operational readiness,
not a model catalogue used as a privacy permission list. ExtraTrees, Random
Forest and the two dsFlower-style boosters use the small runtime provisioned by
the server package. XGBoost remains unavailable on a clean install until the
custodian supplies its separately built and verified platform bundle.

Classification strings/factors require an ordered public `target_levels`;
numeric labels already coded in `[0, K-1]` remain compatible.
Regression and count models require public
`target_bounds = list(lower=..., upper=...)`, with `lower >= 0` for counts and
`lower > 0` for Gamma loss. Nodes never infer label vocabularies or ranges from
their cohorts. Missing, malformed or unknown classification values map to the
public code zero, so private values cannot create a success/error side channel.

Public feature/target bounds and unscaled numeric inputs are limited to magnitude
`1e6`; without bounds, inputs remain unscaled but are coerced and saturated to
that domain. Declarative intermediates and parameters use the same finite cap;
heads are saturated at `30` for logits/log-links and `1e6` for direct MSE
regression. Per-sample gradients are totalised before Opacus performs the
server-owned L2 clip. Neural learning rates must be in `(0, 10]`.

### HookApps

`ds.flower.hook.run()` accepts a Python package exposing only:

```text
initial_arrays(config, input_dim) -> numeric arrays
local_update(global_arrays, X, y, config) -> numeric arrays
```

The call accepts a bounded, JSON-like `app_params` object. Both functions receive
the same validated public configuration:

```text
config = {
  app_params,       # analyst parameters, hash-pinned by the node
  round_index,      # 0 for initialization; 1..num_rounds for updates
  num_rounds,
  task,
  num_classes
}
```

The framework guarantees exact, typed JSON transport and manifest pinning of
`app_params`; the HookApp author remains responsible for validating and using
the keys its own code advertises. Unlike first-party declarative contracts,
arbitrary uploaded code cannot be proven statically to consume a parameter.

Privacy, paths, dependencies, runtime profiles, secrets and round counters are
reserved and cannot be smuggled through `app_params`. Returned arrays have a
fixed count and shape for the complete run. The wrapper clips their concatenated
update and applies the Gaussian mechanism on every round; its calibration
composes the full round transcript under the per-training contract. This is the
standard numeric output-DP mechanism for the fixed released vector (subject to
the finite-precision caveat above), independent of whether the child uses NumPy
or PyTorch. It is not internal DP-SGD and can have materially lower utility for a
high-dimensional update.

A HookApp is not a general trusted Flower App and cannot generically receive
DP-SGD-level protection. The node runs it only when the custodian has enabled
HookApps and attested the required Bubblewrap filesystem/network sandbox and
minimum-duration timing envelope. The result is numerically validated, clipped as one
complete update and passed through a conservatively RDP-calibrated Gaussian
mechanism; an optional sample-and-aggregate mode uses a fixed, administrator-
pinned number of disjoint data-independent blocks.

That envelope is timing defense in depth, not a formal constant-time proof;
cleanup, availability and storage behavior remain outside the numeric DP claim.

The client checks the public Hook readiness flags before upload and fails with a
clear deployment error when an administrator gate is absent. If a public gate
disappears after that preflight, the node refuses to open private data and marks
the unchanged update unavailable; the coordinator refuses to report it as a
trained model. A child crash, timeout or malformed result instead
maps to a zero update and still passes through the same Gaussian mechanism and
timing envelope, so it cannot bypass numeric DP. Archive scanning and hash
pinning are integrity defenses, not proofs that arbitrary code is private. A
trusted-runtime failure after private execution begins is reported only as
`available = FALSE`; its cause and fallback are not exposed as a trained model.
Native libraries or analyst-provided `requirements` are not installed
dynamically; new binary runtime profiles must be curated, locked and enabled by
the node administrator.
HookApps use the same `target_levels`/`target_bounds` contract and must declare
`task = "classification"`, `"regression"` or `"count"`.

## Private model validation

`ds.flower.validate()` evaluates a saved declarative neural or sanitized
native-tree model inside the nodes and releases one fixed, Gaussian-noised
vector of bounded sufficient statistics per node. The current validator accepts
those tabular artifacts; vision artifacts fail explicitly. Exact labels,
predictions, counts and site metrics remain local; the researcher receives only
pooled post-processing:

```r
validation <- ds.flower.validate(
  conns,
  model = fit,
  symbol = "D_test",
  target = "outcome",
  bins = 64L
)
print(validation)
```

Binary validation provides threshold metrics, ROC/PR AUC, Brier score,
calibration and decision-curve summaries; multiclass/ordinal/multilabel use
bounded confusion and one-vs-rest histograms; bounded regression/count contracts
provide MAE, MSE/RMSE, R-squared and, for counts, Poisson deviance. All are
computed from the same private vector, so deriving several reported metrics does
not create additional releases.

`validation$available` is `FALSE` if the complete fixed-layout release was not
available from every expected node. In that case `validation$metrics` is `NULL`:
the API returns no exact, per-node or zero-filled substitute and does not create
a query-count lockout.

Assigning an independent dataset gives external validation; reusing training
data gives resubstitution validation. For tabular declarative neural training,
`ds.flower.fit(..., holdout = 0.2)` instead assigns complete node-owned privacy
units before training, trains only on the complement, and evaluates the final
aggregate on the held-out side in the same all-or-nothing job. The fraction is
canonical, has no analyst seed, and retries recreate the same secret-keyed split.
The returned `fit$holdout` contains only pooled DP metrics; no predictions,
unit assignments or site metrics leave the nodes. Other backends fail explicitly
instead of pretending to support this protocol.

For honest K-fold validation, use the dedicated metrics-only workflow:

```r
cv <- ds.flower.cross_validate(
  conns,
  symbol = "D",
  target = "outcome",
  features = c("age", "biomarker"),
  model = "pytorch_logreg",
  feature_bounds = list(lower = c(18, 0), upper = c(100, 250)),
  target_levels = c("control", "case"),
  folds = 3L
)
cv$metrics
```

This runs `K` clean-initialized federated trainings, excludes each held-out fold
before its first training read, and retains raw OOF sufficient statistics only
in namespaced Flower `Context.state` backed by the pinned in-memory node
runtime—never a file or database. If every fold succeeds, the nodes make one final DP
release and the client accepts only `cv.json`, pinned to the submitted
CV-job and resampling-contract hashes. No fold model, prediction, fold/site metric or history
is returned or saved. A failed or restarted job publishes nothing and recomputes
the whole deterministic job.

Metric selection is local post-processing of that one release. Inspect the
scoreable metrics for the task, including their optimization direction, with
`ds.flower.metrics(cv)`. Curves, calibration bins and confusion matrices remain
in `cv$metrics`, but are deliberately not accepted as scalar HPO objectives.

Local HPO can then use the pooled OOF result directly:

```r
metric <- "roc_auc"
hpo <- ds.flower.hpo(
  objective = function(params) {
    cv <- ds.flower.cross_validate(
      conns,
      symbol = "D",
      target = "outcome",
      features = c("age", "biomarker"),
      model = "pytorch_logreg",
      model_params = list(learning_rate = params$learning_rate),
      feature_bounds = list(lower = c(18, 0), upper = c(100, 250)),
      target_levels = c("control", "case"),
      folds = 3L
    )
    ds.flower.score(cv, metric)
  },
  space = list(
    learning_rate = ds.flower.hpo.float(1e-4, 0.1, log = TRUE)
  ),
  n_trials = 10L,
  direction = ds.flower.metric_direction(metric, task = "binary"),
  seed = 1L
)
hpo$best_params
```

The objective function and Optuna study stay local to the researcher. Each
trial above starts an independent cross-validation job with its own
server-enforced DP contract; HPO does not pool privacy budgets or relax node
policy across trials. Select the objective metric for the task rather than
using one universal score—for example, maximize accuracy or ROC AUC
for binary classification, maximize macro F1 for multilabel classification, or
minimize MAE for bounded regression/count outcomes.

## Public feature bounds

Exact node-side `count`, `sum`, `sumsq`, means, variances and quantiles are not
released for preprocessing. If scale information is useful, pass public bounds:

```r
feature_bounds = list(
  lower = c(age = 18, biomarker = 0),
  upper = c(age = 100, biomarker = 250)
)
```

Bounds must be finite, have magnitude at most `1e6`, satisfy `lower < upper` and
follow the exact feature order.
Training clips each value to its interval and maps it affinely to `[-1, 1]`;
prediction reuses the stored bounds. These constants must come from public domain
knowledge or protocol design, not from a query to the protected node. Without
bounds, neural inputs remain unscaled after coercion/saturation to `[-1e6, 1e6]`,
which can reduce utility when the real scale differs.

## Lower-level API

`ds.flower.fit()` is the preferred end-to-end API. Power users can call the
submission pipeline directly:

```r
result <- ds.flower.submit(
  conns,
  model = ds.flower.model("pytorch_mlp", hidden_layers = c(64, 32)),
  symbol = "D",
  target = "outcome",
  features = c("age", "biomarker"),
  num_rounds = 10L,
  strategy = "fedadam",
  feature_bounds = list(lower = c(18, 0), upper = c(100, 250)),
  target_levels = c("control", "case")
)
```

The lower-level API does not weaken the node policy. Nodes expose neither raw
training logs nor exact per-site metrics. Local SuperLink output and the intended
DP global-model artifact are separate from private node state.

See the
[`dsFlower` architecture specification](https://github.com/isglobal-brge/dsFlower/blob/main/ARCHITECTURE.md)
for the complete trust boundary, deployment requirements and residual limits.

## Authors

- **David Sarrat González** — david.sarrat@isglobal.org
- **Juan R González** — juanr.gonzalez@isglobal.org

[Barcelona Institute for Global Health (ISGlobal)](https://www.isglobal.org/)
