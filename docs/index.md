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
python -m pip install "flwr>=1.31.0"
```

Each Opal/Rock server must have a compatible `dsFlower` installation. Runner ABI
and recursive SHA-256 mismatches fail before a run is submitted.

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
  feature_bounds = list(lower = c(18, 0), upper = c(100, 250))
)

DSI::datashield.logout(conns)
```

`ds.flower.train()` is an alias of `ds.flower.fit()`. `features` can be omitted
for an assigned tabular `symbol`, in which case only column names are queried and
the target is excluded. Feature values, exact feature statistics, node logs and
node metrics are never requested by this workflow.

## Privacy is server-authoritative

Each node uses a persistent lifetime accountant. For new run `n`, it allocates a
geometrically decreasing fraction of its administrator-set total budget:

```text
w_n       = (1 - rho) rho^(n - 1)
epsilon_n = epsilon_total w_n
delta_n   = delta_total w_n
```

This bounds every finite prefix and the infinite transcript by the node's
`(epsilon_total, delta_total)`. Exact Flower-message retries never trigger a
second private release: the cached response is reused when available; otherwise
the incoming public model is returned unchanged. Distinct releases receive
distinct, secret-keyed random streams.

There is no lifetime query-count rejection. Finite total privacy with infinitely
many equally informative answers is mathematically impossible, however, so the
allocations tend to zero. When a per-message allocation becomes numerically too
small, the node safely returns the incoming public model unchanged. The client
cannot introduce a positive epsilon floor or reset the accountant by renaming or
subsetting a dataset.

Noise is deterministic only within one release identity. The node derives
domain-separated ChaCha20 streams from a dedicated 256-bit secret using
HMAC-SHA256. This prevents averaging exact retries while avoiding the unsafe
reuse of one fixed noise vector across related queries. The secret is not a
client seed, R RNG state or `datashield.seed`.

The server's `flowerPrivacyBudgetDS()` method reports the public accountant
policy. Allocation count/status is returned only if the custodian enables
`dsflower.expose_privacy_status`; there is intentionally no analyst privacy
configuration API.

## Supported computation contracts

### Declarative models (recommended)

The client emits data-only specifications; the node-installed runner constructs
and trains the model:

- neural and vision specifications use Opacus DP-SGD with per-example or
  server-selected per-patient clipping and noise;
- tree specifications use DP-GBDT with data-independent structure, bounded
  gradients/Hessians and noisy leaf histograms.

This is the path that reaches `nn.Module`-level granularity because the trusted
runner owns the training loop and sees per-sample gradients. The client never
sends an analyst-controlled training loop for declarative models.

Available registered model names can be inspected with `ds.flower.models()`.
Hyperparameters are supplied through `model_params`:

```r
fit <- ds.flower.fit(
  conns, symbol = "D", target = "y", features = c("x1", "x2"),
  model = "pytorch_mlp",
  model_params = list(hidden_layers = c(64, 32)), rounds = 10L,
  feature_bounds = list(lower = c(0, 0), upper = c(1, 100))
)
```

### HookApps (legacy name: Tier2)

`ds.flower.hook.run()` accepts a Python package exposing only:

```text
initial_arrays(config, input_dim) -> numeric arrays
local_update(global_arrays, X, y, config) -> numeric arrays
```

A HookApp is not a general trusted Flower App and cannot generically receive
DP-SGD-level protection. The node runs it only when the custodian has enabled
HookApps and attested the required Bubblewrap filesystem/network sandbox and
constant-time envelope. The result is numerically validated, clipped as one
complete update and passed through a conservatively RDP-calibrated Gaussian
mechanism; an optional sample-and-aggregate mode uses a fixed, administrator-
pinned number of disjoint data-independent blocks.

If any execution gate is absent, the untrusted package is not run and the node
returns the incoming public model unchanged. Archive scanning and hash pinning
are integrity defenses, not proofs that arbitrary code is private.
`ds.flower.tier2.run()` is a deprecated compatibility alias.

## Public feature bounds

Exact node-side `count`, `sum`, `sumsq`, means, variances and quantiles are not
released for preprocessing. If scale information is useful, pass public bounds:

```r
feature_bounds = list(
  lower = c(age = 18, biomarker = 0),
  upper = c(age = 100, biomarker = 250)
)
```

Bounds must be finite, satisfy `lower < upper` and follow the exact feature order.
Training clips each value to its interval and maps it affinely to `[-1, 1]`;
prediction reuses the stored bounds. These constants must come from public domain
knowledge or protocol design, not from a query to the protected node. Without
bounds, neural inputs remain unscaled and DP-GBDT uses a public `[0, 1]` threshold
prior, which can reduce utility when the real scale differs.

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
  feature_bounds = list(lower = c(18, 0), upper = c(100, 250))
)
```

The lower-level API does not weaken the node policy. Node-returned
`ds.flower.metrics()` and `ds.flower.log()` calls remain for protocol
compatibility but return empty results on hardened nodes. Local SuperLink output
and the intended DP global-model artifact are separate from node log/metric
egress.

See the
[`dsFlower` architecture specification](https://github.com/isglobal-brge/dsFlower/blob/main/ARCHITECTURE.md)
for the complete trust boundary, deployment requirements and residual limits.

## Authors

- **David Sarrat González** — david.sarrat@isglobal.org
- **Juan R González** — juanr.gonzalez@isglobal.org

[Barcelona Institute for Global Health (ISGlobal)](https://www.isglobal.org/)
