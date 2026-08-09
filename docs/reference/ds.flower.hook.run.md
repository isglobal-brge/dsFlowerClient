# Request a HookApp run through the node-side egress policy

The package exposes `initial_arrays()` and `local_update()` but is never
trusted by the node. When the custodian-enabled sandbox and timing gates
are available, the node clips the complete update and applies its
configured Gaussian output mechanism (optionally over disjoint blocks).
Publicly absent gates cause a client preflight error before upload. If a
gate disappears after that preflight, the node refuses to open private
data and marks the unchanged update unavailable; the coordinator does
not report it as a trained model. Hash verification and static scanning
do not provide `nn.Module`/per-sample DP-SGD granularity for arbitrary
code. The timing envelope is defense in depth, not a formal
constant-time guarantee.

## Usage

``` r
ds.flower.hook.run(
  conns,
  user_app_dir,
  target,
  features,
  symbol = "D",
  num_rounds = 1L,
  task = c("classification", "regression", "count"),
  verbose = TRUE,
  target_levels = NULL,
  target_bounds = NULL,
  allow_insecure_http = getOption("dsflower.dsi_allow_insecure_http", character()),
  app_params = list()
)
```

## Arguments

- conns:

  DSI connections object.

- user_app_dir:

  Character; path to the researcher's training package dir (a folder
  with \_\_init\_\_.py exposing initial_arrays + local_update).

- target:

  Character; target column name.

- features:

  Character vector; feature column names.

- symbol:

  Character; server-side data handle symbol (default "D").

- num_rounds:

  Integer; federated rounds (default 1).

- task:

  Character; supervised task type used by node disclosure checks.

- verbose:

  Logical.

- target_levels:

  Optional ordered public label vocabulary for classification HookApps.
  Missing or unknown values map to public code zero.

- target_bounds:

  Required public `list(lower=..., upper=...)` for regression/count
  HookApps.

- allow_insecure_http:

  Character vector of exact connection names allowed to use plaintext
  HTTP. Empty by default. This exception does not provide transport
  security; use it only behind an independently trusted network.

- app_params:

  Named JSON-like list of public HookApp hyperparameters. Values may
  contain bounded nested objects/arrays and finite scalars. Privacy,
  runtime, dependency, credential and filesystem-path fields are
  reserved. The canonical value is hash-pinned and supplied to both app
  hooks.

## Value

A `dsflower_run` object. A public readiness failure is rejected before
upload. If no private release is available, the completed object has
`available = FALSE` and contains no fallback model artifact.
