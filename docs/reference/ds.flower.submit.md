# Submit + run a federated DP job from a model spec (the "pack" API)

Codegens the submission, ships it in one FAB to the DataSHIELD
SuperNodes, runs it under the manifest-pinned enforced-DP track, and
tears it down.

## Usage

``` r
ds.flower.submit(
  conns,
  model,
  target,
  features = NULL,
  data = NULL,
  resource = NULL,
  symbol = NULL,
  num_rounds = 1L,
  model_params = list(),
  data_kind = "tabular",
  strategy = "fedavg",
  output_dir = NULL,
  output_name = NULL,
  torch_backend = "auto",
  verbose = FALSE,
  silent = FALSE,
  feature_bounds = NULL,
  target_levels = NULL,
  target_bounds = NULL,
  allow_insecure_http = getOption("dsflower.dsi_allow_insecure_http", character())
)
```

## Arguments

- conns:

  DSI connections.

- model:

  A model name or `dsflower_model` (registry-resolved).

- target:

  Character; one target column, or exactly `num_labels` distinct columns
  for a multilabel model.

- features:

  Character vector; feature columns.

- data:

  Optional character data source resolved during connection.

- resource:

  Optional Opal resource name.

- symbol:

  Character; server-side data handle symbol.

- num_rounds:

  Integer; federated rounds.

- model_params:

  Named list; params merged over the model's defaults. Neural learning
  rates must be finite and in `(0, 10]`.

- data_kind:

  "tabular" or "image".

- strategy:

  Character strategy name or a `dsflower_strategy` object.

- output_dir:

  Optional model output directory.

- output_name:

  Optional model output name.

- torch_backend:

  Character; node torch backend selection.

- verbose:

  Logical.

- silent:

  Logical; suppress progress feedback.

- feature_bounds:

  Optional public feature bounds as `list(lower=..., upper=...)` in
  feature order. These constants are supplied without querying node data
  and define a clipped affine transform.

- target_levels:

  Optional ordered public label vocabulary for classification.
  Non-numeric targets require it; node values are never used to infer
  label codes, and missing or unknown values map to public code zero.
  Multilabel applies one public two-level vocabulary to each target
  independently.

- target_bounds:

  Required public `list(lower=..., upper=...)` for regression/count
  targets. The node clips each target to these constants.

- allow_insecure_http:

  Character vector of exact connection names allowed to use plaintext
  HTTP. Empty by default. This exception does not provide transport
  security; use it only behind an independently trusted network.

## Value

A `dsflower_run`.
