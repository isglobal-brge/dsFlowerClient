# Differentially-private federated model validation

Evaluates a released tabular declarative neural model on the dataset
assigned for this call inside each data node. Vision artifacts fail
explicitly because this validator does not reconstruct image loaders or
backbones. Reusing the training dataset is resubstitution validation;
assigning an independent dataset is external validation. Each protected
row/patient contributes one bounded sufficient-statistic vector, the
node releases it once through the server-owned Gaussian mechanism, and
only pooled post-processed metrics are returned. Exact predictions,
labels, counts and per-node metrics never leave the node. All nodes must
declare the same row- or patient-level estimand. Privacy is guaranteed
per node; if one person occurs in multiple nodes, those node releases
compose for that person and deployments should account for the overlap
or ensure cohorts are disjoint. If not every expected node returns the
fixed private release, the result has `available=FALSE` and no metrics
rather than exact, per-node or zero-filled substitutes. This is an
operational availability result, not a historical query denial.

## Usage

``` r
ds.flower.validate(
  conns,
  model,
  target,
  data = NULL,
  resource = NULL,
  symbol = NULL,
  bins = 32L,
  torch_backend = "auto",
  verbose = FALSE,
  silent = FALSE,
  allow_insecure_http = getOption("dsflower.dsi_allow_insecure_http", character())
)
```

## Arguments

- conns:

  DSI connections.

- model:

  A successful `dsflower_run` or saved model directory.

- target:

  Target column name(s); multilabel validation requires one per saved
  label.

- data:

  Optional server-side data symbol.

- resource:

  Optional Opal resource name.

- symbol:

  Optional server-side handle symbol.

- bins:

  Public number of probability bins in `[4,512]`.

- torch_backend:

  Node torch backend selection.

- verbose:

  Show Flower output.

- silent:

  Suppress progress messages.

- allow_insecure_http:

  Exact connection names allowed to use HTTP.

## Value

A `dsflower_validation`. Its `available` field is false and `metrics` is
null when the complete pooled release was not available; no exact or
zero-filled substitute metrics are returned.
