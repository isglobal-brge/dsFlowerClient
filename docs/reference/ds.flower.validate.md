# Differentially-private federated model validation

Evaluates a released tabular declarative neural model, saved first-party
ResNet-18 or DenseNet-121 vision classifier, sanitized native-tree
ensemble, or explicitly external-unverified imported XGBoost bundle on
the dataset assigned for this call inside each data node. The native
request, ensemble and prediction-profile sidecar are pinned into the
ephemeral execution contract and every node re-sanitizes the ensemble
before opening its data. Vision validation accepts only
binary/multiclass releases made by `pytorch_resnet18` or
`pytorch_densenet121`, including their volumetric variants. Its
canonical backbone, image size, frozen feature dimension, class
vocabulary, declarative head and bounded `model.pt` digest are pinned
before DSI. The `vision-extractor-profile` is a versioned semantic ABI,
not a cryptographic signature of extractor state; releases without it
fail closed, and semantic implementation/dependency changes require a
profile bump. Image paths and pixels remain node-private. It does not
add local image prediction, holdout, or cross-validation support.
Reusing the training dataset is resubstitution validation; assigning an
independent dataset is external validation. Each protected row/patient
contributes one bounded sufficient-statistic vector, the node releases
it once through the server-owned Gaussian mechanism, and only pooled
post-processed metrics are returned. Exact predictions, labels, counts
and per-node metrics never leave the node. All nodes must declare the
same row- or patient-level estimand. Privacy is guaranteed per node; if
one person occurs in multiple nodes, those node releases compose for
that person and deployments should account for the overlap or ensure
cohorts are disjoint. If not every expected node returns the fixed
private release, the result has `available=FALSE` and no metrics rather
than exact, per-node or zero-filled substitutes. This is an operational
availability result, not a historical query denial.

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

  A successful `dsflower_run`, saved model directory, or path returned
  by
  [`ds.flower.import_xgboost()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.import_xgboost.md).

- target:

  Target column name(s); multilabel validation requires one per saved
  label. Vision validation requires the saved public class vocabulary.

- data:

  Optional server-side data symbol.

- resource:

  Optional Opal resource name.

- symbol:

  Optional server-side handle symbol.

- bins:

  Public number of probability bins in `[4,512]`.

- torch_backend:

  Node torch backend selection for neural artifacts, including vision.
  Native-tree validation does not provision Torch.

- verbose:

  Show Flower output.

- silent:

  Suppress progress messages.

- allow_insecure_http:

  Exact connection names allowed to use HTTP.

## Value

A `dsflower_validation`. Its `available` field is false and `metrics` is
null when the complete pooled release was not available; no exact or
zero-filled substitute metrics are returned. The
`model_training_privacy` field keeps model provenance separate from the
node-DP metric release.
