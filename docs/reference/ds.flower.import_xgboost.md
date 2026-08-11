# Import a Bounded External XGBoost JSON Model

Converts one external XGBoost 3.4.0 JSON model into the same canonical,
data-only ensemble and prediction sidecar used by dsFlower's trusted
local predictor and private validation runner. The importer never loads
XGBoost, pickle, joblib, callbacks, or executable model payloads.

## Usage

``` r
ds.flower.import_xgboost(
  artifact,
  model,
  features,
  feature_bounds,
  feature_cuts,
  target,
  target_levels = NULL,
  target_bounds = NULL,
  output_dir
)
```

## Arguments

- artifact:

  Path to one bounded regular XGBoost JSON model file.

- model:

  A valid
  [`ds.flower.model.xgboost()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.xgboost.md)
  public profile.

- features:

  Ordered public feature names.

- feature_bounds:

  Public `list(lower=..., upper=...)` vectors.

- feature_cuts:

  One ordered public cut vector per feature.

- target:

  Public target name used by the model schema.

- target_levels:

  Two ordered public levels for a binary model.

- target_bounds:

  Public `list(lower=..., upper=...)` for regression.

- output_dir:

  A new destination directory. It is published only after the complete
  bundle has passed the trusted parser and prediction probe.

## Value

The absolute path to the imported model directory.

## Details

The supplied `model` is a public admission profile, not a training
request: its tree count, depth, learning rate and leaf bound must admit
the external artifact. Feature cuts and bounds must exactly describe
every accepted split. The resulting bundle is emitted as
`external-unverified`; dsFlower does not claim that the imported model
was trained with differential privacy. A later
[`ds.flower.validate()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.validate.md)
call still releases its pooled metrics through the normal node-DP
mechanism. The external origin is bound into the hashed ensemble
contract, but this is local format provenance rather than a
cryptographic signature against the model-directory owner. External leaf
values can contain ordinary non-private statistics, so the bundle
records that unnoised statistics may be present.
