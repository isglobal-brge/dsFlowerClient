# Predict with a federated model

Uses a saved declarative PyTorch state dictionary or a sanitized
native-tree ensemble, including an explicitly external-unverified
imported XGBoost bundle, to generate local predictions. For a saved
native dsFlower vision classifier, `newdata` is a character vector of
image or volume paths; the exact checkpoint and frozen extractor are
validated before any image is opened, and inference uses bounded
batches. Native trees run through the bundled standard-library-only
predictor: the artifact, canonical request and public sidecar are
revalidated before execution, and no upstream tree library, pickle or
executable model payload is loaded.

## Usage

``` r
ds.flower.predict(model, newdata, type = c("response", "prob"))
```

## Arguments

- model:

  A `dsflower_run` object, a saved model list (from
  `ds.flower.load_model`), or a path to a model directory.

- newdata:

  A data.frame or matrix with feature columns, or, for a saved native
  dsFlower vision classifier, a non-empty character vector of local
  image or volume paths.

- type:

  Character; `"response"` returns a predicted class for classification
  models and a continuous response for regression/count models. `"prob"`
  returns probabilities for classification models.

## Value

A response vector or probability matrix. Saved vision and native-tree
classifiers return response values from their ordered public target
levels.
