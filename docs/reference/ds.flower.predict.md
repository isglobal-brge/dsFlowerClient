# Predict with a federated model

Uses a saved declarative PyTorch state dictionary to generate tabular
predictions via Python. Vision artifacts are not accepted by this
tabular predictor. The appropriate framework dependencies are installed
on-demand in the client venv if not already present.

## Usage

``` r
ds.flower.predict(model, newdata, type = c("response", "prob"))
```

## Arguments

- model:

  A `dsflower_run` object, a saved model list (from
  `ds.flower.load_model`), or a path to a model directory.

- newdata:

  A data.frame or matrix with feature columns.

- type:

  Character; `"response"` returns a predicted class for classification
  models and a continuous response for regression/count models. `"prob"`
  returns probabilities for classification models.

## Value

A numeric vector, integer class vector, or probability matrix for
multiclass, ordinal, and multilabel models.
