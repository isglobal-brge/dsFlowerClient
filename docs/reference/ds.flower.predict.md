# Predict with a federated model

Uses the saved model in native format (joblib/pt/xgb) to generate
predictions via Python. The appropriate framework dependencies are
installed on-demand in the client venv if not already present.

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

  Character; `"response"` for predicted class (default), `"prob"` for
  probabilities.

## Value

A numeric vector of predictions.
