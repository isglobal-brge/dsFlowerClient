# Ensure ML framework dependencies are installed in the client venv

Checks if the framework's Python packages are available. Installs them
on-demand if missing. Called automatically before training and
prediction.

## Usage

``` r
.ensure_client_framework(framework)
```

## Arguments

- framework:

  Character; "sklearn", "pytorch", "pytorch_vision", or "xgboost".

## Value

Invisible TRUE.
