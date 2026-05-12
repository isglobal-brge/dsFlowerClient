# List saved models

Scans the output directory for previously saved models and returns a
summary data.frame with metadata for each.

## Usage

``` r
ds.flower.models(base_dir = file.path(".", "dsflower_output"))
```

## Arguments

- base_dir:

  Character; base directory to scan. Defaults to `"./dsflower_output"`.

## Value

A data.frame with columns: model_id, model, template, strategy, privacy,
num_rounds, n_clients, created_at, status, path.
