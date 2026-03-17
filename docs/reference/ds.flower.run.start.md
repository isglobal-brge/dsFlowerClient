# Start a Flower run

Builds a Flower App from the recipe, then invokes `flwr run` against the
running SuperLink. The SuperLink must have been started with
[`ds.flower.superlink.start()`](ds.flower.superlink.start.md)
beforehand.

## Usage

``` r
ds.flower.run.start(
  recipe,
  app_dir = NULL,
  run_config = list(),
  verbose = TRUE
)
```

## Arguments

- recipe:

  A `dsflower_recipe` object.

- app_dir:

  Character; path to a pre-built app directory (optional).

- run_config:

  Named list; additional run config overrides.

- verbose:

  Logical; print flwr output (default TRUE).

## Value

A list with run_id, app_dir, and output.
