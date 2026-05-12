# Build a Flower App from a recipe

Fetches the template from the server (via DataSHIELD), writes the Python
files locally, and generates a `pyproject.toml` with run_config from the
recipe.

## Usage

``` r
.build_flower_app(recipe, conns, app_dir = NULL, results_dir = NULL)
```

## Arguments

- recipe:

  A `dsflower_recipe` object.

- conns:

  DSI connections object (used to fetch templates from server).

- app_dir:

  Character; directory to create the app in (default: tempdir).

- results_dir:

  Character; directory for the strategy to save weights/metrics.

## Value

Character; path to the created app directory.
