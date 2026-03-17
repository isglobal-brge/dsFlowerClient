# Build a Flower App from a recipe

Copies the appropriate template from `inst/flower_templates/` and
generates a `pyproject.toml` with run_config from the recipe.

## Usage

``` r
.build_flower_app(recipe, app_dir = NULL)
```

## Arguments

- recipe:

  A `dsflower_recipe` object.

- app_dir:

  Character; directory to create the app in (default: tempdir).

## Value

Character; path to the created app directory.
