# Run federated learning (auto-managed)

Automatically handles SuperLink startup, SuperNode ensure, data
preparation, training, and cleanup. The researcher only needs a
connection and a recipe.

## Usage

``` r
ds.flower.run(flower, recipe, detached = FALSE, verbose = TRUE)
```

## Arguments

- flower:

  A `dsflower_connection` from
  [`ds.flower.connect()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.connect.md),
  or NULL to use the last connection.

- recipe:

  A `dsflower_recipe` object.

- detached:

  Logical; use detached SuperLink (default FALSE).

- verbose:

  Logical; print training output (default TRUE).

## Value

A `dsflower_run` object.

## Details

For advanced control (custom ports, persistent SuperLink, etc.), use the
low-level functions:
[`ds.flower.superlink.start()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.superlink.start.md),
[`ds.flower.nodes.ensure()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.nodes.ensure.md),
[`ds.flower.run.start()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.run.start.md).
