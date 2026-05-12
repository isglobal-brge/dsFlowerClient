# One-shot federated learning (simplest path)

Connects, prepares, trains, and cleans up in a single call. For the
simplest possible researcher experience.

## Usage

``` r
ds.flower.train(conns, data, recipe, ...)
```

## Arguments

- conns:

  DSI connections object.

- data:

  Character; data source (resource name or symbol).

- recipe:

  A `dsflower_recipe` object.

- ...:

  Additional arguments passed to
  [`ds.flower.run()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.run.md).

## Value

A `dsflower_run` object.
