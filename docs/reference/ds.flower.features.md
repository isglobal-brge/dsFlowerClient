# List available feature assets

Queries the server for feature-table assets declared in the node-owned
imaging manifest. Storage locations and data-derived catalog state are
not returned.

## Usage

``` r
ds.flower.features(flower)
```

## Arguments

- flower:

  A `dsflower_connection` from
  [`ds.flower.connect()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.connect.md).

## Value

A data.frame with feature asset info, or empty.
