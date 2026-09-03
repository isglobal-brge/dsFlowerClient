# Show the admitted imaging label

Returns only the single `metadata.label_col` admitted by dsImaging.

## Usage

``` r
ds.flower.labels(flower)
```

## Arguments

- flower:

  A `dsflower_connection` from
  [`ds.flower.connect()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.connect.md),
  or NULL to use the last connection.

## Value

Per-server list of data.frames describing the admitted label.
