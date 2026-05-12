# Describe the connected dataset

Returns a compact summary: modality, sample count (bucketed per
profile), available labels, masks, and feature assets.

## Usage

``` r
ds.flower.describe(flower)
```

## Arguments

- flower:

  A `dsflower_connection` from
  [`ds.flower.connect()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.connect.md).

## Value

A list with dataset summary fields, printed nicely.
