# Describe the connected dataset

Returns a compact summary of public protocol capabilities and node-owned
structural imaging declarations. It does not report cohort-derived
counts or schema discovered from private data.

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
