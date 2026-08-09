# List available segmentation masks

Queries the server for mask assets declared in the node-owned imaging
manifest. This structural view intentionally excludes completion counts,
storage locations, and data-derived catalog state.

## Usage

``` r
ds.flower.masks(flower)
```

## Arguments

- flower:

  A `dsflower_connection`, or NULL for last connection.

## Value

A data.frame with public mask aliases, providers, and the constant
status `"declared"`, or empty if none.
