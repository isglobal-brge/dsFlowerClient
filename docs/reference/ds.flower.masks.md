# List available segmentation masks

Queries the server for validated mask assets from dsImaging. Only shows
ACTIVE, valid, non-partial masks by default.

## Usage

``` r
ds.flower.masks(flower)
```

## Arguments

- flower:

  A `dsflower_connection`, or NULL for last connection.

## Value

A data.frame with mask assets, or empty if none.
