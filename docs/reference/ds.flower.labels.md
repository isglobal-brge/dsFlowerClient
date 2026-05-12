# List available label sets for an imaging dataset

Queries the server for label sets defined in the dataset's manifest.

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

Per-server list of data.frames with columns: name, type, columns,
description.
