# Prepare a training run on all servers

Calls `flowerPrepareRunDS` on each server to stage data.

## Usage

``` r
ds.flower.nodes.prepare(
  conns,
  symbol = "flower",
  target_column,
  feature_columns = NULL,
  run_config = list()
)
```

## Arguments

- conns:

  DSI connections object.

- symbol:

  Character; symbol name of the handle.

- target_column:

  Character; name of the target column.

- feature_columns:

  Character vector or NULL; feature column names.

- run_config:

  Named list; additional run configuration.

## Value

A `dsflower_result` with per-site status.
