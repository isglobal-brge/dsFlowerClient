# Prepare a training run on all servers

Calls `flowerPrepareRunDS` on each server to stage data.

## Usage

``` r
ds.flower.nodes.prepare(
  conns,
  symbol = "flower",
  target_column,
  feature_columns = NULL,
  run_config = list(),
  privacy = NULL,
  template_name = NULL,
  label_set = NULL
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

- privacy:

  Optional `dsflower_privacy` object. When supplied, privacy mode and
  parameters are injected into `run_config`.

- template_name:

  Optional Flower template name used for server-side staging.

- label_set:

  Optional imaging label-set name for imaging-backed runs.

## Value

A `dsflower_result` with per-site status.
