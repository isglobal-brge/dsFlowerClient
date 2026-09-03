# Destroy Flower and session-owned imaging handles

Destroys each handle on each node and removes its DataSHIELD workspace
symbol only after that same node returns an explicit success
acknowledgement. Nodes where a requested symbol is already absent are
skipped, so calling the function again retries only targets retained by
a previous partial failure. When `imaging_symbol` is supplied, the
deterministic temporary resource used by
`ds.flower.nodes.init(resource=)` is also removed. This is the
documented retry path if both automatic resource-removal attempts
failed.

## Usage

``` r
ds.flower.nodes.destroy(conns, symbol = "flower", imaging_symbol = NULL)
```

## Arguments

- conns:

  DSI connections object.

- symbol:

  Character; symbol name of the Flower handle.

- imaging_symbol:

  Character or NULL; symbol of the imaging handle created by
  `ds.flower.nodes.init(resource=)`. Leave NULL when the Flower handle
  consumes caller-owned data, including a caller-owned dsImaging handle.

## Value

A `dsflower_result` with per-site destruction status.
