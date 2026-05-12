# Clean up training run on all servers

Calls `flowerCleanupRunDS` on each server.

## Usage

``` r
ds.flower.nodes.cleanup(conns, symbol = "flower")
```

## Arguments

- conns:

  DSI connections object.

- symbol:

  Character; symbol name of the handle.

## Value

A `dsflower_result` with cleanup confirmation.
