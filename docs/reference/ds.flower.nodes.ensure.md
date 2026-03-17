# Ensure SuperNodes are running on all servers

Calls `flowerEnsureSuperNodeDS` on each server. If `superlink_address`
is `NULL`, auto-detects the correct address per node by querying each
Opal's environment (Docker vs bare metal).

## Usage

``` r
ds.flower.nodes.ensure(conns, symbol = "flower", superlink_address = NULL)
```

## Arguments

- conns:

  DSI connections object.

- symbol:

  Character; symbol name of the handle.

- superlink_address:

  Character, named list, or NULL.

  - `NULL` (default): auto-detect per node.

  - Single string: broadcast to all nodes.

  - Named list: per-node addresses (names must match connection names).

## Value

A `dsflower_result` with per-site status.
