# Wait for all SuperNodes to be running

Polls each server until `supernode_running = TRUE` or timeout.

## Usage

``` r
.wait_supernodes_ready(conns, symbol, timeout = 30)
```

## Arguments

- conns:

  DSI connections object.

- symbol:

  Character; handle symbol name.

- timeout:

  Numeric; seconds to wait.

## Value

Named list of per-node status results.
