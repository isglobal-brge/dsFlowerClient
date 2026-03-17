# Auto-resolve SuperLink address for each Opal node

Resolution strategy per node:

1.  If the node is in a container, try `host.docker.internal`.

2.  If bare-metal, detect the researcher's routable IP via UDP socket.

3.  Verify connectivity from the Opal to the candidate address.

4.  If verification fails, error with guidance.

## Usage

``` r
.auto_resolve_superlink(conns, symbol)
```

## Arguments

- conns:

  DSI connections object.

- symbol:

  Character; handle symbol name.

## Value

A single address string (if all nodes need the same) or a named list of
per-node addresses.
