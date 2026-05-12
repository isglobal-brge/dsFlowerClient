# Auto-resolve SuperLink address for each Opal node

For each node, builds a prioritized list of candidate addresses and
tests connectivity until one succeeds. Candidates include
host.docker.internal (for containerized nodes), the OS-routed IP,
VPN/tunnel IPs, and LAN IPs.

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
