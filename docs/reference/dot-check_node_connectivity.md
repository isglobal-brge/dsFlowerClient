# Check connectivity from a single Opal node to a candidate address

Check connectivity from a single Opal node to a candidate address

## Usage

``` r
.check_node_connectivity(conns, srv, address)
```

## Arguments

- conns:

  DSI connections object.

- srv:

  Character; server name.

- address:

  Character; "host:port" to test.

## Value

Named list with `reachable` and `error`.
