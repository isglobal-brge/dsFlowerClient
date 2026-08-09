# Close the federation link

Stops each node's tunnel forwarder, closes the relay sockets and stops
the local SuperLink. Reverses
[`ds.flower.link.up()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.link.up.md).

## Usage

``` r
ds.flower.link.down(conns, symbol = "flower")
```

## Arguments

- conns:

  DSI connections object.

- symbol:

  Character; handle symbol.

## Value

Invisibly, TRUE.
