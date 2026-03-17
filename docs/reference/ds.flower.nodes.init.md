# Initialize Flower handles on all servers

Assigns the resource and calls `flowerInitDS` on each server.

## Usage

``` r
ds.flower.nodes.init(conns, resource, symbol = "flower")
```

## Arguments

- conns:

  DSI connections object.

- resource:

  Character; name of the resource in the project.

- symbol:

  Character; symbol name for the handle (default "flower").

## Value

A `dsflower_result` with per-site init results.
