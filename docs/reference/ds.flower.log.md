# Get log output from all servers

Get log output from all servers

## Usage

``` r
ds.flower.log(symbol = "flower", last_n = 50L, conns)
```

## Arguments

- symbol:

  Character; Flower session symbol (default "flower").

- last_n:

  Integer; number of log lines to return per server.

- conns:

  DSI connections (required).

## Value

A `dsflower_result` object with log lines per server.
