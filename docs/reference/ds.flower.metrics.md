# Get training metrics from all servers

Get training metrics from all servers

## Usage

``` r
ds.flower.metrics(symbol = "flower", since_round = 0L, pool = TRUE, conns)
```

## Arguments

- symbol:

  Character; Flower session symbol (default "flower").

- since_round:

  Integer; return only metrics from this round onward.

- pool:

  Logical; if TRUE, compute pooled metrics across servers.

- conns:

  DSI connections (required).

## Value

A `dsflower_result` object with training metrics.
