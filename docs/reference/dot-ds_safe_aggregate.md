# Resilient datashield.aggregate that tolerates per-server failures

Resilient datashield.aggregate that tolerates per-server failures

## Usage

``` r
.ds_safe_aggregate(conns, expr)
```

## Arguments

- conns:

  DSI connections object.

- expr:

  The call expression to evaluate.

## Value

Named list of results (only successful servers).
