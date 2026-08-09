# Retry one idempotent aggregate until every node returns an explicit ACK

The exact same call object is reused on every attempt. A named NULL
(DSI's per-node error representation) or a malformed outer mapping is
retryable; an explicit non-NULL but invalid ACK fails immediately.

## Usage

``` r
.dsi_retry_exact_aggregate(conns, expr, validate, operation, attempts = 3L)
```
