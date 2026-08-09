# Validate and order one DSI result per requested node

DSI 1.8 may represent a per-node aggregate error as a named NULL element
or drop that element from the returned list. Both forms, and every
malformed or misassociated outer result, fail closed here.

## Usage

``` r
.dsi_exact_node_results(result, conns)
```
