# Read public feature bounds from a model directory's metadata.json

Returns list(lower, upper) only when both vectors are finite, aligned,
and strictly ordered. Models without public bounds return NULL.

## Usage

``` r
.read_meta_bounds(model_dir)
```
