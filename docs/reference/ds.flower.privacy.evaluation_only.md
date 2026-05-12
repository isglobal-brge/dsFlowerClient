# Apply evaluation_only modifier to a privacy spec

Forces `model_release = "blocked"` and `allow_per_node_metrics = FALSE`
on the server.

## Usage

``` r
ds.flower.privacy.evaluation_only(base_privacy)
```

## Arguments

- base_privacy:

  A `dsflower_privacy` S3 object.

## Value

A modified `dsflower_privacy` S3 object with evaluation_only = TRUE.
