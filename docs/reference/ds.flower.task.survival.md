# Create a survival task specification

For time-to-event models (e.g. Cox PH). Requires target_column to be a
vector of two column names: `c("time_col", "event_col")`.

## Usage

``` r
ds.flower.task.survival()
```

## Value

A `dsflower_task` S3 object with type = "survival".
