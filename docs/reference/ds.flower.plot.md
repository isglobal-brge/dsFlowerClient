# Plot training curves

Plot training curves

## Usage

``` r
ds.flower.plot(result, metric = "loss", per_server = FALSE, title = NULL)
```

## Arguments

- result:

  A `dsflower_result` object or comparison data.frame.

- metric:

  Character; which metric to plot (default "loss").

- per_server:

  Logical; show per-server curves (default FALSE).

- title:

  Character; plot title (default auto-generated).

## Value

A plotly object.
