# Create a strategy spec by name

Convenience wrapper around the concrete `ds.flower.strategy.*`
constructors. Existing `dsflower_strategy` objects are returned
unchanged.

## Usage

``` r
ds.flower.strategy(name = "fedavg", ...)
```

## Arguments

- name:

  Character strategy name or a `dsflower_strategy` object.

- ...:

  Arguments passed to the selected concrete strategy constructor.

## Value

A `dsflower_strategy` object.
