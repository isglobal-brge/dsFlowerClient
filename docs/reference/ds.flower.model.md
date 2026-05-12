# Create a model spec by name

Convenience wrapper around the concrete `ds.flower.model.*`
constructors. Existing `dsflower_model` objects are returned unchanged.

## Usage

``` r
ds.flower.model(name = "sklearn_logreg", ...)
```

## Arguments

- name:

  Character model name or a `dsflower_model` object.

- ...:

  Arguments passed to the selected concrete model constructor.

## Value

A `dsflower_model` object.
