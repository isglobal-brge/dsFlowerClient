# Create a privacy spec by name

Convenience wrapper around the concrete `ds.flower.privacy.*`
constructors. Existing `dsflower_privacy` objects are returned
unchanged.

## Usage

``` r
ds.flower.privacy(name = "clinical_default", ...)
```

## Arguments

- name:

  Character privacy profile name or a `dsflower_privacy` object.

- ...:

  Arguments passed to the selected concrete privacy constructor.

## Value

A `dsflower_privacy` object.
