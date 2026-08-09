# Create a model spec by name

Resolves a registered model name and validates its typed parameter
contract. Existing `dsflower_model` objects are canonicalised and
revalidated against the current registry.

## Usage

``` r
ds.flower.model(name = "pytorch_logreg", ...)
```

## Arguments

- name:

  Character model name or a `dsflower_model` object.

- ...:

  Arguments passed to the selected concrete model constructor.

## Value

A `dsflower_model` object.
