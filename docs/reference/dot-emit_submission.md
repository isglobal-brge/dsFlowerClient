# Resolve + emit a submission artifact from a model spec.

Resolve + emit a submission artifact from a model spec.

## Usage

``` r
.emit_submission(model)
```

## Arguments

- model:

  A `dsflower_model` object, or a model name (resolved via the
  registry). Extra `params` are merged over the registered defaults.

## Value

A list with track="neural", spec, loss, and params.
