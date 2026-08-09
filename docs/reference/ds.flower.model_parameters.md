# Inspect the public parameter contract for a registered model

Inspect the public parameter contract for a registered model

## Usage

``` r
ds.flower.model_parameters(name)
```

## Arguments

- name:

  Character; registered model name or friendly alias accepted by
  [`ds.flower.model()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.md).

## Value

A data frame with one row per accepted canonical parameter and list
columns `default` and `choices`. Conditional optimizer/scheduler
parameters are listed even when they have no default for the selected
default backend.
