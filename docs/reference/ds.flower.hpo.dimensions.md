# Define a bounded floating-point HPO dimension

Define a bounded floating-point HPO dimension

Define a bounded integer HPO dimension

Define a categorical HPO dimension

## Usage

``` r
ds.flower.hpo.float(lower, upper, log = FALSE, step = NULL)

ds.flower.hpo.integer(lower, upper, log = FALSE, step = 1L)

ds.flower.hpo.categorical(values)
```

## Arguments

- lower, upper:

  Integer endpoints with \`lower \< upper\`.

- log:

  Whether Optuna samples on a logarithmic scale. This requires a
  positive lower endpoint and \`step = 1\`.

- step:

  Positive integer step that divides the range exactly.

- values:

  A non-empty atomic vector of unique finite values. Character, logical,
  integer and numeric choices are supported.

## Value

A typed local HPO dimension for \`ds.flower.hpo()\`.

A typed local HPO dimension for \`ds.flower.hpo()\`.

A typed local HPO dimension for \`ds.flower.hpo()\`.
