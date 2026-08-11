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

  Finite endpoints with \`lower \< upper\`. Float dimensions accept
  numeric endpoints; integer dimensions require integer endpoints.

- log:

  Whether Optuna samples on a logarithmic scale. Logarithmic bounds must
  be positive. Float dimensions cannot combine \`log\` with \`step\`;
  integer dimensions require \`step = 1\`.

- step:

  Discretization step that must be positive and divide the range
  exactly. Float dimensions accept a finite numeric step or \`NULL\`;
  integer dimensions require a positive integer step.

- values:

  A non-empty atomic vector of unique finite values. Character, logical,
  integer and numeric choices are supported.

## Value

A typed local HPO dimension for \`ds.flower.hpo()\`.

A typed local HPO dimension for \`ds.flower.hpo()\`.

A typed local HPO dimension for \`ds.flower.hpo()\`.
