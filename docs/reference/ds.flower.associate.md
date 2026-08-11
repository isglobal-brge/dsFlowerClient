# Differentially-private pooled binary association

Builds one bounded 3x3 exposure/outcome table at each node, applies one
server-owned sticky Gaussian release, and returns only the pooled table
and descriptive prevalence measures. Values other than the two ordered
public levels are retained in the `unknown` row or column. No exact or
per-node counts leave a data node.

## Usage

``` r
ds.flower.associate(
  conns,
  outcome,
  exposure,
  outcome_levels,
  exposure_levels,
  data = NULL,
  resource = NULL,
  symbol = NULL,
  verbose = FALSE,
  silent = FALSE,
  allow_insecure_http = getOption("dsflower.dsi_allow_insecure_http", character())
)
```

## Arguments

- conns:

  DSI connections.

- outcome:

  One outcome column name.

- exposure:

  One exposure column name.

- outcome_levels:

  Ordered public `c(reference, positive)` levels.

- exposure_levels:

  Ordered public `c(reference, positive)` levels.

- data:

  Optional server-side data source.

- resource:

  Optional Opal resource name.

- symbol:

  Optional assigned server-side symbol.

- verbose:

  Show Flower output.

- silent:

  Suppress progress messages.

- allow_insecure_http:

  Exact connection names allowed to use HTTP.

## Value

A `dsflower_association`. If a complete release is unavailable,
`available` is false and table/measures/noise are null.

## Details

This is a descriptive association, not a causal or adjusted effect.
Under patient privacy, each axis means "ever positive" across all rows
for that patient; exposure and outcome need not occur at the same visit.
