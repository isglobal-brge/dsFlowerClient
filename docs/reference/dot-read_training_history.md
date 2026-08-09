# Read training history from results directory

Read training history from results directory

## Usage

``` r
.read_training_history(results_dir)
```

## Arguments

- results_dir:

  Character; path to the results directory.

## Value

A data.frame with columns round, n_failures and (when available)
n_examples, or NULL. Loss is not released by the nodes (disclosure
backstop).
