# Save the global model from a training run

Saves the federated model weights to a file. Supported formats: `.rds`
(R native), `.json` (portable).

## Usage

``` r
ds.flower.save_model(run, path)
```

## Arguments

- run:

  A `dsflower_run` object.

- path:

  Character; file path to save to.

## Value

Invisible path.
