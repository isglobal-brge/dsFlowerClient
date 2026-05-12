# Load a saved model

Reads model weights and metadata from a previously saved output
directory or `.rds` file.

## Usage

``` r
ds.flower.load_model(path)
```

## Arguments

- path:

  Character; path to the model directory or `.rds` file.

## Value

A list with model_id, weights, history, and metadata.
