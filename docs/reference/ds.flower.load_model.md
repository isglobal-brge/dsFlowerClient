# Load a saved model

Reads model weights and metadata from a previously saved output
directory, `.rds` file, or `.json` file. Files created by
[`ds.flower.save_model()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.save_model.md)
must remain beside their sibling `<filename>.assets` directory.

## Usage

``` r
ds.flower.load_model(path)
```

## Arguments

- path:

  Character; path to a model directory, `.rds`, or `.json` file.

## Value

A list with model_id, weights, history, and metadata.
