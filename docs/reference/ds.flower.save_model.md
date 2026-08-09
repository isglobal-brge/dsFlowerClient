# Save the global model from a training run

Saves the federated model metadata to a file, together with a sibling
`<filename>.assets` directory containing the native model and its public
reconstruction metadata. Move both entries together. Supported formats
are `.rds` (R native) and `.json`.

## Usage

``` r
ds.flower.save_model(run, path)
```

## Arguments

- run:

  A `dsflower_run` object.

- path:

  Character; unused `.rds` or `.json` file path.

## Value

Invisible path.
