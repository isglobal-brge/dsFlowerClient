# List registered dsFlower models

List registered dsFlower models

## Usage

``` r
ds.flower.list_models()
```

## Value

A data.frame with one row per registered model. Column `available`
reports whether the client-side constructor is implemented; operational
native-runtime availability is probed per node at submission. Released
native-tree ensembles support private external or resubstitution
validation.
