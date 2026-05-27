# Create a Linear SVM model spec

Convenience constructor for a linear Support Vector Machine. Internally
uses the `sklearn_sgd` template with `loss = "hinge"`, i.e. a linear
hinge-loss classifier trained incrementally through the federated SGD
template.

## Usage

``` r
ds.flower.model.sklearn_svm(
  alpha = 1e-04,
  lr_schedule = "optimal",
  eta0 = 0.01,
  max_iter = 1000L
)
```

## Arguments

- alpha:

  Numeric; regularization constant (analogous to 1/C in SVC). Smaller
  values = less regularization.

- lr_schedule:

  Character; learning rate schedule.

- eta0:

  Numeric; initial learning rate for constant/invscaling/adaptive
  schedules.

- max_iter:

  Integer; maximum local sklearn iterations.

## Value

A `dsflower_model` S3 object using the sklearn_sgd template.

## Details

Only linear SVMs are supported in federated learning because kernel SVMs
require the full pairwise kernel matrix (all data in one place).
