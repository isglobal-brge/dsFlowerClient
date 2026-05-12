# Create a scikit-learn SGD Classifier model spec

Create a scikit-learn SGD Classifier model spec

## Usage

``` r
ds.flower.model.sklearn_sgd(
  loss = "log_loss",
  alpha = 1e-04,
  lr_schedule = "optimal",
  eta0 = 0.01,
  max_iter = 1000L
)
```

## Arguments

- loss:

  Character; loss function ("log_loss", "hinge", "modified_huber").

- alpha:

  Numeric; regularization constant.

- lr_schedule:

  Character; learning rate schedule ("optimal", "constant",
  "invscaling").

- eta0:

  Numeric; initial learning rate for constant/invscaling/adaptive
  schedules.

- max_iter:

  Integer; maximum local sklearn iterations.

## Value

A `dsflower_model` S3 object.
