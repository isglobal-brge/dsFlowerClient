# Create a scikit-learn SGD Classifier model spec

Create a scikit-learn SGD Classifier model spec

## Usage

``` r
ds.flower.model.sklearn_sgd(
  loss = "log_loss",
  alpha = 1e-04,
  lr_schedule = "optimal"
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

## Value

A `dsflower_model` S3 object.
