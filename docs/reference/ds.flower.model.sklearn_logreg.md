# Create a scikit-learn Logistic Regression model spec

Create a scikit-learn Logistic Regression model spec

## Usage

``` r
ds.flower.model.sklearn_logreg(penalty = "l2", C = 1, max_iter = 100L)
```

## Arguments

- penalty:

  Character; regularization penalty ("l2", "l1", "none").

- C:

  Numeric; inverse regularization strength.

- max_iter:

  Integer; maximum iterations.

## Value

A `dsflower_model` S3 object.
