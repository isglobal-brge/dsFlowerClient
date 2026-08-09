# Create a FedAvg strategy spec

Every node participates in every training round and private evaluation
is disabled. Node updates carry a constant aggregation weight, so no
exact cohort size is disclosed to the researcher-side SuperLink.

## Usage

``` r
ds.flower.strategy.fedavg()
```

## Value

A `dsflower_strategy` S3 object.
