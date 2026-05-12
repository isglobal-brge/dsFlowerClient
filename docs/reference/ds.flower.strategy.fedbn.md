# Create a FedBN strategy spec

Federated Batch Normalization: keeps BatchNorm layers local (not
aggregated) to handle feature shift between sites. Essential for medical
imaging across different scanners/protocols.

## Usage

``` r
ds.flower.strategy.fedbn(fraction_fit = 1, fraction_evaluate = 1)
```

## Arguments

- fraction_fit:

  Numeric; fraction of clients used for training (0-1).

- fraction_evaluate:

  Numeric; fraction of clients used for evaluation (0-1).

## Value

A `dsflower_strategy` S3 object.

## Details

Built on FedAvg but the server excludes BatchNorm parameters from
aggregation. Each client retains its own BN statistics.
