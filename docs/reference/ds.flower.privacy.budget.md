# Query remaining privacy budget on all servers

Calls `flowerPrivacyBudgetDS` on each server to retrieve the remaining
(epsilon, delta) budget for the dataset.

## Usage

``` r
ds.flower.privacy.budget(conns, symbol = "flower")
```

## Arguments

- conns:

  DSI connections object.

- symbol:

  Character; handle symbol name (default "flower").

## Value

Named list with per-server budget information.
