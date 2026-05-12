# API Quickstart

`dsFlowerClient` exposes two API levels. Use
[`ds.flower.fit()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.fit.md)
for the common case, and drop down to `connect -> recipe -> run` when a
workflow needs more control.

## Connect to DataSHIELD

``` r

library(dsFlowerClient)
library(DSI)
library(DSOpal)

builder <- DSI::newDSLoginBuilder()
builder$append(
  server = "site1",
  url = "https://opal1.example.org",
  user = "researcher",
  password = "...",
  table = "PROJECT.training_data",
  driver = "OpalDriver"
)
builder$append(
  server = "site2",
  url = "https://opal2.example.org",
  user = "researcher",
  password = "...",
  table = "PROJECT.training_data",
  driver = "OpalDriver"
)

conns <- DSI::datashield.login(
  logins = builder$build(),
  assign = TRUE,
  symbol = "D"
)
```

## Happy path

``` r

fit <- ds.flower.fit(
  conns,
  symbol = "D",
  target = "outcome",
  features = c("age", "sex", "chol", "thalach"),
  model = "sklearn_logreg",
  model_params = list(max_iter = 100L),
  strategy = "fedavg",
  privacy = "auto",
  rounds = 5L
)
```

`privacy = "auto"` inspects server capabilities. It uses
`clinical_default` when Secure Aggregation is available on every server,
and `trusted_internal` otherwise. Use an explicit privacy value when a
deployment requires a fixed policy.

## Advanced path

``` r

flower <- ds.flower.connect(conns, symbol = "D")

recipe <- ds.flower.recipe(
  model = ds.flower.model("mlp", hidden_layers = c(64, 32)),
  strategy = ds.flower.strategy("fedprox", proximal_mu = 0.1),
  privacy = ds.flower.privacy("clinical_default"),
  target = "outcome",
  features = c("age", "sex", "chol", "thalach"),
  num_rounds = 10L
)

fit <- ds.flower.run(flower, recipe)
ds.flower.disconnect(flower)
```

## Cleanup

``` r

DSI::datashield.logout(conns)
```
