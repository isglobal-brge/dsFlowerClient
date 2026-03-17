# dsFlowerClient

> Client-side DataSHIELD package for orchestrating **Flower federated
> learning** experiments across multiple Opal/Rock servers.

## What it does

dsFlowerClient lets you train machine learning models across hospitals
without moving patient data. It bridges
[DataSHIELD](https://www.datashield.org/) (privacy-preserving
infrastructure) with [Flower](https://flower.ai/) (federated learning
framework).

     YOUR MACHINE                              HOSPITAL SERVERS
     (researcher)                              (Opal/Rock)
     ┌──────────────────────┐                  ┌──────────────────────┐
     │   dsFlowerClient     │   DataSHIELD     │     dsFlower         │
     │                      │ ───────────────> │                      │
     │   Orchestrates the   │   (HTTPS)        │   Trains locally,    │
     │   experiment         │ <─────────────── │   enforces controls  │
     └──────────────────────┘                  └──────────────────────┘

## Key features

- **Composable recipes** — mix and match tasks, models, strategies, and
  privacy settings
- **Multiple models** — Logistic Regression, SGD, Ridge, PyTorch MLP
- **Aggregation strategies** — FedAvg, FedProx
- **Differential privacy** — optional gradient clipping and noise
- **Auto-TLS** — ephemeral certificates for encrypted gRPC channels
- **Server-side controls** — disclosure thresholds, round limits, spawn
  permissions

## Quick start

``` r

library(dsFlowerClient)
library(DSI)
library(DSOpal)

# 1. Connect to Opal servers
builder <- DSI::newDSLoginBuilder()
builder$append(server = "site_a", url = "https://opal1.example.org",
               user = "researcher", password = "secret",
               driver = "OpalDriver")
builder$append(server = "site_b", url = "https://opal2.example.org",
               user = "researcher", password = "secret",
               driver = "OpalDriver")
conns <- DSI::datashield.login(logins = builder$build(), assign = FALSE)

# 2. Initialize and prepare
ds.flower.nodes.init(conns, resource = "project.flower_node")
ds.flower.nodes.prepare(conns, target_column = "diagnosis",
                        feature_columns = c("age", "bmi", "glucose"))

# 3. Start SuperLink and connect nodes
ds.flower.superlink.start(insecure = FALSE)  # TLS enabled
ds.flower.nodes.ensure(conns)

# 4. Define and run experiment
recipe <- ds.flower.recipe(
  task     = ds.flower.task.classification(),
  model    = ds.flower.model.sklearn_logreg(C = 1.0),
  strategy = ds.flower.strategy.fedavg(min_fit_clients = 2L),
  num_rounds      = 10L,
  target_column   = "diagnosis",
  feature_columns = c("age", "bmi", "glucose")
)

result <- ds.flower.run.start(recipe, verbose = TRUE)

# 5. Clean up
ds.flower.nodes.cleanup(conns)
ds.flower.superlink.stop()
DSI::datashield.logout(conns)
```

## Guides

- [Getting Started](articles/getting-started.md) — full walkthrough with
  live demo across 3 Opal servers
- [Experiment Recipes](articles/experiment-recipes.md) — all models,
  strategies, and privacy settings
- [Secure Connections](articles/secure-connections.md) — TLS
  auto-certificate generation explained

## Installation

``` r

# Install from GitHub
remotes::install_github("dsFlower-framework/dsFlowerClient")
```

## Requirements

- R \>= 4.1.0
- Python \>= 3.8 with Flower (`pip install flwr`)
- [DSI](https://cran.r-project.org/package=DSI) and
  [DSOpal](https://cran.r-project.org/package=DSOpal)
- Opal/Rock servers with
  [dsFlower](https://github.com/dsFlower-framework/dsFlower) installed
