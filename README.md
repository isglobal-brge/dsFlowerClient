# dsFlowerClient

Client-side R package for [dsFlower](https://github.com/isglobal-brge/dsFlower) — federated learning on [DataSHIELD](https://www.datashield.org/) powered by [Flower](https://flower.ai/).

## Installation

```r
remotes::install_github("isglobal-brge/dsFlowerClient")
```

Requires Python with [Flower](https://flower.ai/): `pip install flwr>=1.13.0`

## Local Opal demo

The package ships a runnable three-site Opal demo that creates synthetic
balanced tables, configures the local Opal profile, and trains a federated
scikit-learn logistic regression model:

```sh
Rscript "$(Rscript -e 'cat(system.file("demos", "opal_federated_logreg.R", package = "dsFlowerClient"))')"
```

By default the demo uses `DSFLOWER_DEMO_PRIVACY=trusted_internal`. Profiles
that require Secure Aggregation are only allowed when the connected servers
report `secure_aggregation_supported = TRUE`.

## Benchmark demos

The `inst/demos` directory also includes four central-vs-federated benchmark
scripts. Each script uploads site partitions to the configured Opal servers,
runs Flower through DataSHIELD, compares the federated model with a local
baseline, and writes `summary.json` plus `benchmark.rds` under
`dsflower_output/demo_benchmarks`.

```sh
export DSFLOWER_OPAL_URLS="https://localhost:8443,https://localhost:8444,https://localhost:8445"
export OPAL_USER="administrator"
export OPAL_PASSWORD="admin123"

Rscript inst/demos/benchmark_breast_cancer.R
Rscript inst/demos/benchmark_heart_disease.R
Rscript inst/demos/benchmark_medmnist.R
Rscript inst/demos/benchmark_lung1_radiomics.R
```

The same demos are documented as pkgdown articles. Set
`DSFLOWER_RUN_VIGNETTE_DEMOS=true` before rendering if the vignettes should
execute the Opal runs instead of only showing the runnable workflow.

## Usage

```r
library(dsFlowerClient)
library(DSI)
library(DSOpal)

# Connect to Opal nodes
builder <- DSI::newDSLoginBuilder()
builder$append(server = "site1", url = "https://opal1.example.org",
               user = "researcher", password = "...",
               table = "PROJECT.data", driver = "OpalDriver")
builder$append(server = "site2", url = "https://opal2.example.org",
               user = "researcher", password = "...",
               table = "PROJECT.data", driver = "OpalDriver")
conns <- DSI::datashield.login(logins = builder$build(),
                               assign = TRUE, symbol = "D")
flower <- ds.flower.connect(conns, symbol = "D")

# Train
result <- ds.flower.run(flower, ds.flower.recipe(
  model         = ds.flower.model.pytorch_mlp(hidden_layers = "64,32"),
  strategy      = ds.flower.strategy.fedprox(proximal_mu = 0.1),
  target_column = "outcome",
  num_rounds    = 10L
))

# Cleanup
ds.flower.disconnect(flower)
DSI::datashield.logout(conns)
```

## Models

20 models across scikit-learn, PyTorch, and XGBoost. See the [dsFlower README](https://github.com/isglobal-brge/dsFlower) for the full list.

## Authors

- **David Sarrat González** — david.sarrat@isglobal.org
- **Juan R González** — juanr.gonzalez@isglobal.org

[Barcelona Institute for Global Health (ISGlobal)](https://www.isglobal.org/)
