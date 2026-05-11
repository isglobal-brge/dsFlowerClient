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

For a direct-image smoke test, `inst/demos/direct_image_resnet_smoke.R`
generates small synthetic PNG files, places them on the Rock filesystem,
uploads only metadata tables to Opal, and trains one federated ResNet-18 round:

```sh
export DSFLOWER_OPAL_URLS="https://localhost:8443,https://localhost:8444,https://localhost:8445"
export OPAL_USER="administrator"
export OPAL_PASSWORD="admin123"
Rscript inst/demos/direct_image_resnet_smoke.R
```

This demo uses `sandbox_open` by default because it is intentionally tiny.
Clinical direct-image runs should use larger cohorts and a SecAgg-capable
runtime/profile.

For a direct handoff from `dsImaging`, `inst/demos/dsimaging_direct_image_resnet.R`
connects to an Opal imaging resource, stages the image assets through the
server-side `dsFlower`/`dsImaging` descriptor path, and trains a federated
ResNet-18 model directly from the stored image files:

```sh
export DSFLOWER_OPAL_URLS="https://localhost:8443,https://localhost:8444,https://localhost:8445"
export DSFLOWER_IMAGING_RESOURCE="dsdemo.lung1_study"
export DSFLOWER_IMAGING_TARGET="os_2yr_alive"
Rscript inst/demos/dsimaging_direct_image_resnet.R
```

For a real imaging handoff, `inst/demos/lung1_radiomics_to_flower.R` consumes
published `dsImaging` radiomics assets, loads them as a server-side `rad` table,
and trains a federated model with `ds.flower.fit()`:

```sh
export LUNG1_WORKDIR="/tmp/dsimaging_lung1_full"
export LUNG1_OPAL_RESOURCE="lung1_full_study"
Rscript inst/demos/lung1_radiomics_to_flower.R
```

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

# Happy path: connect, build recipe, train, and clean up the Flower handle
result <- ds.flower.fit(
  conns,
  symbol = "D",
  target = "outcome",
  features = c("age", "chol", "thalach"),
  model = "sklearn_logreg",
  strategy = "fedavg",
  privacy = "auto",
  rounds = 5L
)

# Advanced path remains available for full control
flower <- ds.flower.connect(conns, symbol = "D")
recipe <- ds.flower.recipe(
  model = ds.flower.model("mlp", hidden_layers = c(64, 32)),
  strategy = ds.flower.strategy("fedprox", proximal_mu = 0.1),
  privacy = ds.flower.privacy("clinical_default"),
  target = "outcome",
  features = c("age", "chol", "thalach"),
  num_rounds = 10L
)
result <- ds.flower.run(flower, recipe)
ds.flower.disconnect(flower)

# Recipes also accept character names directly
recipe <- ds.flower.recipe(
  model = "sklearn_logreg",
  strategy = "fedavg",
  privacy = "trusted_internal",
  target = "outcome",
  features = c("age", "chol", "thalach")
)

# Cleanup
DSI::datashield.logout(conns)
```

## Models

20 models across scikit-learn, PyTorch, and XGBoost. See the [dsFlower README](https://github.com/isglobal-brge/dsFlower) for the full list.

## Authors

- **David Sarrat González** — david.sarrat@isglobal.org
- **Juan R González** — juanr.gonzalez@isglobal.org

[Barcelona Institute for Global Health (ISGlobal)](https://www.isglobal.org/)
