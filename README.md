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

The `inst/demos` directory also includes three public central-vs-federated
benchmark scripts. Each script uploads site partitions to the configured Opal
servers, runs Flower through DataSHIELD, compares the federated model with a
local baseline, and writes `summary.json` plus `benchmark.rds` under
`dsflower_output/demo_benchmarks`.

```sh
export DSFLOWER_OPAL_URLS="https://localhost:8443,https://localhost:8444,https://localhost:8445"
export OPAL_USER="administrator"
export OPAL_PASSWORD="admin123"

Rscript inst/demos/benchmark_breast_cancer.R
Rscript inst/demos/benchmark_heart_disease.R
Rscript inst/demos/benchmark_medmnist.R
```

The same demos are documented as pkgdown articles. Set
`DSFLOWER_RENDER_LIVE_VIGNETTES=true` before rendering if the vignettes should
execute the Opal runs instead of showing the recorded output from the latest
live run.

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
ResNet-18 model directly from the stored image files. When
`DSFLOWER_IMAGING_LOCAL_WORKDIR` points to a local copy of the same study, the
script also trains a centralized ResNet-18 baseline and writes
`local_vs_federated.json`:

```sh
export DSFLOWER_OPAL_URLS="https://localhost:8443,https://localhost:8444,https://localhost:8445"
export DSFLOWER_IMAGING_RESOURCE="dsdemo.imaging_demo"
export DSFLOWER_IMAGING_TARGET="os_2yr_alive"
export DSFLOWER_IMAGING_LOCAL_WORKDIR="/tmp/dsimaging_lung1_study"
Rscript inst/demos/dsimaging_direct_image_resnet.R
```

Validation run on 2026-05-11: three Opal/Rock clients, nine LUNG1 NIfTI images,
one federated round, zero client failures, federated loss `0.5847` vs
centralized local loss `0.5540` on the same tiny smoke cohort.

## Method validation suite

`inst/demos/validate_methods.R` validates the tabular, survival, sequence, and
XGBoost method templates against centralized baselines. It creates one synthetic
cohort, uploads site partitions to three Opal/Rock nodes, runs each federated
template, and writes `inst/extdata/dsflower_method_validation_results.json` for
the pkgdown validation vignettes.

```sh
Rscript inst/demos/validate_methods.R
DSFLOWER_VALIDATE_METHODS=pytorch_multilabel Rscript inst/demos/validate_methods.R
```

Validation run on 2026-05-11: all 17 non-vision methods passed against
centralized baselines with zero client failures. XGBoost now validates in the
trusted sandbox profile through the histogram template and native model-artifact
evaluation; clinical/consortium profiles still enforce SecAgg through the
DataSHIELD trust policy.

The suite fails fast when a method has client failures, missing finite losses,
or a federated loss outside the configured acceptance envelope. The generated
JSON records the centralized loss, federated loss, delta, acceptance margin, and
per-method pass/fail status used by the validation vignettes.

## Vision method validation

`inst/demos/validate_vision_methods.R` validates the image templates separately
from the tabular suite. It generates synthetic PNG images and masks, copies the
files to the Rock-side image root, uploads only metadata tables to Opal, and
compares federated training with centralized PyTorch baselines.

```sh
Rscript inst/demos/validate_vision_methods.R
DSFLOWER_VALIDATE_VISION_METHODS=pytorch_unet2d Rscript inst/demos/validate_vision_methods.R
```

Validation run on 2026-05-11: `pytorch_resnet18`, `pytorch_densenet121`, and
`pytorch_unet2d` all passed on three Opal/Rock nodes with zero client failures.
The run writes `inst/extdata/dsflower_vision_validation_results.json` for the
vision validation pkgdown articles, including the same acceptance fields used by
the non-vision suite.

For a real imaging handoff, `inst/demos/lung1_radiomics_to_flower.R` consumes
published `dsImaging` radiomics assets, loads them as a server-side `rad` table,
and trains a federated model with `ds.flower.fit()`:

```sh
export LUNG1_WORKDIR="/tmp/dsimaging_lung1_full"
export LUNG1_OPAL_RESOURCE="lung1_full_study"
Rscript inst/demos/lung1_radiomics_to_flower.R
```

## Clinical privacy and method-family validation

`inst/demos/benchmark_clinical_algorithms.R` runs public clinical
central-vs-federated benchmarks under `trusted_internal`, `clinical_default`
and `high_sensitivity_dp`. The clinical profile evidence covers logistic
regression, ridge, SGD, PyTorch logistic regression, PyTorch MLP and the
one-round secure histogram-stump XGBoost path. The DP evidence covers PyTorch
MLP and PyTorch logistic regression with Opacus DP-SGD, with epsilon curves at
2, 4 and 8 on the same CDC split. XGBoost also has a separate update-noise
curve at epsilon 8, 12 and 16 for the secure histogram route.

```sh
DSFLOWER_DEMO_PRIVACY_PROFILE=clinical_default \
DSFLOWER_CLINICAL_MODELS=sklearn_logreg,sklearn_ridge,sklearn_sgd,pytorch_logreg,pytorch_mlp,xgboost_histogram \
Rscript inst/demos/benchmark_clinical_algorithms.R
```

`inst/demos/benchmark_method_families.R` complements the binary classification
benchmarks with SUPPORT2 clinical tasks under `clinical_default`. It validates
continuous regression, count regression, multiclass classification, multilabel
classification and Cox survival over three 1,000-row Opal partitions.

```sh
DSFLOWER_SUPPORT2_LIMIT=3000 Rscript inst/demos/benchmark_method_families.R
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
