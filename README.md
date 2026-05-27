# dsFlowerClient

`dsFlowerClient` is the researcher-side R package for running controlled
[Flower](https://flower.ai/) federated learning workflows through
[DataSHIELD](https://www.datashield.org/). It works with the server-side
[`dsFlower`](https://github.com/isglobal-brge/dsFlower) package installed on
each Opal/Rock node.

The package starts a local Flower SuperLink, asks each DataSHIELD server to
stage data and start a Flower SuperNode, launches an authorised template and
returns only the outputs permitted by the active server-side privacy profile.
Raw rows, images, masks and local staging files remain inside the data-owning
servers.

## Current validation evidence

The documentation separates current biomedical evidence from catalogue coverage.
The recent validation evidence is the one used for presentation and thesis
claims; the synthetic catalogue suite complements it by exercising the full
template surface.

| Evidence layer | Dataset or source | Profile/mechanism | Current result |
|---|---|---|---|
| Clinical benchmark matrix | Breast Cancer Wisconsin, UCI Heart Disease, Pima Indians Diabetes, CDC Diabetes Health Indicators | `trusted_internal`, `clinical_default`, `clinical_update_noise`, `high_sensitivity_dp` | Centralised-vs-federated held-out metrics with zero client failures across the committed runs. |
| Clinical Secure Aggregation | Breast Cancer Wisconsin, Pima, CDC Diabetes | `clinical_default` | SecAgg+ enforced, per-node metrics suppressed, all selected runs pass. |
| Clinical DP-SGD | CDC Diabetes Health Indicators | `high_sensitivity_dp` with Opacus DP-SGD | PyTorch logistic regression and MLP validated at epsilon 8, with additional epsilon curves at 2, 4 and 8. |
| XGBoost histogram update noise | CDC Diabetes Health Indicators | `clinical_update_noise` | Secure histogram route validated with update-noise curve at epsilon 8, 12 and 16. |
| SUPPORT2 method families | SUPPORT2 critical-care study | `clinical_default` and `high_sensitivity_dp` | Regression, count, multiclass, multilabel and Cox survival validated under SecAgg; four DP-SGD-compatible families repeated under DP-SGD. |
| Imaging handoff | LUNG1 radiomics and direct NIfTI assets from `dsImaging` | `trusted_internal` path evidence | Radiomics-derived logistic regression and direct-image ResNet path validate the `dsImaging` to `dsFlower` handoff. |
| Direct biomedical images | MedMNIST v2 PathMNIST | `trusted_internal` and `consortium_internal` | 1,500 server-side images; centralised accuracy 0.9947 and federated checkpoint accuracy 0.9860; SecAgg branch completes with global checkpoint. |
| Catalogue coverage | Deterministic synthetic tabular, sequence, survival, image and mask fixtures | `sandbox_open` | 17 non-vision and 3 vision templates execute end to end, complementing the biomedical benchmarks with full template-surface coverage. |

Rendered reports are available in the pkgdown site:
<https://isglobal-brge.github.io/dsFlowerClient/>.

## Installation

```r
remotes::install_github("isglobal-brge/dsFlowerClient")
```

The client runtime needs the Flower command-line tools. A simple local setup is:

```sh
python -m pip install "flwr>=1.13.0"
```

Each Opal/Rock server must have `dsFlower` installed and configured.

## Basic use

For most analyses, use `ds.flower.fit()`. It performs the full lifecycle:
capability checks, server-side preparation, SuperLink start, SuperNode start,
Flower run, result collection and cleanup.

```r
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

conns <- DSI::datashield.login(builder$build(), assign = TRUE, symbol = "D")

fit <- ds.flower.fit(
  conns,
  symbol = "D",
  target = "outcome",
  features = c("age", "cholesterol", "biomarker"),
  model = "sklearn_logreg",
  strategy = "fedavg",
  privacy = "clinical_default",
  rounds = 5L
)

ds.flower.metrics(fit)
DSI::datashield.logout(conns)
```

The lower-level API remains available when a study needs to inspect or modify
the recipe explicitly:

```r
flower <- ds.flower.connect(conns, symbol = "D")

recipe <- ds.flower.recipe(
  model = ds.flower.model("pytorch_mlp", hidden_layers = c(64, 32)),
  strategy = ds.flower.strategy("fedprox", proximal_mu = 0.1),
  privacy = ds.flower.privacy("clinical_default"),
  target = "outcome",
  features = c("age", "cholesterol", "biomarker"),
  num_rounds = 10L
)

result <- ds.flower.run(flower, recipe)
ds.flower.disconnect(flower)
```

## Privacy profiles

Server-side privacy profiles define the effective policy. The client request is
validated against what each server allows.

| Profile | Use |
|---|---|
| `sandbox_open` | Local development and catalogue coverage; requires explicit server opt-in. |
| `trusted_internal` | Controlled internal experiments without required Secure Aggregation. |
| `consortium_internal` | Multi-site runs with SecAgg+, fixed participation and suppressed per-node metrics. |
| `clinical_default` | Default clinical profile with SecAgg+, stricter row guards and controlled metric release. |
| `clinical_hardened` | Higher-threshold clinical profile. |
| `clinical_update_noise` | SecAgg+ plus bounded update or histogram noise where the template supports it. |
| `high_sensitivity_dp` | SecAgg+ plus patient-level DP-SGD for templates validated with Opacus per-example gradients. |

`clinical_update_noise` is not presented as formal patient-level DP-SGD. It is
an update or histogram hardening profile. Use `high_sensitivity_dp` when the
study requires patient-level DP-SGD and the selected template supports it.

## Re-running the current evidence

The commands below assume a local three-Opal/Rock stack with the server package
installed and published.

```sh
export DSFLOWER_OPAL_URLS="https://localhost:8443,https://localhost:8444,https://localhost:8445"
export OPAL_USER="administrator"
export OPAL_PASSWORD="admin123"
```

Clinical algorithm and privacy-profile matrix:

```sh
DSFLOWER_DEMO_PRIVACY_PROFILE=clinical_default \
DSFLOWER_CLINICAL_MODELS=sklearn_logreg,sklearn_ridge,sklearn_sgd,pytorch_logreg,pytorch_mlp,xgboost_histogram \
Rscript inst/demos/benchmark_clinical_algorithms.R
```

SUPPORT2 method-family benchmarks:

```sh
DSFLOWER_SUPPORT2_LIMIT=3000 \
DSFLOWER_FAMILY_PRIVACY_PROFILE=clinical_default \
Rscript inst/demos/benchmark_method_families.R

DSFLOWER_SUPPORT2_LIMIT=3000 \
DSFLOWER_FAMILY_PRIVACY_PROFILE=high_sensitivity_dp \
DSFLOWER_FAMILY_MODELS=pytorch_linear_regression,pytorch_poisson,pytorch_multiclass,pytorch_multilabel \
DSFLOWER_FAMILY_MULTICLASS_ROUNDS=6 \
Rscript inst/demos/benchmark_method_families.R
```

PathMNIST direct-image benchmark:

```sh
DSFLOWER_PATHMNIST_N_PER_SITE=500 \
DSFLOWER_PATHMNIST_PRIVACY_PROFILES=trusted_internal,consortium_internal \
Rscript inst/demos/benchmark_pathmnist_direct_image.R
```

LUNG1 radiomics handoff from `dsImaging`:

```sh
export LUNG1_WORKDIR="/tmp/dsimaging_lung1_full"
export LUNG1_OPAL_PROJECT="dsdemo"
export LUNG1_OPAL_RESOURCE="lung1_full_study"
Rscript inst/demos/lung1_radiomics_to_flower.R
```

LUNG1 direct-image handoff from `dsImaging`:

```sh
export DSFLOWER_IMAGING_RESOURCE="dsdemo.imaging_demo"
export DSFLOWER_IMAGING_TARGET="os_2yr_alive"
export DSFLOWER_IMAGING_LOCAL_WORKDIR="/tmp/dsimaging_lung1_study"
Rscript inst/demos/dsimaging_direct_image_resnet.R
```

## Catalogue coverage and development demos

The catalogue validation scripts are still useful, but they should be read as
product-surface checks. They use deterministic synthetic fixtures to confirm
that every authorised template can be staged, executed, aggregated, persisted
and cleaned up.

```sh
Rscript inst/demos/validate_methods.R
Rscript inst/demos/validate_vision_methods.R
```

The repository also keeps smaller development demos:

```sh
Rscript inst/demos/opal_federated_logreg.R
Rscript inst/demos/benchmark_breast_cancer.R
Rscript inst/demos/benchmark_heart_disease.R
Rscript inst/demos/benchmark_medmnist.R
Rscript inst/demos/direct_image_resnet_smoke.R
```

These are useful when checking installation, connectivity or template mechanics;
the current validation reports above provide the biomedical evidence used for
presentation and thesis claims.

## Models

The client exposes 20 model constructors across scikit-learn, PyTorch and
XGBoost. See the `dsFlower` README and the pkgdown reference index for the
current template catalogue and privacy-profile compatibility.

## Authors

- **David Sarrat González** — david.sarrat@isglobal.org
- **Juan R González** — juanr.gonzalez@isglobal.org

[Barcelona Institute for Global Health (ISGlobal)](https://www.isglobal.org/)
