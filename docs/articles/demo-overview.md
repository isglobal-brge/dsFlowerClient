# Demo Results Overview

This page is the landing point for the rendered dsFlowerClient demos.
The current evidence is split into four groups: clinical privacy-profile
runs, SUPPORT2 method-family runs, imaging handoff/direct-image runs and
catalogue coverage. The older self-contained public demos remain useful
for installation checks, but the presentation evidence is the recent
clinical, imaging and privacy-profile matrix.

All live demos use the same DataSHIELD execution pattern: the researcher
connects to Opal, references or stages server-side data, launches Flower
training, and inspects approved model-scale outputs.

## Current Clinical And Privacy Results

``` r

clinical_privacy <- data.frame(
  profile = c("trusted_internal", "clinical_default",
              "clinical_update_noise XGBoost",
              "high_sensitivity_dp", "DP-SGD epsilon curves",
              "XGBoost update-noise curve",
              "clinical_default method families",
              "high_sensitivity_dp method families"),
  article = c(rep("clinical-privacy-benchmarks.html", 6),
              "clinical-method-family-benchmarks.html",
              "clinical-method-family-benchmarks.html"),
  data_path = c(
    "Breast Cancer, Heart Disease, Pima Diabetes and CDC Diabetes",
    "Breast Cancer, Pima Diabetes and CDC Diabetes",
    "CDC Diabetes",
    "CDC Diabetes",
    "CDC Diabetes",
    "CDC Diabetes",
    "SUPPORT2",
    "SUPPORT2"
  ),
  mechanism = c(
    "Controlled federated execution",
    "Secure Aggregation plus stricter metric/count policy",
    "Secure Aggregation plus histogram update noise",
    "Secure Aggregation plus Opacus DP-SGD",
    "Opacus DP-SGD at epsilon 2, 4 and 8 for two PyTorch templates",
    "Histogram update noise at epsilon 8, 12 and 16",
    "Secure Aggregation plus family-specific loss validation",
    "Secure Aggregation plus Opacus DP-SGD for the four compatible SUPPORT2 families"
  ),
  result = c(
    "6/6 runs pass, 0 client failures",
    "8/8 runs pass, 0 client failures",
    "1/1 run passes, 0 client failures",
    "2/2 runs pass, 0 client failures",
    "6/6 curve points pass, 0 client failures",
    "3/3 curve points pass, 0 client failures",
    "5/5 runs pass, 0 client failures",
    "4/4 DP-SGD-compatible runs pass, 0 client failures"
  ),
  stringsAsFactors = FALSE
)
knitr::kable(clinical_privacy)
```

| profile | article | data_path | mechanism | result |
|:---|:---|:---|:---|:---|
| trusted_internal | clinical-privacy-benchmarks.html | Breast Cancer, Heart Disease, Pima Diabetes and CDC Diabetes | Controlled federated execution | 6/6 runs pass, 0 client failures |
| clinical_default | clinical-privacy-benchmarks.html | Breast Cancer, Pima Diabetes and CDC Diabetes | Secure Aggregation plus stricter metric/count policy | 8/8 runs pass, 0 client failures |
| clinical_update_noise XGBoost | clinical-privacy-benchmarks.html | CDC Diabetes | Secure Aggregation plus histogram update noise | 1/1 run passes, 0 client failures |
| high_sensitivity_dp | clinical-privacy-benchmarks.html | CDC Diabetes | Secure Aggregation plus Opacus DP-SGD | 2/2 runs pass, 0 client failures |
| DP-SGD epsilon curves | clinical-privacy-benchmarks.html | CDC Diabetes | Opacus DP-SGD at epsilon 2, 4 and 8 for two PyTorch templates | 6/6 curve points pass, 0 client failures |
| XGBoost update-noise curve | clinical-privacy-benchmarks.html | CDC Diabetes | Histogram update noise at epsilon 8, 12 and 16 | 3/3 curve points pass, 0 client failures |
| clinical_default method families | clinical-method-family-benchmarks.html | SUPPORT2 | Secure Aggregation plus family-specific loss validation | 5/5 runs pass, 0 client failures |
| high_sensitivity_dp method families | clinical-method-family-benchmarks.html | SUPPORT2 | Secure Aggregation plus Opacus DP-SGD for the four compatible SUPPORT2 families | 4/4 DP-SGD-compatible runs pass, 0 client failures |

## Imaging Handoff Results

``` r

imaging_results <- data.frame(
  demo = c(
    "LUNG1 dsImaging Radiomics Evidence",
    "LUNG1 Direct dsImaging Image Evidence",
    "PathMNIST Direct Image ResNet Benchmark"
  ),
  article = c(
    "lung1-radiomics-to-flower.html",
    "lung1-direct-image-dsimaging-resnet.html",
    "pathmnist-direct-image-resnet.html"
  ),
  data_path = c(
    "dsImaging-derived LUNG1 radiomics table",
    "dsImaging-resolved LUNG1 NIfTI assets",
    "MedMNIST PathMNIST server-side PNG files"
  ),
  sites = c(3L, 3L, 3L),
  result = c(
    "422 rows; loss 0.6503 -> 0.6474; 0 client failures",
    "9 images; central loss 0.5540; federated loss 0.5847; 0 client failures",
    "1,500 images; central accuracy 0.9947; federated accuracy 0.9860; SecAgg branch pass"
  ),
  stringsAsFactors = FALSE
)
knitr::kable(imaging_results)
```

| demo | article | data_path | sites | result |
|:---|:---|:---|---:|:---|
| LUNG1 dsImaging Radiomics Evidence | lung1-radiomics-to-flower.html | dsImaging-derived LUNG1 radiomics table | 3 | 422 rows; loss 0.6503 -\> 0.6474; 0 client failures |
| LUNG1 Direct dsImaging Image Evidence | lung1-direct-image-dsimaging-resnet.html | dsImaging-resolved LUNG1 NIfTI assets | 3 | 9 images; central loss 0.5540; federated loss 0.5847; 0 client failures |
| PathMNIST Direct Image ResNet Benchmark | pathmnist-direct-image-resnet.html | MedMNIST PathMNIST server-side PNG files | 3 | 1,500 images; central accuracy 0.9947; federated accuracy 0.9860; SecAgg branch pass |

## Catalogue Coverage

``` r

validation_results <- data.frame(
  suite = c("Non-vision method validation", "Vision method validation"),
  article = c("validation-overview.html", "validation-vision-overview.html"),
  data_path = c(
    "Controlled tabular, sequence and survival cohort",
    "Controlled image and mask cohort"
  ),
  sites = c(3L, 3L),
  result = c("17/17 methods pass", "3/3 methods pass"),
  role = c("template catalogue coverage", "vision path coverage"),
  stringsAsFactors = FALSE
)
knitr::kable(validation_results)
```

| suite | article | data_path | sites | result | role |
|:---|:---|:---|---:|:---|:---|
| Non-vision method validation | validation-overview.html | Controlled tabular, sequence and survival cohort | 3 | 17/17 methods pass | template catalogue coverage |
| Vision method validation | validation-vision-overview.html | Controlled image and mask cohort | 3 | 3/3 methods pass | vision path coverage |

## Development Public Demos

``` r

public_benchmarks <- data.frame(
  demo = c(
    "Breast Cancer Benchmark",
    "Heart Disease Benchmark",
    "MedMNIST Benchmark"
  ),
  article = c(
    "demo-breast-cancer.html",
    "demo-heart-disease.html",
    "demo-medmnist.html"
  ),
  data_path = c(
    "UCI Breast Cancer Wisconsin (Original)",
    "UCI Heart Disease processed Cleveland",
    "MedMNIST BreastMNIST pooled image features"
  ),
  sites = c(3L, 3L, 3L),
  central_metric = c(
    "AUC 0.9878",
    "AUC 0.8919",
    "AUC 0.7654"
  ),
  federated_metric = c(
    "AUC 0.9908",
    "AUC 0.8926",
    "AUC 0.7985"
  ),
  failures = c(0L, 0L, 0L),
  stringsAsFactors = FALSE
)
knitr::kable(public_benchmarks)
```

| demo | article | data_path | sites | central_metric | federated_metric | failures |
|:---|:---|:---|---:|:---|:---|---:|
| Breast Cancer Benchmark | demo-breast-cancer.html | UCI Breast Cancer Wisconsin (Original) | 3 | AUC 0.9878 | AUC 0.9908 | 0 |
| Heart Disease Benchmark | demo-heart-disease.html | UCI Heart Disease processed Cleveland | 3 | AUC 0.8919 | AUC 0.8926 | 0 |
| MedMNIST Benchmark | demo-medmnist.html | MedMNIST BreastMNIST pooled image features | 3 | AUC 0.7654 | AUC 0.7985 | 0 |

## Article Links

- [API
  Quickstart](https://isglobal-brge.github.io/dsFlowerClient/articles/api-quickstart.md)
- [Breast Cancer
  Benchmark](https://isglobal-brge.github.io/dsFlowerClient/articles/demo-breast-cancer.md)
- [Heart Disease
  Benchmark](https://isglobal-brge.github.io/dsFlowerClient/articles/demo-heart-disease.md)
- [MedMNIST
  Benchmark](https://isglobal-brge.github.io/dsFlowerClient/articles/demo-medmnist.md)
- [Clinical Privacy
  Benchmarks](https://isglobal-brge.github.io/dsFlowerClient/articles/clinical-privacy-benchmarks.md)
- [Clinical Method-Family
  Benchmarks](https://isglobal-brge.github.io/dsFlowerClient/articles/clinical-method-family-benchmarks.md)
- [PathMNIST Direct Image ResNet
  Benchmark](https://isglobal-brge.github.io/dsFlowerClient/articles/pathmnist-direct-image-resnet.md)
- [LUNG1 Direct dsImaging Image
  Evidence](https://isglobal-brge.github.io/dsFlowerClient/articles/lung1-direct-image-dsimaging-resnet.md)
- [LUNG1 dsImaging Radiomics
  Evidence](https://isglobal-brge.github.io/dsFlowerClient/articles/lung1-radiomics-to-flower.md)
- [Validation Evidence
  Overview](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-overview.md)
- [Vision Method Validation
  Overview](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-vision-overview.md)

The repository also keeps self-contained smoke demos for development
(`demo-lung1-radiomics.html`, `radiomics-to-flower.html` and
`direct-image-resnet.html`). They are useful for checking the mechanics
without external data, but the thesis evidence above uses public
datasets or dsImaging assets.

## Common Execution Shape

All live demos use the same DataSHIELD shape: build Opal logins, assign
a server-side symbol, call
[`ds.flower.fit()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.fit.md)
or the lower-level `connect -> prepare -> run` flow, then inspect
metrics and cleanup state.

``` r

library(dsFlowerClient)
library(DSI)
library(DSOpal)

builder <- DSI::newDSLoginBuilder()
builder$append(
  server = "opal1",
  url = "https://localhost:8443",
  user = Sys.getenv("OPAL_USER", "administrator"),
  password = Sys.getenv("OPAL_PASSWORD", "admin123"),
  table = "dsflower_demo.demo_site1",
  driver = "OpalDriver"
)
# Repeat builder$append(...) for opal2 and opal3.

conns <- DSI::datashield.login(
  logins = builder$build(),
  assign = TRUE,
  symbol = "D"
)

fit <- ds.flower.fit(
  conns,
  symbol = "D",
  target = "outcome",
  features = c("x1", "x2", "x3"),
  model = "sklearn_logreg",
  strategy = "fedavg",
  privacy = "trusted_internal",
  rounds = 2L
)

ds.flower.metrics(fit)
DSI::datashield.logout(conns)
```

For full method-by-method validation numbers, use the rendered
validation overview. It reads the committed JSON result files and links
to every individual method article.
