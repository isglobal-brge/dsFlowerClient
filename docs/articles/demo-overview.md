# Demo Results Overview

This page is the landing point for the rendered dsFlowerClient demos. It
links to each transparent workflow article and records the final
validation numbers that were available at handoff time. The demos are
intended to show how the researcher connects to Opal/DataSHIELD, stages
or references server-side data, launches Flower training, and checks the
resulting federated model.

## Demo Results

``` r

demo_results <- data.frame(
  demo = c(
    "Breast Cancer Benchmark",
    "Heart Disease Benchmark",
    "MedMNIST Benchmark",
    "Radiomics Benchmark",
    "Radiomics To Flower",
    "Direct Image ResNet",
    "Method Validation Suite",
    "Vision Method Validation"
  ),
  article = c(
    "demo-breast-cancer.html",
    "demo-heart-disease.html",
    "demo-medmnist.html",
    "demo-lung1-radiomics.html",
    "radiomics-to-flower.html",
    "direct-image-resnet.html",
    "validation-overview.html",
    "validation-vision-overview.html"
  ),
  data_path = c(
    "mlbench BreastCancer table",
    "UCI Cleveland or deterministic fallback",
    "BreastMNIST pooled image features",
    "LUNG1-style radiomics features",
    "dsImaging-derived LUNG1 radiomics table",
    "dsImaging direct NIfTI image resource",
    "Synthetic tabular/sequence/survival cohort",
    "Synthetic PNG image/mask cohort"
  ),
  sites = c(3L, 3L, 3L, 3L, 3L, 3L, 3L, 3L),
  central_metric = c(
    "AUC 0.9878",
    "AUC 0.8919",
    "AUC 0.7654",
    "AUC 0.7217",
    "not applicable",
    "loss 0.5540",
    "17/17 methods pass",
    "3/3 methods pass"
  ),
  federated_metric = c(
    "AUC 0.9908",
    "AUC 0.8926",
    "AUC 0.7985",
    "AUC 0.7205",
    "loss 0.6503 -> 0.6474",
    "loss 0.5847",
    "17/17 acceptable",
    "3/3 acceptable"
  ),
  failures = c(0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L),
  stringsAsFactors = FALSE
)
knitr::kable(demo_results)
```

| demo | article | data_path | sites | central_metric | federated_metric | failures |
|:---|:---|:---|---:|:---|:---|---:|
| Breast Cancer Benchmark | demo-breast-cancer.html | mlbench BreastCancer table | 3 | AUC 0.9878 | AUC 0.9908 | 0 |
| Heart Disease Benchmark | demo-heart-disease.html | UCI Cleveland or deterministic fallback | 3 | AUC 0.8919 | AUC 0.8926 | 0 |
| MedMNIST Benchmark | demo-medmnist.html | BreastMNIST pooled image features | 3 | AUC 0.7654 | AUC 0.7985 | 0 |
| Radiomics Benchmark | demo-lung1-radiomics.html | LUNG1-style radiomics features | 3 | AUC 0.7217 | AUC 0.7205 | 0 |
| Radiomics To Flower | radiomics-to-flower.html | dsImaging-derived LUNG1 radiomics table | 3 | not applicable | loss 0.6503 -\> 0.6474 | 0 |
| Direct Image ResNet | direct-image-resnet.html | dsImaging direct NIfTI image resource | 3 | loss 0.5540 | loss 0.5847 | 0 |
| Method Validation Suite | validation-overview.html | Synthetic tabular/sequence/survival cohort | 3 | 17/17 methods pass | 17/17 acceptable | 0 |
| Vision Method Validation | validation-vision-overview.html | Synthetic PNG image/mask cohort | 3 | 3/3 methods pass | 3/3 acceptable | 0 |

## Article Links

- [API
  Quickstart](https://isglobal-brge.github.io/dsFlowerClient/articles/api-quickstart.md)
- [Breast Cancer
  Benchmark](https://isglobal-brge.github.io/dsFlowerClient/articles/demo-breast-cancer.md)
- [Heart Disease
  Benchmark](https://isglobal-brge.github.io/dsFlowerClient/articles/demo-heart-disease.md)
- [MedMNIST
  Benchmark](https://isglobal-brge.github.io/dsFlowerClient/articles/demo-medmnist.md)
- [Radiomics
  Benchmark](https://isglobal-brge.github.io/dsFlowerClient/articles/demo-lung1-radiomics.md)
- [Radiomics To
  Flower](https://isglobal-brge.github.io/dsFlowerClient/articles/radiomics-to-flower.md)
- [Direct Image
  ResNet](https://isglobal-brge.github.io/dsFlowerClient/articles/direct-image-resnet.md)
- [All Method Validation
  Overview](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-overview.md)
- [Vision Method Validation
  Overview](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-vision-overview.md)

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
