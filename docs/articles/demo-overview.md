# Demo Results Overview

This page is the landing point for the rendered dsFlowerClient demos.
The main evidence is split into three groups: public benchmark runs with
centralised and federated test metrics, LUNG1 imaging handoff runs that
show composition with dsImaging, and the template validation suite that
exercises every authorised method. Together they show the same
DataSHIELD execution pattern: the researcher connects to Opal,
references or stages server-side data, launches Flower training, and
inspects approved model-scale outputs.

## Public Benchmark Results

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

## Imaging Handoff Results

``` r

imaging_results <- data.frame(
  demo = c(
    "LUNG1 dsImaging Radiomics Evidence",
    "LUNG1 Direct dsImaging Image Evidence"
  ),
  article = c(
    "lung1-radiomics-to-flower.html",
    "lung1-direct-image-dsimaging-resnet.html"
  ),
  data_path = c(
    "dsImaging-derived LUNG1 radiomics table",
    "dsImaging-resolved LUNG1 NIfTI assets"
  ),
  sites = c(3L, 3L),
  result = c(
    "422 rows; loss 0.6503 -> 0.6474; 0 client failures",
    "9 images; central loss 0.5540; federated loss 0.5847; 0 client failures"
  ),
  stringsAsFactors = FALSE
)
knitr::kable(imaging_results)
```

| demo | article | data_path | sites | result |
|:---|:---|:---|---:|:---|
| LUNG1 dsImaging Radiomics Evidence | lung1-radiomics-to-flower.html | dsImaging-derived LUNG1 radiomics table | 3 | 422 rows; loss 0.6503 -\> 0.6474; 0 client failures |
| LUNG1 Direct dsImaging Image Evidence | lung1-direct-image-dsimaging-resnet.html | dsImaging-resolved LUNG1 NIfTI assets | 3 | 9 images; central loss 0.5540; federated loss 0.5847; 0 client failures |

## Template Validation Coverage

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
  stringsAsFactors = FALSE
)
knitr::kable(validation_results)
```

| suite | article | data_path | sites | result |
|:---|:---|:---|---:|:---|
| Non-vision method validation | validation-overview.html | Controlled tabular, sequence and survival cohort | 3 | 17/17 methods pass |
| Vision method validation | validation-vision-overview.html | Controlled image and mask cohort | 3 | 3/3 methods pass |

## Article Links

- [API
  Quickstart](https://isglobal-brge.github.io/dsFlowerClient/articles/api-quickstart.md)
- [Breast Cancer
  Benchmark](https://isglobal-brge.github.io/dsFlowerClient/articles/demo-breast-cancer.md)
- [Heart Disease
  Benchmark](https://isglobal-brge.github.io/dsFlowerClient/articles/demo-heart-disease.md)
- [MedMNIST
  Benchmark](https://isglobal-brge.github.io/dsFlowerClient/articles/demo-medmnist.md)
- [LUNG1 Direct dsImaging Image
  Evidence](https://isglobal-brge.github.io/dsFlowerClient/articles/lung1-direct-image-dsimaging-resnet.md)
- [LUNG1 dsImaging Radiomics
  Evidence](https://isglobal-brge.github.io/dsFlowerClient/articles/lung1-radiomics-to-flower.md)
- [All Method Validation
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
