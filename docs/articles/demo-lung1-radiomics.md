# Radiomics Federated Benchmark

This vignette demonstrates downstream federated modeling on
radiomics-derived features. By default the script uses a deterministic
LUNG1-style fixture with feature names and signal structure similar to
radiomics studies. When real radiomics features are available, point the
inline loader at a CSV/RDS file:

``` r

external_path <- Sys.getenv("DSFLOWER_LUNG1_FEATURES", "/path/to/radiomics_features.csv")
target_column <- Sys.getenv("DSFLOWER_LUNG1_TARGET", "outcome")
```

The external file must contain an `id` column or one will be generated.
The target column is renamed internally to `outcome`, and all numeric
columns other than the target are used as model features.

## What The Script Does

This benchmark is the lightweight radiomics-feature modeling path. It
does not run segmentation or radiomics extraction; it assumes those have
happened upstream and validates that a radiomics-like feature matrix can
train through the same DataSHIELD/Flower path.

``` r

external_path <- Sys.getenv("DSFLOWER_LUNG1_FEATURES", "")
dataset <- load_external_radiomics(external_path)

if (is.null(dataset)) {
  dataset <- make_lung1_style()
}

features <- setdiff(names(dataset), c("id", "outcome"))

run_dsflower_benchmark(
  data = dataset,
  features = features,
  demo_id = "lung1_radiomics",
  dataset_label = "LUNG1-style radiomics federated benchmark",
  data_mode = "deterministic LUNG1-style derived radiomics fixture",
  default_rounds = 2L
)
```

The helper uploads one standardized training table per Opal and then
calls:

``` r

fit <- ds.flower.fit(
  conns,
  symbol = "D",
  target = "outcome",
  features = features,
  model = "sklearn_logreg",
  model_params = list(max_iter = 100L),
  strategy = "fedavg",
  privacy = "trusted_internal",
  rounds = 2L
)
```

Validation run on 2026-05-12:

    #>               metric    value
    #> 1               rows 420.0000
    #> 2              train 316.0000
    #> 3               test 104.0000
    #> 4              sites   3.0000
    #> 5           features  10.0000
    #> 6        central_auc   0.7217
    #> 7      federated_auc   0.7205
    #> 8   central_accuracy   0.6827
    #> 9 federated_accuracy   0.6635

This vignette covers the modeling stage. Image segmentation and
radiomics extraction should happen upstream in `dsImaging`; this
benchmark consumes the derived feature table as a federated learning
input.
