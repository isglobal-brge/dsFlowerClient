# LUNG1 Direct dsImaging Image ResNet Evidence

``` r

knitr::opts_chunk$set(collapse = TRUE, comment = "#>")
artifact_paths <- c(
  file.path("..", "inst", "extdata", "dsflower_lung1_direct_image_results.json"),
  file.path("inst", "extdata", "dsflower_lung1_direct_image_results.json"),
  system.file("extdata", "dsflower_lung1_direct_image_results.json",
              package = "dsFlowerClient")
)
artifact_path <- artifact_paths[nzchar(artifact_paths) & file.exists(artifact_paths)][1]
evidence <- jsonlite::fromJSON(artifact_path, simplifyVector = FALSE)
cat("Evidence artifact:", basename(artifact_path), "\n")
```

    ## Evidence artifact: dsflower_lung1_direct_image_results.json

``` r

cat("Source script: inst/demos/dsimaging_direct_image_resnet.R\n")
```

    ## Source script: inst/demos/dsimaging_direct_image_resnet.R

``` r

cat("Recorded run_id:", evidence$run_id, "\n")
```

    ## Recorded run_id: 12188684698149675604

This article records the LUNG1 direct-image run used as thesis evidence.
The run trained `pytorch_resnet18` directly from NIfTI assets resolved
by `dsImaging`, without first converting the images into radiomic
features. It is a technical path validation: the cohort is deliberately
tiny, and the comparison is not intended as a clinical imaging result.

The committed JSON is the raw `summary.json` emitted by the live May
2026 workflow.

The validation path is:

1.  `dsFlowerClient` connects to a `dsImaging` resource with
    `ds.flower.connect(resource = ...)`.
2.  Server-side metadata and image asset descriptors are staged into
    Flower manifests inside each Rock.
3.  A Flower SuperNode on each Rock trains the `pytorch_resnet18`
    template from local NIfTI files.
4.  A centralized PyTorch baseline trains on the same local demo workdir
    for a path-level comparison.
5.  Cleanup leaves no active SuperNodes.

## Reproduction Command

``` r

Sys.setenv(
  DSFLOWER_OPAL_URLS = "https://localhost:8443,https://localhost:8444,https://localhost:8445",
  OPAL_USER = "administrator",
  OPAL_PASSWORD = "admin123",
  DSFLOWER_IMAGING_RESOURCE = "dsdemo.lung1_study",
  DSFLOWER_IMAGING_TARGET = "os_2yr_alive",
  DSFLOWER_IMAGING_PRIVACY = "sandbox_open",
  DSFLOWER_IMAGING_ROUNDS = "1",
  DSFLOWER_IMAGING_LOCAL_WORKDIR = "/tmp/dsimaging_lung1_study",
  DSFLOWER_IMAGING_LOCAL_BASELINE = "true"
)
source(system.file("demos", "dsimaging_direct_image_resnet.R",
                   package = "dsFlowerClient"))
```

## Server-Side Image Resource

``` r

site_counts <- data.frame(
  site = names(evidence$metadata_n),
  samples = as.integer(unlist(evidence$metadata_n)),
  row.names = NULL
)
class_counts <- data.frame(
  class = names(evidence$local_baseline$class_counts),
  samples = as.integer(unlist(evidence$local_baseline$class_counts)),
  row.names = NULL
)

knitr::kable(site_counts)
```

| site  | samples |
|:------|--------:|
| opal1 |       3 |
| opal2 |       3 |
| opal3 |       3 |

``` r

knitr::kable(class_counts)
```

| class | samples |
|:------|--------:|
| 0     |       7 |
| 1     |       2 |

``` r


data.frame(
  resource = evidence$resource,
  image_type = "NIfTI",
  target = evidence$target,
  total_samples = sum(site_counts$samples),
  stringsAsFactors = FALSE
)
#>             resource image_type       target total_samples
#> 1 dsdemo.lung1_study      NIfTI os_2yr_alive             9
```

## Training Configuration

``` r

training <- data.frame(
  model = evidence$model,
  strategy = "fedavg",
  privacy_profile = evidence$privacy,
  rounds = evidence$rounds,
  local_epochs = evidence$local_baseline$epochs,
  batch_size = evidence$local_baseline$batch_size,
  learning_rate = evidence$local_baseline$learning_rate,
  stringsAsFactors = FALSE
)
knitr::kable(training)
```

| model | strategy | privacy_profile | rounds | local_epochs | batch_size | learning_rate |
|:---|:---|:---|---:|---:|---:|---:|
| pytorch_resnet18 | fedavg | sandbox_open | 1 | 1 | 1 | 5e-04 |

## Centralized vs Federated Result

``` r

comparison <- data.frame(
  run = c("centralized local", "federated dsFlower"),
  engine = c("PyTorch ResNet18", "DataSHIELD + Flower ResNet18"),
  samples = c(evidence$local_baseline$n_samples, sum(site_counts$samples)),
  sites_or_clients = c(1L, evidence$local_vs_federated$federated$clients),
  rounds_or_epochs = c(evidence$local_baseline$epochs,
                       evidence$local_vs_federated$federated$rounds),
  loss = c(evidence$local_baseline$eval$loss,
           evidence$local_vs_federated$federated$loss),
  accuracy = c(evidence$local_baseline$eval$accuracy, NA_real_),
  failures = c(NA_integer_, evidence$local_vs_federated$federated$n_failures)
)
knitr::kable(comparison, digits = 4)
```

| run | engine | samples | sites_or_clients | rounds_or_epochs | loss | accuracy | failures |
|:---|:---|---:|---:|---:|---:|---:|---:|
| centralized local | PyTorch ResNet18 | 9 | 1 | 1 | 0.5540 | 0.7778 | NA |
| federated dsFlower | DataSHIELD + Flower ResNet18 | 9 | 3 | 1 | 0.5847 | NA | 0 |

``` r


cat("[centralized] train loss epoch 1:",
    sprintf("%.6f", unlist(evidence$local_baseline$train_loss_by_epoch)[[1]]), "\n")
#> [centralized] train loss epoch 1: 0.707024
cat("[centralized] eval loss:",
    sprintf("%.6f", evidence$local_baseline$eval$loss), "\n")
#> [centralized] eval loss: 0.554012
cat("[federated] round 1 loss:",
    sprintf("%.6f", evidence$local_vs_federated$federated$loss), "\n")
#> [federated] round 1 loss: 0.584744
cat("[delta] federated - centralized:",
    sprintf("%.6f",
            evidence$local_vs_federated$delta$loss_federated_minus_centralized), "\n")
#> [delta] federated - centralized: 0.030732
cat("[acceptance] federated client failures:",
    evidence$local_vs_federated$federated$n_failures, "\n")
#> [acceptance] federated client failures: 0
```

``` r

plot_data <- data.frame(
  run = factor(comparison$run, levels = comparison$run),
  loss = comparison$loss
)
if (requireNamespace("ggplot2", quietly = TRUE)) {
  ggplot2::ggplot(plot_data, ggplot2::aes(x = run, y = loss, fill = run)) +
    ggplot2::geom_col(width = 0.62, show.legend = FALSE) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.4f", loss)),
                       vjust = -0.35, size = 3.5) +
    ggplot2::scale_fill_manual(values = c("#4C78A8", "#F58518")) +
    ggplot2::coord_cartesian(ylim = c(0, max(plot_data$loss) + 0.15)) +
    ggplot2::labs(x = NULL, y = "Loss",
                  title = "LUNG1 direct-image ResNet path validation") +
    ggplot2::theme_minimal(base_size = 11)
} else {
  barplot(plot_data$loss, names.arg = as.character(plot_data$run),
          ylim = c(0, max(plot_data$loss) + 0.15), ylab = "Loss",
          main = "LUNG1 direct-image ResNet path validation",
          col = c("#4C78A8", "#F58518"))
}
```

![Bar chart comparing centralized and federated ResNet18 loss on nine
LUNG1 NIfTI
images.](lung1-direct-image-dsimaging-resnet_files/figure-html/loss-plot-1.png)

## Cleanup And Evidence Scope

``` r

cleanup <- data.frame(
  site = names(evidence$active_supernodes_after),
  active_supernodes = as.integer(unlist(evidence$active_supernodes_after)),
  row.names = NULL
)
knitr::kable(cleanup)
```

| site  | active_supernodes |
|:------|------------------:|
| opal1 |                 0 |
| opal2 |                 0 |
| opal3 |                 0 |

``` r

cat("[cleanup] PASS:", all(cleanup$active_supernodes == 0L), "\n")
#> [cleanup] PASS: TRUE
```

``` r

scope <- data.frame(
  item = c("Validated claim", "Explicit non-claim", "Recorded output dir"),
  value = c(
    "End-to-end functional validation that dsFlower can consume dsImaging-resolved server-side NIfTI image assets and train a vision template without transferring image pixels to the researcher.",
    "This nine-image run is a technical path validation, not a clinical imaging benchmark.",
    evidence$output_dir
  ),
  stringsAsFactors = FALSE
)
knitr::kable(scope)
```

| item | value |
|:---|:---|
| Validated claim | End-to-end functional validation that dsFlower can consume dsImaging-resolved server-side NIfTI image assets and train a vision template without transferring image pixels to the researcher. |
| Explicit non-claim | This nine-image run is a technical path validation, not a clinical imaging benchmark. |
| Recorded output dir | /Users/david/Documents/GitHub/dsFlower-framework/dsflower_output/direct_dsimaging_images/20260511_140204 |
