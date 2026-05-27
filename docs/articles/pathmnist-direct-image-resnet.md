# PathMNIST Direct Image ResNet Benchmark

This vignette records a direct-image dsFlower benchmark on PathMNIST, a
MedMNIST v2 histopathology image dataset. The run uses a balanced binary
subset of adipose tissue versus colorectal adenocarcinoma epithelium,
split evenly across three Opal/Rock servers. The Opal table contains
metadata and relative image paths; image files are read inside each Rock
by the `pytorch_resnet18` template.

The benchmark has two branches. The `trusted_internal` branch compares a
federated ResNet-18 checkpoint with a centralized PyTorch baseline over
the same 1,500 images. The `consortium_internal` branch repeats the
federated execution under the profile that requires Secure Aggregation,
suppresses per-node metrics and still produces a global model
checkpoint.

## Reproduction Command

``` r

Sys.setenv(
  DSFLOWER_OPAL_URLS = "https://localhost:8443,https://localhost:8444,https://localhost:8445",
  OPAL_USER = "administrator",
  OPAL_PASSWORD = "admin123",
  DSFLOWER_PATHMNIST_N_PER_SITE = "500",
  DSFLOWER_PATHMNIST_PRIVACY_PROFILES = "trusted_internal,consortium_internal"
)
source(system.file("demos", "benchmark_pathmnist_direct_image.R",
                   package = "dsFlowerClient"))
```

## Dataset And Split

``` r

class_table <- data.frame(
  source_class = names(evidence$dataset$classes),
  binary_label = as.integer(unlist(evidence$dataset$binary_mapping)),
  class_name = unname(unlist(evidence$dataset$classes)),
  row.names = NULL
)
site_table <- evidence$site_counts

summary_table <- data.frame(
  field = c("dataset", "source", "sites", "rows per site",
            "total rows", "image size", "model", "rounds"),
  value = c(evidence$dataset$dataset,
            evidence$dataset$source,
            evidence$n_sites,
            evidence$n_per_site,
            evidence$n_total,
            paste0(evidence$image_size, " x ", evidence$image_size),
            evidence$model,
            evidence$rounds),
  stringsAsFactors = FALSE
)

knitr::kable(class_table)
```

| source_class | binary_label | class_name                           |
|:-------------|-------------:|:-------------------------------------|
| 0            |            0 | adipose                              |
| 8            |            1 | colorectal adenocarcinoma epithelium |

``` r

knitr::kable(site_table)
```

| site  | rows | label0 | label1 |
|:------|-----:|-------:|-------:|
| opal1 |  500 |    250 |    250 |
| opal2 |  500 |    250 |    250 |
| opal3 |  500 |    250 |    250 |

``` r

knitr::kable(summary_table)
```

| field         | value                   |
|:--------------|:------------------------|
| dataset       | PathMNIST               |
| source        | MedMNIST v2 train split |
| sites         | 3                       |
| rows per site | 500                     |
| total rows    | 1500                    |
| image size    | 32 x 32                 |
| model         | pytorch_resnet18        |
| rounds        | 2                       |

## Centralized Baseline

``` r

centralized <- data.frame(
  engine = "centralized PyTorch",
  samples = evidence$centralized$n_samples,
  epochs = evidence$centralized$epochs,
  loss = evidence$centralized$eval$loss,
  accuracy = evidence$centralized$eval$accuracy,
  stringsAsFactors = FALSE
)
knitr::kable(centralized, digits = 4)
```

| engine              | samples | epochs |   loss | accuracy |
|:--------------------|--------:|-------:|-------:|---------:|
| centralized PyTorch |    1500 |      2 | 0.0143 |   0.9947 |

``` r


cat("[centralized] epoch losses:",
    paste(sprintf("%.6f", evidence$centralized$train_loss_by_epoch),
          collapse = " -> "), "\n")
#> [centralized] epoch losses: 0.174467 -> 0.045983
cat("[centralized] eval loss:",
    sprintf("%.6f", evidence$centralized$eval$loss), "\n")
#> [centralized] eval loss: 0.014301
cat("[centralized] eval accuracy:",
    sprintf("%.4f", evidence$centralized$eval$accuracy), "\n")
#> [centralized] eval accuracy: 0.9947
```

## Federated Trusted Branch

``` r

trusted_profile <- evidence$profile_results$trusted_internal
trusted_history <- trusted_profile$history
comparison <- evidence$trusted_internal_comparison

knitr::kable(trusted_history, digits = 4)
```

| round |   loss | n_clients | n_failures |
|------:|-------:|----------:|-----------:|
|     1 | 0.1484 |         3 |          0 |
|     2 | 0.0535 |         3 |          0 |

``` r


trusted_table <- data.frame(
  run = c("centralized", "federated checkpoint"),
  loss = c(comparison$centralized_loss,
           comparison$federated_checkpoint_loss),
  accuracy = c(comparison$centralized_accuracy,
               comparison$federated_checkpoint_accuracy),
  failures = c(NA_integer_, comparison$federated_failures),
  stringsAsFactors = FALSE
)
knitr::kable(trusted_table, digits = 4)
```

| run                  |   loss | accuracy | failures |
|:---------------------|-------:|---------:|---------:|
| centralized          | 0.0143 |   0.9947 |       NA |
| federated checkpoint | 0.0535 |   0.9860 |        0 |

``` r


cat("[federated] final history loss:",
    sprintf("%.6f", comparison$federated_history_loss), "\n")
#> [federated] final history loss: 0.053542
cat("[delta] checkpoint loss - centralized loss:",
    sprintf("%.6f", comparison$delta_loss), "\n")
#> [delta] checkpoint loss - centralized loss: 0.039241
cat("[delta] checkpoint accuracy - centralized accuracy:",
    sprintf("%.4f", comparison$delta_accuracy), "\n")
#> [delta] checkpoint accuracy - centralized accuracy: -0.0087
cat("[validation] status:", comparison$validation_status, "\n")
#> [validation] status: pass
```

``` r

plot_df <- data.frame(
  run = rep(c("centralized", "federated"), 2),
  metric = rep(c("loss", "accuracy"), each = 2),
  value = c(comparison$centralized_loss,
            comparison$federated_checkpoint_loss,
            comparison$centralized_accuracy,
            comparison$federated_checkpoint_accuracy),
  stringsAsFactors = FALSE
)
if (requireNamespace("ggplot2", quietly = TRUE)) {
  ggplot2::ggplot(plot_df, ggplot2::aes(x = run, y = value, fill = run)) +
    ggplot2::geom_col(width = 0.62, show.legend = FALSE) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.4f", value)),
                       vjust = -0.35, size = 3.3) +
    ggplot2::facet_wrap(~ metric, scales = "free_y") +
    ggplot2::scale_fill_manual(values = c("#4C78A8", "#F58518")) +
    ggplot2::labs(x = NULL, y = NULL) +
    ggplot2::theme_minimal(base_size = 11)
} else {
  old <- par(mfrow = c(1, 2))
  on.exit(par(old), add = TRUE)
  barplot(plot_df$value[plot_df$metric == "loss"],
          names.arg = c("central", "fed"), ylab = "loss",
          col = c("#4C78A8", "#F58518"))
  barplot(plot_df$value[plot_df$metric == "accuracy"],
          names.arg = c("central", "fed"), ylab = "accuracy",
          col = c("#4C78A8", "#F58518"))
}
```

![Bar chart comparing centralized and federated PathMNIST ResNet loss
and
accuracy.](pathmnist-direct-image-resnet_files/figure-html/trusted-plot-1.png)

## Secure Aggregation Branch

``` r

secagg <- evidence$consortium_internal_secagg
secagg_table <- data.frame(
  field = c("privacy profile", "secure aggregation required",
            "metric release", "federated failures",
            "global checkpoint produced", "validation status"),
  value = c("consortium_internal",
            as.character(secagg$secure_aggregation_required),
            secagg$metric_release,
            secagg$federated_failures,
            as.character(secagg$saved_model_exists),
            secagg$validation_status),
  stringsAsFactors = FALSE
)
knitr::kable(secagg_table)
```

| field                       | value                                     |
|:----------------------------|:------------------------------------------|
| privacy profile             | consortium_internal                       |
| secure aggregation required | TRUE                                      |
| metric release              | suppressed by consortium_internal profile |
| federated failures          | 0                                         |
| global checkpoint produced  | TRUE                                      |
| validation status           | pass                                      |

The SecAgg branch is not interpreted through its loss value because
`consortium_internal` suppresses released per-node metrics. The
validation criteria for this branch are run completion, zero federated
client failures, activation of the Secure Aggregation requirement and
production of the global model artefact.

## Evidence Scope

``` r

scope <- data.frame(
  claim = c("Direct-image learning path",
            "Centralized-versus-federated utility",
            "SecAgg profile execution"),
  evidence = c(
    "1,500 PathMNIST PNG images remain file-backed on the server side while Opal tables expose only metadata and relative paths.",
    "The federated ResNet-18 checkpoint reaches 0.9860 accuracy versus 0.9947 for the centralized PyTorch baseline on the same image table.",
    "The consortium profile completes with zero client failures, metric suppression and a persisted global checkpoint."
  ),
  stringsAsFactors = FALSE
)
knitr::kable(scope)
```

| claim | evidence |
|:---|:---|
| Direct-image learning path | 1,500 PathMNIST PNG images remain file-backed on the server side while Opal tables expose only metadata and relative paths. |
| Centralized-versus-federated utility | The federated ResNet-18 checkpoint reaches 0.9860 accuracy versus 0.9947 for the centralized PyTorch baseline on the same image table. |
| SecAgg profile execution | The consortium profile completes with zero client failures, metric suppression and a persisted global checkpoint. |
