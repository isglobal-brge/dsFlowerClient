# Vision Method Validation Overview

This macro-vignette summarizes the live dsFlower vision-template
validation. The experiment uses synthetic image and mask files staged on
the Rock servers, while metadata remains in Opal tables. Each federated
run is compared with a centralized PyTorch baseline trained over the
pooled files.

``` r

summary_table <- data.frame(
  Field = c('generated_at', 'privacy_profile', 'n_sites', 'n_per_site', 'n_total', 'image_size'),
  Value = c(validation$generated_at, validation$privacy_profile, validation$n_sites,
            validation$n_per_site, validation$n_total, validation$image_size)
)
knitr::kable(summary_table)
```

| Field           | Value                    |
|:----------------|:-------------------------|
| generated_at    | 2026-05-12T14:54:38.270Z |
| privacy_profile | sandbox_open             |
| n_sites         | 3                        |
| n_per_site      | 4                        |
| n_total         | 12                       |
| image_size      | 64                       |

``` r

display <- results[, c(
  'method', 'task', 'centralized_metric', 'centralized_loss',
  'federated_status', 'federated_loss', 'delta_loss',
  'acceptable_loss', 'validation_status'
)]
knitr::kable(display)
```

| method | task | centralized_metric | centralized_loss | federated_status | federated_loss | delta_loss | acceptable_loss | validation_status |
|:---|:---|:---|---:|:---|---:|---:|:---|:---|
| pytorch_resnet18 | classification | cross_entropy | 0.5605 | ok | 0.6322 | 0.0717 | TRUE | pass |
| pytorch_densenet121 | classification | cross_entropy | 0.6033 | ok | 0.8186 | 0.2152 | TRUE | pass |
| pytorch_unet2d | segmentation | dice_bce_loss | 1.3099 | ok | 1.3597 | 0.0499 | TRUE | pass |

``` r

plot_df <- results[is.finite(results$delta_loss), , drop = FALSE]
if (nrow(plot_df) > 0 && requireNamespace('ggplot2', quietly = TRUE)) {
  ggplot2::ggplot(plot_df, ggplot2::aes(x = stats::reorder(method, delta_loss), y = delta_loss, fill = task)) +
    ggplot2::geom_hline(yintercept = 0, linewidth = 0.3, color = 'grey40') +
    ggplot2::geom_col(width = 0.7) +
    ggplot2::coord_flip() +
    ggplot2::labs(x = NULL, y = 'Federated loss - centralized loss') +
    ggplot2::theme_minimal(base_size = 11)
}
```

![Horizontal bar chart of vision-template federated minus centralized
loss.](validation-vision-overview_files/figure-html/unnamed-chunk-4-1.png)

The complete vision suite is produced by repeating the inline
Opal/DataSHIELD/Flower workflow shown in each per-method vignette for
all vision templates listed above. The persisted JSON artifact is the
audit trail consumed by this pkgdown page.
