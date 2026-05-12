# Vision Validation: pytorch_unet2d

This vignette records a live validation of a dsFlower vision template.
Synthetic PNG images were split across three Opal/Rock nodes, trained
federatively through DataSHIELD, and compared with a centralized PyTorch
baseline trained on the pooled synthetic cohort. This is a path
validation for image staging, template execution, result persistence,
and cleanup; it is not a clinical imaging benchmark.

``` r

overview <- data.frame(
  Field = c(
    'method', 'task', 'target', 'n_total', 'image_size', 'rounds',
    'centralized_metric', 'centralized_loss', 'centralized_accuracy',
    'centralized_dice', 'federated_status', 'federated_loss',
    'federated_n_failures', 'delta_loss',
    'acceptance_max_loss_ratio', 'acceptance_max_loss_margin',
    'acceptable_loss', 'validation_status'
  ),
  Value = c(
    row$method, row$task, row$target, row$n_total, validation$image_size, row$rounds,
    row$centralized_metric, fmt(row$centralized_loss), fmt(row$centralized_accuracy),
    fmt(row$centralized_dice), row$federated_status, fmt(row$federated_loss),
    fmt(row$federated_n_failures), fmt(row$delta_loss),
    fmt(row$acceptance_max_loss_ratio), fmt(row$acceptance_max_loss_margin),
    row$acceptable_loss, row$validation_status
  )
)
knitr::kable(overview)
```

| Field                      | Value          |
|:---------------------------|:---------------|
| method                     | pytorch_unet2d |
| task                       | segmentation   |
| target                     | mask_path      |
| n_total                    | 12             |
| image_size                 | 64             |
| rounds                     | 1              |
| centralized_metric         | dice_bce_loss  |
| centralized_loss           | 1.3099         |
| centralized_accuracy       | NA             |
| centralized_dice           | 0.0000         |
| federated_status           | ok             |
| federated_loss             | 1.4397         |
| federated_n_failures       | 0.0000         |
| delta_loss                 | 0.1298         |
| acceptance_max_loss_ratio  | 2.5000         |
| acceptance_max_loss_margin | 0.2500         |
| acceptable_loss            | TRUE           |
| validation_status          | pass           |

``` r

loss_df <- data.frame(
  mode = c('Centralized', 'Federated'),
  loss = c(row$centralized_loss, row$federated_loss)
)
loss_df <- loss_df[is.finite(loss_df$loss), , drop = FALSE]
if (nrow(loss_df) > 0 && requireNamespace('ggplot2', quietly = TRUE)) {
  ggplot2::ggplot(loss_df, ggplot2::aes(x = mode, y = loss, fill = mode)) +
    ggplot2::geom_col(width = 0.65, show.legend = FALSE) +
    ggplot2::labs(x = NULL, y = row$centralized_metric[[1]], title = method_id) +
    ggplot2::theme_minimal(base_size = 11)
} else {
  plot.new()
  text(0.5, 0.5, 'No federated loss available for this method')
}
```

![Bar chart comparing centralized and federated validation loss for
pytorch_unet2d.](validation-vision-pytorch-unet2d_files/figure-html/unnamed-chunk-3-1.png)

To reproduce only this method against configured Opal servers:

``` sh
DSFLOWER_VALIDATE_VISION_METHODS=pytorch_unet2d Rscript inst/demos/validate_vision_methods.R
```

``` r

if (nzchar(row$error)) cat(row$error)
```
