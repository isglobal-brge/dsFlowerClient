# All Method Validation Overview

This macro-vignette summarizes the live dsFlower validation suite across
tabular, sequence, survival, XGBoost, image-classification, and
segmentation templates. Each method has its own vignette, and each
federated run is compared with a centralized baseline trained on the
corresponding pooled synthetic cohort.

The suite is deliberately a functional and numerical sanity check, not a
privacy claim and not a clinical benchmark. It uses `sandbox_open` so
that per-node diagnostics can be observed during development. Production
studies should use the stricter DataSHIELD privacy profiles. XGBoost is
validated in this trusted demo mode; profiles that require Secure
Aggregation still enforce SecAgg before execution.

``` r

summary_table <- data.frame(
  suite = c('tabular_sequence_survival', 'vision'),
  generated_at = c(method_validation$generated_at, if (!is.null(vision_validation)) vision_validation$generated_at else NA),
  privacy_profile = c(method_validation$privacy_profile, if (!is.null(vision_validation)) vision_validation$privacy_profile else NA),
  n_sites = c(method_validation$n_sites, if (!is.null(vision_validation)) vision_validation$n_sites else NA),
  n_total = c(method_validation$n_total, if (!is.null(vision_validation)) vision_validation$n_total else NA),
  secagg_supported = c(method_validation$secagg_supported, NA),
  stringsAsFactors = FALSE
)
knitr::kable(summary_table)
```

| suite | generated_at | privacy_profile | n_sites | n_total | secagg_supported |
|:---|:---|:---|---:|---:|:---|
| tabular_sequence_survival | 2026-05-12T16:48:18+0200 | sandbox_open | 3 | 270 | FALSE |
| vision | 2026-05-12T14:54:38.270Z | sandbox_open | 3 | 12 | NA |

``` r

knitr::kable(results)
```

| suite | method | task | centralized_metric | centralized_loss | federated_status | federated_loss | delta_loss | acceptable_loss | validation_status |
|:---|:---|:---|:---|---:|:---|---:|---:|:---|:---|
| tabular_sequence_survival | sklearn_logreg | classification | log_loss | 0.2051486 | ok | 0.2334913 | 0.0283427 | TRUE | pass |
| tabular_sequence_survival | sklearn_ridge | classification | ridge_decision_mse | 0.4977802 | ok | 0.5091765 | 0.0113963 | TRUE | pass |
| tabular_sequence_survival | sklearn_sgd | classification | log_loss | 0.1955451 | ok | 0.2008949 | 0.0053498 | TRUE | pass |
| tabular_sequence_survival | sklearn_svm | classification | hinge_loss | 0.8285259 | ok | 1.8000502 | 0.9715243 | TRUE | pass |
| tabular_sequence_survival | sklearn_elastic_net | classification | log_loss | 0.1964865 | ok | 0.2000316 | 0.0035451 | TRUE | pass |
| tabular_sequence_survival | pytorch_logreg | classification | binary_cross_entropy | 0.6188036 | ok | 0.6002623 | -0.0185413 | TRUE | pass |
| tabular_sequence_survival | pytorch_mlp | classification | binary_cross_entropy | 0.7006344 | ok | 0.6847426 | -0.0158919 | TRUE | pass |
| tabular_sequence_survival | pytorch_linear_regression | regression | mse | 3.2117758 | ok | 3.5989498 | 0.3871740 | TRUE | pass |
| tabular_sequence_survival | pytorch_multiclass | classification | cross_entropy | 0.9693452 | ok | 0.9260484 | -0.0432968 | TRUE | pass |
| tabular_sequence_survival | pytorch_multilabel | classification | multilabel_bce | 0.5932636 | ok | 0.6415418 | 0.0482783 | TRUE | pass |
| tabular_sequence_survival | pytorch_poisson | regression | poisson_nll | 0.8903766 | ok | 1.0228395 | 0.1324629 | TRUE | pass |
| tabular_sequence_survival | pytorch_coxph | survival | cox_partial_likelihood | 4.8610749 | ok | 2.6362184 | -2.2248566 | TRUE | pass |
| tabular_sequence_survival | pytorch_lognormal_aft | survival | lognormal_aft_nll | 3.4966099 | ok | 3.6048540 | 0.1082441 | TRUE | pass |
| tabular_sequence_survival | pytorch_cause_specific_cox | survival | cause_specific_cox_loss | 4.6180305 | ok | 2.4723849 | -2.1456457 | TRUE | pass |
| tabular_sequence_survival | pytorch_lstm | classification | binary_cross_entropy | 0.6957106 | ok | 0.6933520 | -0.0023586 | TRUE | pass |
| tabular_sequence_survival | pytorch_tcn | classification | binary_cross_entropy | 0.6658388 | ok | 0.6879575 | 0.0221187 | TRUE | pass |
| tabular_sequence_survival | xgboost | classification | log_loss | 0.4348304 | ok | 0.6931472 | 0.2583168 | TRUE | pass |
| vision | pytorch_resnet18 | classification | cross_entropy | 0.5605000 | ok | 0.6322000 | 0.0717000 | TRUE | pass |
| vision | pytorch_densenet121 | classification | cross_entropy | 0.6033000 | ok | 0.8186000 | 0.2152000 | TRUE | pass |
| vision | pytorch_unet2d | segmentation | dice_bce_loss | 1.3099000 | ok | 1.3597000 | 0.0499000 | TRUE | pass |

``` r

plot_df <- results[is.finite(results$delta_loss), , drop = FALSE]
if (nrow(plot_df) > 0 && requireNamespace('ggplot2', quietly = TRUE)) {
  plot_df$label <- paste(plot_df$suite, plot_df$method, sep = ': ')
  ggplot2::ggplot(plot_df, ggplot2::aes(x = stats::reorder(label, delta_loss), y = delta_loss, fill = task)) +
    ggplot2::geom_hline(yintercept = 0, linewidth = 0.3, color = 'grey40') +
    ggplot2::geom_col(width = 0.7) +
    ggplot2::coord_flip() +
    ggplot2::labs(x = NULL, y = 'Federated loss - centralized loss') +
    ggplot2::theme_minimal(base_size = 11)
}
```

![Horizontal bar chart of federated minus centralized loss by
method.](validation-overview_files/figure-html/unnamed-chunk-4-1.png)

Per-method vignettes:

- [sklearn_logreg](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-sklearn-logreg.md)
- [sklearn_ridge](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-sklearn-ridge.md)
- [sklearn_sgd](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-sklearn-sgd.md)
- [sklearn_svm](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-sklearn-svm.md)
- [sklearn_elastic_net](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-sklearn-elastic-net.md)
- [pytorch_logreg](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-pytorch-logreg.md)
- [pytorch_mlp](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-pytorch-mlp.md)
- [pytorch_linear_regression](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-pytorch-linear-regression.md)
- [pytorch_multiclass](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-pytorch-multiclass.md)
- [pytorch_multilabel](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-pytorch-multilabel.md)
- [pytorch_poisson](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-pytorch-poisson.md)
- [pytorch_coxph](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-pytorch-coxph.md)
- [pytorch_lognormal_aft](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-pytorch-lognormal-aft.md)
- [pytorch_cause_specific_cox](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-pytorch-cause-specific-cox.md)
- [pytorch_lstm](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-pytorch-lstm.md)
- [pytorch_tcn](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-pytorch-tcn.md)
- [xgboost](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-xgboost.md)

Vision method vignettes:

- [pytorch_resnet18](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-vision-pytorch-resnet18.md)
- [pytorch_densenet121](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-vision-pytorch-densenet121.md)
- [pytorch_unet2d](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-vision-pytorch-unet2d.md)

The vision-specific overview remains available at [Vision Method
Validation
Overview](https://isglobal-brge.github.io/dsFlowerClient/articles/validation-vision-overview.md).

The full suite is produced by repeating the inline
Opal/DataSHIELD/Flower workflow shown in each per-method vignette for
every method listed above. The persisted JSON artifacts are the audit
trail consumed by this pkgdown site.
