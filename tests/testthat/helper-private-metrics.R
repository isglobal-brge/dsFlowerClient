.test_binary_metrics <- function(bins = 4L) {
  list(
    n = 6, accuracy = 0.75, sensitivity = 0.8, specificity = 0.7,
    precision = 0.8, negative_predictive_value = 0.7, f1 = 0.8,
    balanced_accuracy = 0.75, roc_auc = 0.81, pr_auc = 0.79,
    brier = 0.18, expected_calibration_error = 0.12,
    roc = list(
      fpr = seq(0, 1, length.out = bins + 2L),
      tpr = seq(0, 1, length.out = bins + 2L)),
    precision_recall = list(
      recall = seq(0, 1, length.out = bins + 1L),
      precision = seq(1, 0, length.out = bins + 1L)),
    calibration = list(
      predicted = seq(0.1, 0.9, length.out = bins),
      observed = seq(0, 1, length.out = bins),
      weight = rep(1, bins)),
    decision_curve = list(
      threshold = seq(0.1, 0.9, length.out = bins),
      net_benefit = rep(0.1, bins),
      treat_all = rep(0, bins),
      treat_none = rep(0, bins)))
}

.test_multiclass_metrics <- function(task = "multiclass",
                                     null_primary = FALSE) {
  metrics <- list(
    n = 0, accuracy = if (isTRUE(null_primary)) NULL else 0.75,
    balanced_accuracy = 0, macro_precision = 0, macro_recall = 0,
    macro_f1 = 0, macro_roc_auc = NULL,
    confusion_matrix = matrix(0, nrow = 3L, ncol = 3L))
  if (identical(task, "ordinal")) metrics["ordinal_mae"] <- list(NULL)
  metrics
}

.test_multilabel_metrics <- function(null_primary = FALSE) {
  labels <- list(.test_binary_metrics(), .test_binary_metrics())
  if (isTRUE(null_primary)) {
    for (index in seq_along(labels)) {
      labels[[index]]["f1"] <- list(NULL)
      labels[[index]]["roc_auc"] <- list(NULL)
    }
  }
  list(
    labels = labels,
    macro_roc_auc = if (isTRUE(null_primary)) NULL else 0.81,
    macro_f1 = if (isTRUE(null_primary)) NULL else 0.8)
}

.test_numeric_metrics <- function(task = "regression") {
  metrics <- list(n = 3, mae = 0.4, mse = 0.36, rmse = 0.6,
                  r_squared = -0.25)
  if (identical(task, "count")) {
    metrics$mean_poisson_deviance_normalized <- 0.2
  }
  metrics
}

.test_private_metrics <- function(task, null_primary = FALSE) {
  switch(task,
    binary = .test_binary_metrics(),
    multiclass = .test_multiclass_metrics(task, null_primary),
    ordinal = .test_multiclass_metrics(task, null_primary),
    multilabel = .test_multilabel_metrics(null_primary),
    regression = .test_numeric_metrics(task),
    count = .test_numeric_metrics(task))
}
