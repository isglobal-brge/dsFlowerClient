# Tests for R/metrics.R — Result Comparison

test_that("ds.flower.compare combines runs", {
  r1 <- dsFlowerClient:::dsflower_result(
    per_site = list(srv1 = data.frame(
      round = 1:2, metric = "loss", value = c(0.6, 0.4),
      stringsAsFactors = FALSE
    ))
  )
  r2 <- dsFlowerClient:::dsflower_result(
    per_site = list(srv1 = data.frame(
      round = 1:2, metric = "loss", value = c(0.7, 0.3),
      stringsAsFactors = FALSE
    ))
  )

  comparison <- ds.flower.compare(baseline = r1, experiment = r2)
  expect_s3_class(comparison, "data.frame")
  expect_true("run" %in% names(comparison))
  expect_equal(sort(unique(comparison$run)), c("baseline", "experiment"))
})

test_that("ds.flower.compare errors with no arguments", {
  expect_error(ds.flower.compare(), "At least one")
})

test_that("private metric catalog exposes only task-compatible scalar scores", {
  result <- structure(list(
    available = TRUE,
    task = "binary",
    metrics = list(
      n = 24,
      accuracy = 0.75,
      roc_auc = 0.81,
      brier = 0.18,
      sensitivity = NULL,
      roc = list(fpr = c(0, 1), tpr = c(0, 1))
    )
  ), class = "dsflower_validation")

  catalog <- ds.flower.metrics(result)
  expect_named(catalog, c(
    "task", "metric", "direction", "available", "value"
  ))
  expect_true(all(c(
    "accuracy", "sensitivity", "roc_auc", "brier"
  ) %in% catalog$metric))
  expect_false(any(c("n", "roc") %in% catalog$metric))
  expect_identical(
    catalog$direction[match("roc_auc", catalog$metric)], "maximize")
  expect_identical(
    catalog$direction[match("brier", catalog$metric)], "minimize")
  expect_true(catalog$available[match("accuracy", catalog$metric)])
  expect_false(catalog$available[match("sensitivity", catalog$metric)])
  expect_equal(catalog$value[match("roc_auc", catalog$metric)], 0.81)
  expect_true(is.na(catalog$value[match("sensitivity", catalog$metric)]))
})

test_that("score extracts one finite metric and rejects unsuitable objectives", {
  binary <- structure(list(
    available = TRUE,
    task = "binary",
    metrics = list(accuracy = 0.75, roc_auc = 0.81, sensitivity = NULL)
  ), class = "dsflower_validation")
  expect_identical(ds.flower.score(binary, "roc_auc"), 0.81)
  expect_identical(
    ds.flower.metric_direction("roc_auc", "binary"), "maximize")
  expect_identical(
    ds.flower.metric_direction("rmse", "regression"), "minimize")
  expect_error(ds.flower.score(binary, "sensitivity"), "unavailable")
  expect_error(ds.flower.score(binary, "roc"), "not a scoreable")
  expect_error(
    ds.flower.metric_direction("macro_f1", "binary"),
    "not a scoreable")

  binary$metrics$accuracy <- Inf
  expect_error(ds.flower.metrics(binary), "malformed")
  binary$metrics$accuracy <- c(0.7, 0.8)
  expect_error(ds.flower.score(binary, "accuracy"), "malformed")

  binary$metrics <- structure(
    list(accuracy = 0.7, accuracy = 0.8),
    names = c("accuracy", "accuracy"))
  expect_error(ds.flower.metrics(binary), "malformed")
})

test_that("metric helpers cover CV, atomic holdout, and unavailable releases", {
  cv <- structure(list(
    task = "regression",
    metrics = list(mae = 0.4, rmse = 0.6, r_squared = NULL)
  ), class = "dsflower_cv")
  expect_identical(ds.flower.score(cv, "mae"), 0.4)
  expect_true(ds.flower.metrics(cv)$available[
    match("rmse", ds.flower.metrics(cv)$metric)])

  run <- structure(list(
    available = TRUE,
    holdout_task = "multilabel",
    holdout = list(macro_f1 = 0.7, macro_roc_auc = 0.8)
  ), class = "dsflower_run")
  expect_identical(ds.flower.score(run, "macro_f1"), 0.7)
  run$available <- FALSE
  expect_error(ds.flower.metrics(run), "malformed")

  unavailable <- structure(list(
    available = FALSE, task = "count", metrics = NULL
  ), class = "dsflower_validation")
  catalog <- ds.flower.metrics(unavailable)
  expect_false(any(catalog$available))
  expect_true(all(is.na(catalog$value)))
  expect_error(ds.flower.score(unavailable, "mae"), "unavailable")
  unavailable$metrics <- list(mae = 0.1)
  expect_error(ds.flower.metrics(unavailable), "malformed")

  numeric_cv <- structure(list(
    task = "regression",
    metrics = list(mae = 1L, r_squared = -0.25)
  ), class = "dsflower_cv")
  expect_identical(ds.flower.score(numeric_cv, "mae"), 1)
  expect_identical(ds.flower.score(numeric_cv, "r_squared"), -0.25)

  expect_error(
    ds.flower.metrics(structure(list(), class = "dsflower_run")),
    "pooled holdout")
  expect_error(ds.flower.metrics(list(metrics = list())), "must be")
  expect_error(ds.flower.metric_direction("accuracy", "image"), "task")
  expect_error(ds.flower.metric_direction(NA_character_, "binary"), "metric")
  expect_error(ds.flower.metric_direction(c("accuracy", "f1"), "binary"),
               "metric")
})

test_that("all validation tasks have an explicit score direction", {
  expected <- list(
    binary = c(
      accuracy = "maximize", sensitivity = "maximize",
      specificity = "maximize", precision = "maximize",
      negative_predictive_value = "maximize", f1 = "maximize",
      balanced_accuracy = "maximize", roc_auc = "maximize",
      pr_auc = "maximize", brier = "minimize",
      expected_calibration_error = "minimize"),
    multiclass = c(
      accuracy = "maximize", balanced_accuracy = "maximize",
      macro_precision = "maximize", macro_recall = "maximize",
      macro_f1 = "maximize", macro_roc_auc = "maximize"),
    ordinal = c(
      accuracy = "maximize", balanced_accuracy = "maximize",
      macro_precision = "maximize", macro_recall = "maximize",
      macro_f1 = "maximize", macro_roc_auc = "maximize",
      ordinal_mae = "minimize"),
    multilabel = c(
      macro_roc_auc = "maximize", macro_f1 = "maximize"),
    regression = c(
      mae = "minimize", mse = "minimize", rmse = "minimize",
      r_squared = "maximize"),
    count = c(
      mae = "minimize", mse = "minimize", rmse = "minimize",
      r_squared = "maximize",
      mean_poisson_deviance_normalized = "minimize")
  )
  for (task in names(expected)) {
    expect_identical(
      dsFlowerClient:::.DSFLOWER_SCORE_METRICS[[task]], expected[[task]])
    for (metric in names(expected[[task]])) {
      expect_identical(
        ds.flower.metric_direction(metric, task),
        unname(expected[[task]][[metric]]))
    }
  }
})
