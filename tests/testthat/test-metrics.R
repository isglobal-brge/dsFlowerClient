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
  metrics <- .test_private_metrics("binary")
  metrics["sensitivity"] <- list(NULL)
  result <- structure(list(
    available = TRUE,
    task = "binary",
    metrics = metrics
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
  metrics <- .test_private_metrics("binary")
  metrics["sensitivity"] <- list(NULL)
  binary <- structure(list(
    available = TRUE,
    task = "binary",
    metrics = metrics
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
  regression <- .test_private_metrics("regression")
  regression["r_squared"] <- list(NULL)
  cv <- structure(list(
    task = "regression",
    metrics = regression
  ), class = "dsflower_cv")
  expect_identical(ds.flower.score(cv, "mae"), 0.4)
  expect_true(ds.flower.metrics(cv)$available[
    match("rmse", ds.flower.metrics(cv)$metric)])

  run <- structure(list(
    available = TRUE,
    holdout_task = "multilabel",
    holdout = .test_private_metrics("multilabel")
  ), class = "dsflower_run")
  expect_identical(ds.flower.score(run, "macro_f1"), 0.8)
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
    metrics = utils::modifyList(
      .test_private_metrics("regression"), list(mae = 1L))
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

test_that("private metric schemas reject impossible values and extra trees", {
  for (task in c(
      "binary", "multiclass", "ordinal", "multilabel",
      "regression", "count")) {
    valid <- structure(list(
      available = TRUE, task = task,
      metrics = .test_private_metrics(task)
    ), class = "dsflower_validation")
    expect_no_error(ds.flower.metrics(valid))

    extra <- valid
    extra$metrics$diagnostics <- list(value = 1)
    expect_error(ds.flower.metrics(extra), "malformed", info = task)
  }

  impossible <- structure(list(
    available = TRUE, task = "binary",
    metrics = .test_private_metrics("binary")
  ), class = "dsflower_validation")
  impossible$metrics$roc_auc <- 9
  expect_error(ds.flower.score(impossible, "roc_auc"), "malformed")

  impossible <- structure(list(
    task = "multiclass", metrics = .test_private_metrics("multiclass")
  ), class = "dsflower_cv")
  impossible$metrics$confusion_matrix[[1L]] <- -1
  expect_error(ds.flower.metrics(impossible), "malformed")

  impossible <- structure(list(
    available = TRUE, holdout_task = "count",
    holdout = .test_private_metrics("count")
  ), class = "dsflower_run")
  impossible$holdout$mean_poisson_deviance_normalized <- 1.1
  expect_error(ds.flower.metrics(impossible), "malformed")
})

test_that("private metric schemas pin shapes, ranges, and task nullability", {
  valid <- function(metrics, task) {
    dsFlowerClient:::.private_metrics_valid(metrics, task)
  }

  binary <- .test_private_metrics("binary")
  nullable_binary <- binary
  nullable_binary["roc_auc"] <- list(NULL)
  expect_true(valid(nullable_binary, "binary"))
  invalid <- binary
  invalid["accuracy"] <- list(NULL)
  expect_false(valid(invalid, "binary"))
  invalid <- binary
  invalid$roc$fpr <- invalid$roc$fpr[-1L]
  expect_false(valid(invalid, "binary"))
  invalid <- binary
  invalid$roc$extra <- 0
  expect_false(valid(invalid, "binary"))

  multiclass <- .test_private_metrics("multiclass", null_primary = TRUE)
  expect_true(valid(multiclass, "multiclass"))
  invalid <- multiclass
  invalid["macro_f1"] <- list(NULL)
  expect_false(valid(invalid, "multiclass"))
  invalid <- multiclass
  invalid$confusion_matrix <- matrix(0, nrow = 2L, ncol = 3L)
  expect_false(valid(invalid, "multiclass"))

  ordinal <- .test_private_metrics("ordinal", null_primary = TRUE)
  expect_true(valid(ordinal, "ordinal"))
  invalid <- ordinal
  invalid$ordinal_mae <- 3
  expect_false(valid(invalid, "ordinal"))

  multilabel <- .test_private_metrics("multilabel", null_primary = TRUE)
  expect_true(valid(multilabel, "multilabel"))
  invalid <- multilabel
  invalid$labels <- invalid$labels[1L]
  expect_false(valid(invalid, "multilabel"))
  invalid <- multilabel
  invalid$labels[[2L]] <- .test_binary_metrics(5L)
  expect_false(valid(invalid, "multilabel"))

  regression <- .test_private_metrics("regression")
  regression["r_squared"] <- list(NULL)
  expect_true(valid(regression, "regression"))
  invalid <- regression
  invalid["mae"] <- list(NULL)
  expect_false(valid(invalid, "regression"))

  count <- .test_private_metrics("count")
  count["mean_poisson_deviance_normalized"] <- list(NULL)
  expect_true(valid(count, "count"))
  invalid <- count
  invalid$mean_poisson_deviance_normalized <- 1.1
  expect_false(valid(invalid, "count"))
})

test_that("JSON-list metric arrays retain their exact scalar contract", {
  metrics <- .test_binary_metrics()
  groups <- c("roc", "precision_recall", "calibration", "decision_curve")
  for (group in groups) {
    metrics[[group]] <- lapply(metrics[[group]], as.list)
  }
  metrics$calibration$predicted[[1L]] <- 0L
  expect_true(dsFlowerClient:::.private_metrics_valid(metrics, "binary"))

  values <- metrics$calibration$predicted
  adversarial <- list(
    logical = replace(values, 2L, list(TRUE)),
    character = replace(values, 2L, list("0.5")),
    nested = replace(values, 2L, list(list(0.5))),
    named = replace(values, 2L, list(structure(0.5, names = "x"))),
    dimensioned = replace(values, 2L, list(matrix(0.5))),
    multi_value = replace(values, 2L, list(c(0.4, 0.5))),
    null = replace(values, 2L, list(NULL)),
    non_finite = replace(values, 2L, list(Inf))
  )
  for (case in names(adversarial)) {
    invalid <- metrics
    invalid$calibration$predicted <- adversarial[[case]]
    expect_false(
      dsFlowerClient:::.private_metrics_valid(invalid, "binary"),
      info = case)
  }
})

test_that("noise-only nullable primary metrics remain valid", {
  for (task in c("multiclass", "ordinal", "multilabel")) {
    metrics <- .test_private_metrics(task, null_primary = TRUE)
    result <- structure(list(
      available = TRUE, task = task, metrics = metrics
    ), class = "dsflower_validation")
    catalog <- ds.flower.metrics(result)
    primary <- if (identical(task, "multilabel")) "macro_f1" else "accuracy"
    expect_false(catalog$available[match(primary, catalog$metric)], info = task)
    expect_error(ds.flower.score(result, primary), "unavailable", info = task)
  }
})

test_that("all private readers accept task-valid noise-only primary NULL", {
  for (task in c("multiclass", "ordinal", "multilabel")) {
    root <- withr::local_tempdir()
    metrics <- .test_private_metrics(task, null_primary = TRUE)
    write_release <- function(value, name) jsonlite::write_json(
      value, file.path(root, name), auto_unbox = TRUE, null = "null")

    write_release(list(
      pooled_only = TRUE, privacy = "node-dp-pooled-postprocessing",
      task = task, n_nodes = 2L, available = TRUE, metrics = metrics
    ), "validation.json")
    expect_no_error(dsFlowerClient:::.read_private_validation_result(root))

    write_release(list(
      pooled_only = TRUE, privacy = "node-dp-pooled-postprocessing",
      method = "holdout", task = task, n_nodes = 2L, metrics = metrics
    ), "holdout.json")
    expect_no_error(dsFlowerClient:::.read_holdout_result(root))

    write_release(list(
      pooled_only = TRUE, privacy = "node-dp-pooled-postprocessing",
      method = "cross_validation", task = task, n_nodes = 2L, folds = 3L,
      cv_contract_sha256 = strrep("a", 64L),
      cv_job_sha256 = strrep("b", 64L), metrics = metrics
    ), "cv.json")
    expect_no_error(dsFlowerClient:::.read_cross_validation_result(root))
  }
})

test_that("all private readers reject impossible and extra metric payloads", {
  root <- withr::local_tempdir()
  metrics <- .test_private_metrics("binary")
  metrics$roc_auc <- 9
  metrics$diagnostics <- list(value = 1)
  write_release <- function(value, name) jsonlite::write_json(
    value, file.path(root, name), auto_unbox = TRUE, null = "null")

  write_release(list(
    pooled_only = TRUE, privacy = "node-dp-pooled-postprocessing",
    task = "binary", n_nodes = 2L, available = TRUE, metrics = metrics
  ), "validation.json")
  expect_error(
    dsFlowerClient:::.read_private_validation_result(root), "pooled-only")

  write_release(list(
    pooled_only = TRUE, privacy = "node-dp-pooled-postprocessing",
    method = "holdout", task = "binary", n_nodes = 2L, metrics = metrics
  ), "holdout.json")
  expect_error(dsFlowerClient:::.read_holdout_result(root), "pooled-only")

  write_release(list(
    pooled_only = TRUE, privacy = "node-dp-pooled-postprocessing",
    method = "cross_validation", task = "binary", n_nodes = 2L, folds = 3L,
    cv_contract_sha256 = strrep("a", 64L),
    cv_job_sha256 = strrep("b", 64L), metrics = metrics
  ), "cv.json")
  expect_error(
    dsFlowerClient:::.read_cross_validation_result(root), "pooled-only")
})

test_that("private result readers require a scalar task string", {
  metrics <- .test_private_metrics("binary")
  for (task in list(list("binary"), list(x = "binary"))) {
    root <- withr::local_tempdir()
    write_release <- function(value, name) jsonlite::write_json(
      value, file.path(root, name), auto_unbox = TRUE, null = "null")

    write_release(list(
      pooled_only = TRUE, privacy = "node-dp-pooled-postprocessing",
      task = task, n_nodes = 2L, available = TRUE, metrics = metrics
    ), "validation.json")
    expect_error(
      dsFlowerClient:::.read_private_validation_result(root), "pooled-only")

    write_release(list(
      pooled_only = TRUE, privacy = "node-dp-pooled-postprocessing",
      method = "holdout", task = task, n_nodes = 2L, metrics = metrics
    ), "holdout.json")
    expect_error(dsFlowerClient:::.read_holdout_result(root), "pooled-only")

    write_release(list(
      pooled_only = TRUE, privacy = "node-dp-pooled-postprocessing",
      method = "cross_validation", task = task, n_nodes = 2L, folds = 3L,
      cv_contract_sha256 = strrep("a", 64L),
      cv_job_sha256 = strrep("b", 64L), metrics = metrics
    ), "cv.json")
    expect_error(
      dsFlowerClient:::.read_cross_validation_result(root), "pooled-only")
  }
})

test_that("private result readers enforce the public 160 MiB wire cap", {
  root <- withr::local_tempdir()
  paths <- file.path(root, c("validation.json", "holdout.json", "cv.json"))
  for (path in paths) writeBin(charToRaw("{}"), path)
  expect_identical(
    dsFlowerClient:::.PRIVATE_METRIC_RESULT_MAX_BYTES, 160 * 1024^2)
  local_mocked_bindings(
    .PRIVATE_METRIC_RESULT_MAX_BYTES = 1,
    .package = "dsFlowerClient")

  expect_error(
    dsFlowerClient:::.read_private_validation_result(root), "pooled-only")
  expect_error(dsFlowerClient:::.read_holdout_result(root), "pooled-only")
  expect_error(
    dsFlowerClient:::.read_cross_validation_result(root), "pooled-only")
  expect_false(dsFlowerClient:::.training_artifacts_complete(
    root, num_rounds = 1L, cross_validation = TRUE))
})
