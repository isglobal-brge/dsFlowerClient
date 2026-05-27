`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || (length(x) == 1 && is.na(x))) y else x
}

test_that("clinical benchmark evidence contains only completed runs", {
  files <- c(
    "dsflower_clinical_algorithm_results.json",
    "dsflower_clinical_secagg_results.json",
    "dsflower_xgboost_histogram_privacy_results.json",
    "dsflower_xgboost_histogram_privacy_curve_results.json",
    "dsflower_clinical_dp_results.json",
    "dsflower_clinical_dp_curve_results.json"
  )

  for (file in files) {
    path <- system.file("extdata", file, package = "dsFlowerClient")
    expect_true(file.exists(path), info = file)
    evidence <- jsonlite::fromJSON(path, simplifyVector = FALSE)
    expect_true(length(evidence$results) > 0L, info = file)

    for (result in evidence$results) {
      expect_equal(result$status, "pass", info = file)
      expect_true(is.finite(result$central_metrics$auc), info = file)
      expect_true(is.finite(result$federated_metrics$auc), info = file)
      auc_limit <- if (identical(result$privacy_profile, "clinical_update_noise") &&
                       identical(result$model_id, "xgboost_histogram")) {
        0.12
      } else if (identical(result$model_id, "xgboost_histogram")) {
        0.05
      } else if (identical(result$privacy_profile, "high_sensitivity_dp")) {
        0.06
      } else if (identical(result$model_id, "sklearn_sgd")) {
        0.025
      } else {
        0.02
      }
      expect_true(abs(result$metric_delta$auc) <= auc_limit, info = file)

      failures <- sum(vapply(
        result$history %||% list(),
        function(round) round$n_failures %||% 0L,
        integer(1)
      ))
      expect_equal(failures, 0L, info = file)
    }
  }
})

test_that("clinical benchmark evidence records the enforced privacy policy", {
  files <- c(
    "dsflower_clinical_algorithm_results.json",
    "dsflower_clinical_secagg_results.json",
    "dsflower_xgboost_histogram_privacy_results.json",
    "dsflower_xgboost_histogram_privacy_curve_results.json",
    "dsflower_clinical_dp_results.json",
    "dsflower_clinical_dp_curve_results.json"
  )

  for (file in files) {
    path <- system.file("extdata", file, package = "dsFlowerClient")
    evidence <- jsonlite::fromJSON(path, simplifyVector = FALSE)
    policy <- evidence$privacy_policy
    expect_false(is.null(policy), info = file)
    expect_equal(policy$privacy_profile, evidence$privacy_profile, info = file)

    expected_secagg <- evidence$privacy_profile %in% c(
      "consortium_internal", "clinical_default", "clinical_hardened",
      "clinical_update_noise", "high_sensitivity_dp"
    )
    expected_dp <- evidence$privacy_profile %in% c(
      "clinical_update_noise", "high_sensitivity_dp"
    )
    expected_scope <- switch(evidence$privacy_profile,
      clinical_update_noise = "update_noise_only",
      high_sensitivity_dp = "patient_level_dp_sgd",
      "none"
    )

    expect_identical(policy$secure_aggregation_required, expected_secagg,
                     info = file)
    expect_identical(policy$dp_required, expected_dp, info = file)
    expect_equal(policy$dp_scope, expected_scope, info = file)
    if (expected_secagg) {
      expect_true(isTRUE(policy$secure_aggregation_supported), info = file)
      expect_false(policy$allow_per_node_metrics, info = file)
      expect_true(policy$fixed_client_sampling, info = file)
    }

    for (result in evidence$results) {
      expect_equal(result$privacy_policy$privacy_profile,
                   result$privacy_profile, info = file)
      expect_identical(result$secure_aggregation_required,
                       expected_secagg, info = file)
      expect_identical(result$dp_required, expected_dp, info = file)
      expect_equal(result$dp_scope, expected_scope, info = file)
      expect_equal(result$privacy_mechanism,
                   policy$privacy_mechanism, info = file)
    }
  }
})

test_that("clinical_default evidence uses datasets satisfying profile row policy", {
  path <- system.file(
    "extdata", "dsflower_clinical_secagg_results.json",
    package = "dsFlowerClient"
  )
  evidence <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  expect_equal(evidence$privacy_profile, "clinical_default")

  for (result in evidence$results) {
    expect_false(
      identical(result$dataset_id, "uci_heart_disease"),
      info = "Heart Disease is below the clinical_default per-site row policy"
    )
    site_n <- unlist(result$site_train_n, use.names = FALSE)
    if (identical(result$family, "linear classifier")) {
      expect_gte(min(site_n), 100)
    }
    if (identical(result$family, "neural network")) {
      expect_gte(min(site_n), 500)
    }
    if (identical(result$family, "tree ensemble")) {
      expect_gte(min(site_n), 200)
    }
  }
})

test_that("high_sensitivity_dp evidence uses DP-compatible data volume", {
  path <- system.file(
    "extdata", "dsflower_clinical_dp_results.json",
    package = "dsFlowerClient"
  )
  evidence <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  expect_equal(evidence$privacy_profile, "high_sensitivity_dp")

  for (result in evidence$results) {
    expect_equal(result$dataset_id, "cdc_diabetes_health_indicators")
    expect_equal(result$family, "neural network")
    expect_gte(min(unlist(result$site_train_n, use.names = FALSE)), 1000)
    expect_equal(result$privacy_parameters$epsilon, 8)
    expect_equal(result$privacy_parameters$delta, 1e-5)
    expect_equal(result$privacy_parameters$clipping_norm, 1)
  }
})

test_that("DP epsilon curves are comparable within each PyTorch template", {
  path <- system.file(
    "extdata", "dsflower_clinical_dp_curve_results.json",
    package = "dsFlowerClient"
  )
  evidence <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  models <- vapply(evidence$results, `[[`, character(1), "model_id")

  expect_equal(sort(unique(models)), c("pytorch_logreg", "pytorch_mlp"))
  for (model in unique(models)) {
    rows <- evidence$results[models == model]
    eps <- vapply(rows, `[[`, numeric(1), "curve_epsilon")
    auc <- vapply(rows, function(x) x$federated_metrics$auc, numeric(1))
    failures <- vapply(rows, function(x) {
      sum(vapply(x$history %||% list(), function(round) round$n_failures %||% 0L, integer(1)))
    }, integer(1))

    expect_equal(unname(sort(eps)), c(2, 4, 8), info = model)
    expect_true(all(failures == 0L), info = model)
    expect_true(all(diff(auc[order(eps)]) >= -1e-6), info = model)
    expect_true(all(vapply(rows, function(x) {
      x$n_total == 9000L &&
        identical(x$privacy_parameters$delta, 1e-5) &&
        identical(x$privacy_parameters$clipping_norm, 1L)
    }, logical(1))), info = model)
  }
})

test_that("XGBoost update-noise curve covers practical histogram-noise budgets", {
  path <- system.file(
    "extdata", "dsflower_xgboost_histogram_privacy_curve_results.json",
    package = "dsFlowerClient"
  )
  evidence <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  eps <- vapply(evidence$results, `[[`, numeric(1), "curve_epsilon")
  auc <- vapply(evidence$results, function(x) x$federated_metrics$auc, numeric(1))

  expect_equal(evidence$privacy_profile, "clinical_update_noise")
  expect_equal(unname(sort(eps)), c(8, 12, 16))
  expect_true(all(diff(auc[order(eps)]) >= -1e-6))
  expect_true(all(vapply(evidence$results, function(x) {
    x$dataset_id == "cdc_diabetes_health_indicators" &&
      x$model_id == "xgboost_histogram" &&
      x$n_total == 9000L &&
      identical(x$privacy_parameters$delta, 1e-5) &&
      identical(x$privacy_parameters$clipping_norm, 5L) &&
      abs(x$metric_delta$auc) <= 0.12
  }, logical(1))))
})

test_that("standalone XGBoost update-noise evidence uses the representative curve point", {
  path <- system.file(
    "extdata", "dsflower_xgboost_histogram_privacy_results.json",
    package = "dsFlowerClient"
  )
  evidence <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  result <- evidence$results[[1]]

  expect_equal(evidence$privacy_profile, "clinical_update_noise")
  expect_equal(evidence$source_curve_file,
               "dsflower_xgboost_histogram_privacy_curve_results.json")
  expect_equal(evidence$representative_curve_epsilon, 12)
  expect_equal(result$curve_epsilon, 12)
  expect_equal(result$privacy_parameters$epsilon, 12)
  expect_lte(abs(result$metric_delta$auc), 0.06)
})

test_that("SUPPORT2 method-family evidence covers non-binary families under SecAgg", {
  path <- system.file(
    "extdata", "dsflower_method_family_results.json",
    package = "dsFlowerClient"
  )
  expect_true(file.exists(path))
  evidence <- jsonlite::fromJSON(path)

  expect_equal(evidence$dataset_id, "support2")
  expect_equal(evidence$privacy_profile, "clinical_default")
  expect_true(isTRUE(evidence$secagg_supported))
  expect_true(isTRUE(evidence$privacy_policy$secure_aggregation_required))
  expect_equal(evidence$privacy_policy$dp_scope, "none")
  expect_equal(evidence$n_total, 3000)
  expect_true(all(unlist(evidence$site_train_n, use.names = FALSE) >= 1000))
  expect_true(all(unlist(evidence$site_event_n, use.names = FALSE) >= 20))
  expect_equal(
    sort(evidence$results$method),
    sort(c(
      "pytorch_linear_regression", "pytorch_poisson",
      "pytorch_multiclass", "pytorch_multilabel", "pytorch_coxph"
    ))
  )
  expect_true(all(evidence$results$validation_status == "pass"))
  expect_true(all(evidence$results$acceptable_loss))
  expect_true(all(evidence$results$federated_n_failures == 0))
  expect_true(all(evidence$results$secure_aggregation_required))
  expect_true(all(!evidence$results$dp_required))
})

test_that("SUPPORT2 DP-SGD method-family evidence covers compatible non-survival families", {
  path <- system.file(
    "extdata", "dsflower_method_family_dp_results.json",
    package = "dsFlowerClient"
  )
  expect_true(file.exists(path))
  evidence <- jsonlite::fromJSON(path)

  expect_equal(evidence$dataset_id, "support2")
  expect_equal(evidence$privacy_profile, "high_sensitivity_dp")
  expect_true(isTRUE(evidence$secagg_supported))
  expect_true(isTRUE(evidence$privacy_policy$secure_aggregation_required))
  expect_equal(evidence$privacy_policy$dp_scope, "patient_level_dp_sgd")
  expect_equal(evidence$n_total, 3000)
  expect_true(all(unlist(evidence$site_train_n, use.names = FALSE) >= 1000))
  expect_equal(
    sort(evidence$results$method),
    sort(c(
      "pytorch_linear_regression", "pytorch_poisson",
      "pytorch_multiclass", "pytorch_multilabel"
    ))
  )
  expect_true(all(evidence$results$validation_status == "pass"))
  expect_true(all(evidence$results$acceptable_loss))
  expect_true(all(evidence$results$federated_n_failures == 0))
  expect_true(all(evidence$results$secure_aggregation_required))
  expect_true(all(evidence$results$dp_required))
  expect_equal(
    evidence$results$rounds[evidence$results$method == "pytorch_multiclass"],
    6
  )
})
