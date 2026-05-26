`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || (length(x) == 1 && is.na(x))) y else x
}

test_that("clinical benchmark evidence contains only completed runs", {
  files <- c(
    "dsflower_clinical_algorithm_results.json",
    "dsflower_clinical_secagg_results.json",
    "dsflower_clinical_dp_results.json"
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
      expect_true(abs(result$metric_delta$auc) <= 0.02, info = file)

      failures <- sum(vapply(
        result$history %||% list(),
        function(round) round$n_failures %||% 0L,
        integer(1)
      ))
      expect_equal(failures, 0L, info = file)
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
