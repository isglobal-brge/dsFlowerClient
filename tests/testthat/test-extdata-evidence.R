extdata_evidence <- function(file) {
  candidates <- c(
    testthat::test_path("..", "..", "inst", "extdata", file),
    system.file("extdata", file, package = "dsFlowerClient")
  )
  candidates <- candidates[nzchar(candidates) & file.exists(candidates)]
  expect_gt(length(candidates), 0)
  candidates[[1]]
}

test_that("committed public benchmark evidence matches thesis constants", {
  path <- extdata_evidence("dsflower_public_benchmark_results.json")
  evidence <- jsonlite::fromJSON(path, simplifyVector = FALSE)

  expect_equal(evidence$n_demos, 3)
  expect_equal(sort(names(evidence$results)),
               sort(c("breast_cancer_wisconsin", "uci_heart_disease",
                      "medmnist_breastmnist")))

  breast <- evidence$results$breast_cancer_wisconsin
  expect_equal(breast$n_total, 683)
  expect_equal(breast$n_train, 513)
  expect_equal(breast$n_test, 170)
  expect_equal(breast$central_metrics$auc, 0.9878, tolerance = 1e-6)
  expect_equal(breast$federated_metrics$auc, 0.9908, tolerance = 1e-6)
  expect_equal(sum(vapply(breast$history, `[[`, numeric(1), "n_failures")), 0)

  heart <- evidence$results$uci_heart_disease
  expect_equal(heart$n_total, 297)
  expect_equal(heart$central_metrics$auc, 0.8919, tolerance = 1e-6)
  expect_equal(heart$federated_metrics$auc, 0.8926, tolerance = 1e-6)
  expect_equal(sum(vapply(heart$history, `[[`, numeric(1), "n_failures")), 0)

  medmnist <- evidence$results$medmnist_breastmnist
  expect_equal(medmnist$n_total, 510)
  expect_equal(medmnist$central_metrics$auc, 0.7654, tolerance = 1e-6)
  expect_equal(medmnist$federated_metrics$auc, 0.7985, tolerance = 1e-6)
  expect_equal(sum(vapply(medmnist$history, `[[`, numeric(1), "n_failures")), 0)
})
