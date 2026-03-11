# Tests for R/metrics.R — Metrics Collection

test_that(".pool_metrics computes mean across servers", {
  results <- list(
    srv1 = data.frame(
      round = c(1L, 2L), metric = c("loss", "loss"),
      value = c(0.6, 0.4), stringsAsFactors = FALSE
    ),
    srv2 = data.frame(
      round = c(1L, 2L), metric = c("loss", "loss"),
      value = c(0.8, 0.5), stringsAsFactors = FALSE
    )
  )

  pooled <- dsFlowerClient:::.pool_metrics(results)
  expect_s3_class(pooled, "data.frame")

  r1_loss <- pooled[pooled$round == 1 & pooled$metric == "loss", ]
  expect_equal(r1_loss$value, 0.7)
  expect_equal(r1_loss$n_servers, 2L)
})

test_that(".pool_metrics handles empty results", {
  pooled <- dsFlowerClient:::.pool_metrics(list())
  expect_s3_class(pooled, "data.frame")
  expect_equal(nrow(pooled), 0)
})

test_that(".pool_metrics handles single server", {
  results <- list(
    srv1 = data.frame(
      round = 1L, metric = "loss", value = 0.5,
      stringsAsFactors = FALSE
    )
  )
  pooled <- dsFlowerClient:::.pool_metrics(results)
  expect_equal(nrow(pooled), 1)
  expect_equal(pooled$value, 0.5)
  expect_equal(pooled$n_servers, 1L)
})

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
