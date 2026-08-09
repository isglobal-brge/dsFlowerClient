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
