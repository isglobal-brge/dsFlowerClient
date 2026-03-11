# Tests for R/plot.R — Plotting

test_that("ds.flower.plot creates plotly for result", {
  skip_if_not_installed("plotly")

  df <- data.frame(
    round = 1:3, metric = rep("loss", 3),
    value = c(0.6, 0.4, 0.3),
    stringsAsFactors = FALSE
  )
  result <- dsFlowerClient:::dsflower_result(
    per_site = list(srv1 = df),
    pooled = df
  )

  p <- ds.flower.plot(result, metric = "loss")
  expect_s3_class(p, "plotly")
})

test_that("ds.flower.plot handles per_server mode", {
  skip_if_not_installed("plotly")

  df <- data.frame(
    round = 1:3, metric = rep("loss", 3),
    value = c(0.6, 0.4, 0.3),
    stringsAsFactors = FALSE
  )
  result <- dsFlowerClient:::dsflower_result(
    per_site = list(srv1 = df, srv2 = df)
  )

  p <- ds.flower.plot(result, metric = "loss", per_server = TRUE)
  expect_s3_class(p, "plotly")
})

test_that("ds.flower.plot handles comparison data.frame", {
  skip_if_not_installed("plotly")

  df <- data.frame(
    run = rep(c("run1", "run2"), each = 3),
    round = rep(1:3, 2),
    metric = rep("loss", 6),
    value = c(0.6, 0.4, 0.3, 0.7, 0.5, 0.4),
    stringsAsFactors = FALSE
  )

  p <- ds.flower.plot(df, metric = "loss")
  expect_s3_class(p, "plotly")
})

test_that("ds.flower.plot handles empty result", {
  skip_if_not_installed("plotly")

  result <- dsFlowerClient:::dsflower_result(
    per_site = list()
  )

  p <- ds.flower.plot(result, metric = "loss")
  expect_s3_class(p, "plotly")
})

test_that("ds.flower.plot errors without plotly", {
  # Can't really test this without unloading plotly, but test error for wrong input
  expect_error(
    ds.flower.plot(42),
    "dsflower_result"
  )
})
