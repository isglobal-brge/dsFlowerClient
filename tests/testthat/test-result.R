# Tests for R/result.R — Result Objects

test_that("dsflower_result creates correct structure", {
  result <- dsFlowerClient:::dsflower_result(
    per_site = list(srv1 = list(status = "ok")),
    meta = list(call_code = "test()")
  )
  expect_s3_class(result, "dsflower_result")
  expect_equal(result$meta$servers, "srv1")
  expect_equal(result$meta$call_code, "test()")
})

test_that("dsflower_result defaults", {
  result <- dsFlowerClient:::dsflower_result(per_site = list())
  expect_equal(result$meta$call_code, "")
  expect_equal(result$meta$scope, "per_site")
  expect_null(result$pooled)
})

test_that("print.dsflower_result works", {
  result <- dsFlowerClient:::dsflower_result(
    per_site = list(srv1 = "ok"),
    meta = list(scope = "per_site")
  )
  expect_output(print(result), "dsflower_result")
  expect_output(print(result), "srv1")
})

test_that("$ operator accesses per_site elements", {
  result <- dsFlowerClient:::dsflower_result(
    per_site = list(srv1 = "value1")
  )
  expect_equal(result$srv1, "value1")
  expect_equal(result$per_site, list(srv1 = "value1"))
  expect_null(result$pooled)
})

test_that("as.data.frame returns pooled if available", {
  df <- data.frame(round = 1:3, metric = "loss", value = c(0.5, 0.4, 0.3))
  result <- dsFlowerClient:::dsflower_result(
    per_site = list(),
    pooled = df
  )
  expect_equal(as.data.frame(result), df)
})

test_that("as.data.frame returns first per_site if no pooled", {
  df <- data.frame(round = 1L, metric = "loss", value = 0.5)
  result <- dsFlowerClient:::dsflower_result(
    per_site = list(srv1 = df)
  )
  expect_equal(as.data.frame(result), df)
})

test_that("ds.flower.code extracts code", {
  result <- dsFlowerClient:::dsflower_result(
    per_site = list(),
    meta = list(call_code = 'ds.flower.metrics("flower")')
  )
  expect_equal(ds.flower.code(result), 'ds.flower.metrics("flower")')
})

test_that("ds.flower.code errors for non-result", {
  expect_error(ds.flower.code(list()), "dsflower_result")
})
