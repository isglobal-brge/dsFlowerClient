# Tests for R/privacy.R — Privacy Specs

test_that("research mode has correct structure", {
  p <- ds.flower.privacy.research()
  expect_s3_class(p, "dsflower_privacy")
  expect_equal(p$mode, "research")
  expect_equal(length(p$params), 0)
})

test_that("dp mode has correct defaults", {
  p <- ds.flower.privacy.dp()
  expect_s3_class(p, "dsflower_privacy")
  expect_equal(p$mode, "dp")
  expect_equal(p$params$epsilon, 1.0)
  expect_equal(p$params$delta, 1e-5)
  expect_equal(p$params$clipping_norm, 1.0)
})

test_that("dp mode accepts overrides", {
  p <- ds.flower.privacy.dp(epsilon = 0.5, delta = 1e-6, clipping_norm = 2.0)
  expect_equal(p$params$epsilon, 0.5)
  expect_equal(p$params$delta, 1e-6)
  expect_equal(p$params$clipping_norm, 2.0)
})

test_that("privacy prints correctly", {
  p <- ds.flower.privacy.research()
  expect_output(print(p), "research")

  p2 <- ds.flower.privacy.dp()
  expect_output(print(p2), "dp")
})
