# Tests for R/privacy.R -- single differential-privacy budget constructor.
# Differential privacy is always enforced; there are no privacy profiles.

test_that("ds.flower.privacy has correct structure and defaults", {
  p <- ds.flower.privacy()
  expect_s3_class(p, "dsflower_privacy")
  expect_equal(p$epsilon, 3.0)
  expect_equal(p$delta, 1e-5)
  expect_equal(p$clipping_norm, 1.0)
})

test_that("ds.flower.privacy accepts overrides", {
  p <- ds.flower.privacy(epsilon = 1.0, delta = 1e-6, clipping_norm = 0.5)
  expect_equal(p$epsilon, 1.0)
  expect_equal(p$delta, 1e-6)
  expect_equal(p$clipping_norm, 0.5)
})

test_that("ds.flower.privacy validates its arguments", {
  expect_error(ds.flower.privacy(epsilon = 0), "epsilon")
  expect_error(ds.flower.privacy(epsilon = -1), "epsilon")
  expect_error(ds.flower.privacy(delta = 0), "delta")
  expect_error(ds.flower.privacy(delta = 1), "delta")
  expect_error(ds.flower.privacy(clipping_norm = 0), "clipping")
})

test_that("print.dsflower_privacy reports DP", {
  expect_output(print(ds.flower.privacy()), "DP")
})

test_that("the removed profile constructors no longer exist", {
  expect_false(exists("ds.flower.privacy.clinical_default"))
  expect_false(exists("ds.flower.privacy.high_sensitivity_dp"))
  expect_false(exists("ds.flower.privacy.auto"))
})
