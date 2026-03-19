# Tests for R/privacy.R -- Privacy Specs (7 profiles + evaluation_only)

# --- All 7 constructors ---

test_that("sandbox_open has correct structure", {
  p <- ds.flower.privacy.sandbox_open()
  expect_s3_class(p, "dsflower_privacy")
  expect_equal(p$mode, "sandbox_open")
  expect_equal(length(p$params), 0)
})

test_that("trusted_internal has correct structure", {
  p <- ds.flower.privacy.trusted_internal()
  expect_s3_class(p, "dsflower_privacy")
  expect_equal(p$mode, "trusted_internal")
  expect_equal(length(p$params), 0)
})

test_that("consortium_internal has correct structure", {
  p <- ds.flower.privacy.consortium_internal()
  expect_s3_class(p, "dsflower_privacy")
  expect_equal(p$mode, "consortium_internal")
  expect_equal(length(p$params), 0)
})

test_that("clinical_default has correct structure", {
  p <- ds.flower.privacy.clinical_default()
  expect_s3_class(p, "dsflower_privacy")
  expect_equal(p$mode, "clinical_default")
  expect_equal(length(p$params), 0)
})

test_that("clinical_hardened has correct structure", {
  p <- ds.flower.privacy.clinical_hardened()
  expect_s3_class(p, "dsflower_privacy")
  expect_equal(p$mode, "clinical_hardened")
  expect_equal(length(p$params), 0)
})

test_that("clinical_dp has correct defaults", {
  p <- ds.flower.privacy.clinical_dp()
  expect_s3_class(p, "dsflower_privacy")
  expect_equal(p$mode, "clinical_dp")
  expect_equal(p$params$epsilon, 1.0)
  expect_equal(p$params$delta, 1e-5)
  expect_equal(p$params$clipping_norm, 1.0)
})

test_that("clinical_dp accepts overrides", {
  p <- ds.flower.privacy.clinical_dp(epsilon = 0.5, delta = 1e-6,
                                      clipping_norm = 2.0)
  expect_equal(p$params$epsilon, 0.5)
  expect_equal(p$params$delta, 1e-6)
  expect_equal(p$params$clipping_norm, 2.0)
})

test_that("high_sensitivity_dp has correct defaults", {
  p <- ds.flower.privacy.high_sensitivity_dp()
  expect_s3_class(p, "dsflower_privacy")
  expect_equal(p$mode, "high_sensitivity_dp")
  expect_equal(p$params$epsilon, 1.0)
  expect_equal(p$params$delta, 1e-5)
  expect_equal(p$params$clipping_norm, 1.0)
})

test_that("high_sensitivity_dp accepts overrides", {
  p <- ds.flower.privacy.high_sensitivity_dp(epsilon = 0.1, delta = 1e-7,
                                               clipping_norm = 0.5)
  expect_equal(p$params$epsilon, 0.1)
  expect_equal(p$params$delta, 1e-7)
  expect_equal(p$params$clipping_norm, 0.5)
})

# --- DP param validation ---

test_that("clinical_dp rejects invalid params", {
  expect_error(ds.flower.privacy.clinical_dp(epsilon = -1), "positive")
  expect_error(ds.flower.privacy.clinical_dp(delta = 0), "\\(0, 1\\)")
  expect_error(ds.flower.privacy.clinical_dp(delta = 1), "\\(0, 1\\)")
  expect_error(ds.flower.privacy.clinical_dp(clipping_norm = 0), "positive")
})

test_that("high_sensitivity_dp rejects invalid params", {
  expect_error(ds.flower.privacy.high_sensitivity_dp(epsilon = -1), "positive")
  expect_error(ds.flower.privacy.high_sensitivity_dp(delta = 0), "\\(0, 1\\)")
  expect_error(ds.flower.privacy.high_sensitivity_dp(delta = 1), "\\(0, 1\\)")
  expect_error(ds.flower.privacy.high_sensitivity_dp(clipping_norm = 0), "positive")
})

# --- evaluation_only modifier ---

test_that("evaluation_only sets flag on base privacy", {
  base <- ds.flower.privacy.clinical_default()
  p <- ds.flower.privacy.evaluation_only(base)
  expect_s3_class(p, "dsflower_privacy")
  expect_equal(p$mode, "clinical_default")
  expect_true(p$params$evaluation_only)
})

test_that("evaluation_only works with DP profiles", {
  base <- ds.flower.privacy.clinical_dp(epsilon = 0.5)
  p <- ds.flower.privacy.evaluation_only(base)
  expect_equal(p$mode, "clinical_dp")
  expect_true(p$params$evaluation_only)
  expect_equal(p$params$epsilon, 0.5)
})

test_that("evaluation_only rejects non-privacy input", {
  expect_error(
    ds.flower.privacy.evaluation_only(list(mode = "test")),
    "dsflower_privacy"
  )
})

# --- Print method ---

test_that("privacy prints correctly for all modes", {
  expect_output(print(ds.flower.privacy.sandbox_open()), "sandbox_open")
  expect_output(print(ds.flower.privacy.trusted_internal()), "trusted_internal")
  expect_output(print(ds.flower.privacy.consortium_internal()), "consortium_internal")
  expect_output(print(ds.flower.privacy.clinical_default()), "clinical_default")
  expect_output(print(ds.flower.privacy.clinical_hardened()), "clinical_hardened")
  expect_output(print(ds.flower.privacy.clinical_dp()), "clinical_dp")
  expect_output(print(ds.flower.privacy.high_sensitivity_dp()), "high_sensitivity_dp")
})

test_that("evaluation_only modifier appears in print output", {
  p <- ds.flower.privacy.evaluation_only(ds.flower.privacy.clinical_default())
  expect_output(print(p), "evaluation_only")
})
