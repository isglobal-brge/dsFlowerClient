# Privacy parameters are server-owned.  The client intentionally exports no
# constructor or profiles capable of suggesting that an analyst can set them.
test_that("client-side privacy controls do not exist", {
  expect_false(exists("ds.flower.privacy", mode = "function"))
  expect_false(exists("ds.flower.privacy.clinical_default"))
  expect_false(exists("ds.flower.privacy.high_sensitivity_dp"))
  expect_false(exists("ds.flower.privacy.auto"))
})
