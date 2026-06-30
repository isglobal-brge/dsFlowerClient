# Tests for R/convenience.R -- string-friendly user API
# (Tests asserting the removed sklearn_* models / the removed ds.flower.run mock
# were deleted with the legacy sklearn + recipe-run path; the fit happy path is
# covered end-to-end by the live federated validation.)

test_that("generic strategy constructor resolves aliases", {
  strategy <- ds.flower.strategy("fedprox", proximal_mu = 0.25)
  expect_s3_class(strategy, "dsflower_strategy")
  expect_equal(strategy$name, "FedProx")
  expect_equal(strategy$params$proximal_mu, 0.25)

  expect_equal(ds.flower.strategy("fed-average")$name, "FedAvg")
  expect_error(ds.flower.strategy("not_a_strategy"), "Unknown strategy")
})

test_that("generic task constructor resolves aliases", {
  expect_equal(ds.flower.task("class")$type, "classification")
  expect_equal(ds.flower.task("survival")$type, "survival")
  expect_error(ds.flower.task("not_a_task"), "Unknown task")
})

test_that("fit validates required arguments before connecting", {
  expect_error(ds.flower.fit(conns = list()), "target")
  expect_error(
    ds.flower.fit(conns = list(), symbol = "D", data = "D", target = "y"),
    "only one"
  )
})
