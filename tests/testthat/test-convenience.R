# Tests for R/convenience.R -- string-friendly user API
# (Tests asserting the removed sklearn_* models / the removed ds.flower.run mock
# were deleted with the legacy sklearn + recipe-run path; the fit happy path is
# covered end-to-end by the live federated validation.)

test_that("generic strategy constructor resolves aliases", {
  strategy <- ds.flower.strategy("yogi", server_learning_rate = 0.25)
  expect_s3_class(strategy, "dsflower_strategy")
  expect_equal(strategy$name, "FedYogi")
  expect_equal(strategy$params$eta, 0.25)

  expect_equal(ds.flower.strategy("fed-average")$name, "FedAvg")
  expect_equal(ds.flower.strategy("avgm")$name, "FedAvgM")
  expect_error(ds.flower.strategy("fedprox"), "not supported")
  expect_error(ds.flower.strategy("fedbn"), "not supported")
  expect_error(ds.flower.strategy("not_a_strategy"), "Unknown strategy")
})

test_that("generic task constructor resolves aliases", {
  expect_equal(ds.flower.task("class")$type, "classification")
  expect_error(ds.flower.task("survival"), "not supported")
  expect_error(ds.flower.task("segmentation"), "not supported")
  expect_error(ds.flower.task("not_a_task"), "Unknown task")
})

test_that("fit resolves strategy_params into the executable strategy", {
  seen <- NULL
  local_mocked_bindings(
    ds.flower.submit = function(...) {
      seen <<- list(...)
      structure(list(), class = "dsflower_run")
    },
    .package = "dsFlowerClient"
  )

  ds.flower.fit(
    conns = list(site = TRUE), symbol = "D", target = "y", features = "x",
    strategy = "fedadam",
    allow_insecure_http = "site",
    strategy_params = list(server_learning_rate = 0.2,
                           beta_1 = 0.8, beta_2 = 0.95, tau = 1e-4)
  )

  expect_s3_class(seen$strategy, "dsflower_strategy")
  expect_identical(seen$strategy$name, "FedAdam")
  expect_equal(seen$strategy$params,
               list(eta = 0.2, beta_1 = 0.8,
                    beta_2 = 0.95, tau = 1e-4))
  expect_identical(seen$allow_insecure_http, "site")
})

test_that("fit forwards registry parameters not present in model defaults", {
  seen <- NULL
  local_mocked_bindings(
    ds.flower.submit = function(...) {
      seen <<- list(...)
      structure(list(), class = "dsflower_run")
    },
    .package = "dsFlowerClient"
  )

  ds.flower.fit(
    conns = list(site = TRUE), symbol = "D", target = "y",
    features = paste0("x", 1:24), model = "pytorch_tcn",
    model_params = list(input_shape = c(2L, 12L), channels = 16L)
  )

  expect_identical(seen$model$params$input_shape, c(2L, 12L))
  expect_identical(seen$model$params$channels, 16L)
})

test_that("fit validates required arguments before connecting", {
  expect_error(ds.flower.fit(conns = list()), "target")
  expect_error(
    ds.flower.fit(conns = list(), symbol = "D", data = "D", target = "y"),
    "only one"
  )
  expect_error(
    ds.flower.fit(conns = list(), target = "y", masks = "tumour"),
    "segmentation"
  )
})
