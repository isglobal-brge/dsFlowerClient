# Tests for R/strategy.R — Strategy Specs

test_that("fedavg creates correct strategy with defaults", {
  s <- ds.flower.strategy.fedavg()
  expect_s3_class(s, "dsflower_strategy")
  expect_equal(s$name, "FedAvg")
  expect_identical(s$params, list())
})

test_that("unsupported local-training strategies fail at construction", {
  expect_error(ds.flower.strategy("fedprox"), "not supported")
  expect_error(ds.flower.strategy("fedbn"), "not supported")
})

test_that("fedadam creates correct strategy with defaults", {
  s <- ds.flower.strategy.fedadam()
  expect_s3_class(s, "dsflower_strategy")
  expect_equal(s$name, "FedAdam")
  expect_equal(s$params$eta, 0.1)
  expect_equal(s$params$beta_1, 0.9)
  expect_equal(s$params$beta_2, 0.99)
  expect_equal(s$params$tau, 1e-3)
})

test_that("fedadam accepts overrides", {
  s <- ds.flower.strategy.fedadam(server_learning_rate = 0.2, tau = 0.01)
  expect_equal(s$params$eta, 0.2)
  expect_equal(s$params$tau, 0.01)
})

test_that("fedadagrad creates correct strategy with defaults", {
  s <- ds.flower.strategy.fedadagrad()
  expect_s3_class(s, "dsflower_strategy")
  expect_equal(s$name, "FedAdagrad")
  expect_equal(s$params$eta, 0.1)
  expect_equal(s$params$tau, 1e-3)
})

test_that("fedadagrad accepts overrides", {
  s <- ds.flower.strategy.fedadagrad(server_learning_rate = 0.05, tau = 0.1)
  expect_equal(s$params$eta, 0.05)
  expect_equal(s$params$tau, 0.1)
})

test_that("FedYogi and FedAvgM expose all runtime hyperparameters", {
  y <- ds.flower.strategy.fedyogi(
    server_learning_rate = 0.02,
    beta_1 = 0.8, beta_2 = 0.95, tau = 0.002)
  expect_equal(y$params,
               list(eta = 0.02, beta_1 = 0.8,
                    beta_2 = 0.95, tau = 0.002))

  m <- ds.flower.strategy.fedavgm(
    server_learning_rate = 0.8, server_momentum = 0.7)
  expect_equal(m$params,
               list(server_learning_rate = 0.8, server_momentum = 0.7))
})

test_that("strategy config serializes only server-side post-processing knobs", {
  lines <- dsFlowerClient:::.strategy_config_lines(
    ds.flower.strategy.fedadam(server_learning_rate = 0.2, beta_1 = 0.8),
    client_learning_rate = 0.05)
  expect_true('strategy = "fedadam"' %in% lines)
  expect_true("strategy-eta = 0.2" %in% lines)
  expect_true("strategy-beta-1 = 0.8" %in% lines)
  expect_true("strategy-eta-l = 0.05" %in% lines)
  expect_false(any(grepl("fraction|num-examples", lines)))
})

test_that("client learning rate cannot be overridden inside a strategy object", {
  forged <- structure(
    list(name = "FedAdam", params = list(eta_l = 9)),
    class = "dsflower_strategy"
  )
  expect_error(
    dsFlowerClient:::.strategy_config_lines(
      forged, client_learning_rate = 0.05),
    "Unknown parameters"
  )
})

test_that("strategy prints correctly", {
  s <- ds.flower.strategy.fedavg()
  expect_output(print(s), "FedAvg")
})
