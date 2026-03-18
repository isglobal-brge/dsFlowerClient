# Tests for R/strategy.R — Strategy Specs

test_that("fedavg creates correct strategy with defaults", {
  s <- ds.flower.strategy.fedavg()
  expect_s3_class(s, "dsflower_strategy")
  expect_equal(s$name, "FedAvg")
  expect_equal(s$params$fraction_fit, 1.0)
  expect_equal(s$params$fraction_evaluate, 1.0)
  expect_equal(s$params$min_fit_clients, 2L)
  expect_equal(s$params$min_available_clients, 2L)
})

test_that("fedavg accepts overrides", {
  s <- ds.flower.strategy.fedavg(fraction_fit = 0.5, min_fit_clients = 3L)
  expect_equal(s$params$fraction_fit, 0.5)
  expect_equal(s$params$min_fit_clients, 3L)
})

test_that("fedprox creates correct strategy with defaults", {
  s <- ds.flower.strategy.fedprox()
  expect_s3_class(s, "dsflower_strategy")
  expect_equal(s$name, "FedProx")
  expect_equal(s$params$proximal_mu, 0.1)
  expect_equal(s$params$fraction_fit, 1.0)
})

test_that("fedprox accepts overrides", {
  s <- ds.flower.strategy.fedprox(proximal_mu = 0.5, min_fit_clients = 4L)
  expect_equal(s$params$proximal_mu, 0.5)
  expect_equal(s$params$min_fit_clients, 4L)
})

test_that("fedadam creates correct strategy with defaults", {
  s <- ds.flower.strategy.fedadam()
  expect_s3_class(s, "dsflower_strategy")
  expect_equal(s$name, "FedAdam")
  expect_equal(s$params$eta, 0.01)
  expect_equal(s$params$tau, 1e-3)
  expect_equal(s$params$fraction_fit, 1.0)
  expect_equal(s$params$fraction_evaluate, 1.0)
})

test_that("fedadam accepts overrides", {
  s <- ds.flower.strategy.fedadam(server_learning_rate = 0.1, tau = 0.01)
  expect_equal(s$params$eta, 0.1)
  expect_equal(s$params$tau, 0.01)
})

test_that("fedadagrad creates correct strategy with defaults", {
  s <- ds.flower.strategy.fedadagrad()
  expect_s3_class(s, "dsflower_strategy")
  expect_equal(s$name, "FedAdagrad")
  expect_equal(s$params$eta, 0.01)
  expect_equal(s$params$tau, 1e-3)
  expect_equal(s$params$fraction_fit, 1.0)
  expect_equal(s$params$fraction_evaluate, 1.0)
})

test_that("fedadagrad accepts overrides", {
  s <- ds.flower.strategy.fedadagrad(server_learning_rate = 0.05, tau = 0.1)
  expect_equal(s$params$eta, 0.05)
  expect_equal(s$params$tau, 0.1)
})

test_that("strategy prints correctly", {
  s <- ds.flower.strategy.fedavg()
  expect_output(print(s), "FedAvg")
})
