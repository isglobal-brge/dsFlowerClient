test_that("FedProx validates a finite bounded numeric mu and aliases", {
  for (bad in list(TRUE, "0.1", numeric(), c(0, 1), NA_real_, NaN, Inf, -0.01, 1.01)) {
    expect_error(ds.flower.strategy.fedprox(bad), "mu")
  }
  expect_identical(ds.flower.strategy("prox", mu = 0.25), ds.flower.strategy.fedprox(0.25))
  expect_identical(dsFlowerClient:::.strategy_config_values(ds.flower.strategy.fedprox(0.25)),
                   list(strategy = "fedprox", "strategy-mu" = 0.25))
  bad <- ds.flower.strategy.fedprox()
  bad$params$ignored <- 1
  expect_error(dsFlowerClient:::.strategy_config_values(bad), "inapplicable")
})

test_that("FedProx zero is exactly FedAvg on supported tracks", {
  zero <- ds.flower.strategy.fedprox(0)
  avg <- ds.flower.strategy.fedavg()
  expect_identical(dsFlowerClient:::.strategy_config_values(zero), dsFlowerClient:::.strategy_config_values(avg))
  expect_identical(ds.flower.recipe("logreg", strategy = zero), ds.flower.recipe("logreg", strategy = avg))
  for (track in c("native_tree", "association", "validation")) {
    expect_error(dsFlowerClient:::.effective_strategy(zero, track), "unsupported")
  }
  expect_error(dsFlowerClient:::.effective_strategy(ds.flower.strategy.fedadam(), "egress"), "only FedAvg or FedProx")
})

test_that("Every native tree rejects raw FedProx including zero", {
  for (engine in c("extra_trees", "random_forest", "lightgbm", "catboost", "xgboost")) {
    for (mu in c(0, .25)) expect_error(
      ds.flower.recipe(engine, strategy = ds.flower.strategy.fedprox(mu)), "FedProx")
  }
})

test_that("FedProx checks the complete public learning rate horizon", {
  sub <- list(params = list(learning_rate = 0.5, local_epochs = 2L,
                           scheduler = "exponential", scheduler_gamma = 2))
  expect_error(dsFlowerClient:::.validate_fedprox_horizon(ds.flower.strategy.fedprox(1), sub, 2L), "learning_rate")
  expect_silent(dsFlowerClient:::.validate_fedprox_horizon(ds.flower.strategy.fedprox(0), sub, 2L))
})

test_that("FedProx zero preserves the complete CV strategy projection", {
  avg <- dsFlowerClient:::.cv_job_strategy(list(strategy = "fedavg"))
  zero <- dsFlowerClient:::.cv_job_strategy(list(strategy = "fedprox", "strategy-mu" = 0))
  expect_identical(avg, zero)
  expect_identical(dsFlowerClient:::.cv_job_strategy(list(strategy = "fedprox", "strategy-mu" = .2))$params$mu, .2)
  expect_error(dsFlowerClient:::.cv_job_strategy(list(strategy = "fedprox", "strategy-mu" = .2, "strategy-eta" = 1)), "fields")
})
