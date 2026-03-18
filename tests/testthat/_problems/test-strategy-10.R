# Extracted from test-strategy.R:10

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "dsFlowerClient", path = "..")
attach(test_env, warn.conflicts = FALSE)

# test -------------------------------------------------------------------------
s <- ds.flower.strategy.fedavg()
expect_s3_class(s, "dsflower_strategy")
expect_equal(s$name, "FedAvg")
expect_equal(s$params$fraction_fit, 1.0)
expect_equal(s$params$fraction_evaluate, 1.0)
expect_equal(s$params$min_fit_clients, 2L)
expect_equal(s$params$min_available_clients, 2L)
