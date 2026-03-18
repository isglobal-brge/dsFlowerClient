# Extracted from test-strategy.R:14

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "dsFlowerClient", path = "..")
attach(test_env, warn.conflicts = FALSE)

# test -------------------------------------------------------------------------
s <- ds.flower.strategy.fedavg(fraction_fit = 0.5, min_fit_clients = 3L)
