# Extracted from test-strategy.R:28

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "dsFlowerClient", path = "..")
attach(test_env, warn.conflicts = FALSE)

# test -------------------------------------------------------------------------
s <- ds.flower.strategy.fedprox(proximal_mu = 0.5, min_fit_clients = 4L)
