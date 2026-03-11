# Tests for R/model.R — Model Specs

test_that("sklearn_logreg creates correct model", {
  m <- ds.flower.model.sklearn_logreg()
  expect_s3_class(m, "dsflower_model")
  expect_equal(m$name, "sklearn_logreg")
  expect_equal(m$framework, "sklearn")
  expect_equal(m$template, "sklearn_logreg")
  expect_equal(m$params$penalty, "l2")
  expect_equal(m$params$C, 1.0)
  expect_equal(m$params$max_iter, 100L)
})

test_that("sklearn_logreg accepts overrides", {
  m <- ds.flower.model.sklearn_logreg(penalty = "l1", C = 0.5, max_iter = 200L)
  expect_equal(m$params$penalty, "l1")
  expect_equal(m$params$C, 0.5)
  expect_equal(m$params$max_iter, 200L)
})

test_that("sklearn_ridge creates correct model", {
  m <- ds.flower.model.sklearn_ridge()
  expect_s3_class(m, "dsflower_model")
  expect_equal(m$name, "sklearn_ridge")
  expect_equal(m$framework, "sklearn")
  expect_equal(m$params$alpha, 1.0)
})

test_that("sklearn_ridge accepts overrides", {
  m <- ds.flower.model.sklearn_ridge(alpha = 0.1)
  expect_equal(m$params$alpha, 0.1)
})

test_that("sklearn_sgd creates correct model", {
  m <- ds.flower.model.sklearn_sgd()
  expect_s3_class(m, "dsflower_model")
  expect_equal(m$name, "sklearn_sgd")
  expect_equal(m$params$loss, "log_loss")
  expect_equal(m$params$alpha, 0.0001)
  expect_equal(m$params$lr_schedule, "optimal")
})

test_that("pytorch_mlp creates correct model", {
  m <- ds.flower.model.pytorch_mlp()
  expect_s3_class(m, "dsflower_model")
  expect_equal(m$name, "pytorch_mlp")
  expect_equal(m$framework, "pytorch")
  expect_equal(m$params$hidden_layers, c(64L, 32L))
  expect_equal(m$params$learning_rate, 0.01)
  expect_equal(m$params$batch_size, 32L)
  expect_equal(m$params$local_epochs, 1L)
})

test_that("pytorch_mlp accepts overrides", {
  m <- ds.flower.model.pytorch_mlp(
    hidden_layers = c(128L, 64L, 32L),
    learning_rate = 0.001,
    batch_size = 64L
  )
  expect_equal(m$params$hidden_layers, c(128L, 64L, 32L))
  expect_equal(m$params$learning_rate, 0.001)
  expect_equal(m$params$batch_size, 64L)
})

test_that("model prints correctly", {
  m <- ds.flower.model.sklearn_logreg()
  expect_output(print(m), "sklearn_logreg")
})
