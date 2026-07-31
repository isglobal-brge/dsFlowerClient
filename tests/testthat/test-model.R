# Tests for R/model.R — Model Specs

test_that("pytorch_mlp creates correct model", {
  m <- ds.flower.model.pytorch_mlp()
  expect_s3_class(m, "dsflower_model")
  expect_equal(m$name, "pytorch_mlp")
  expect_equal(m$framework, "pytorch")
  expect_equal(m$params$hidden_layers, "64,32")
  expect_equal(m$params$learning_rate, 0.1)
  expect_equal(m$params$batch_size, 32L)
  expect_equal(m$params$local_epochs, 1L)
})

test_that("pytorch_mlp accepts overrides", {
  m <- ds.flower.model.pytorch_mlp(
    hidden_layers = c(128L, 64L, 32L),
    learning_rate = 0.001,
    batch_size = 64L
  )
  expect_equal(m$params$hidden_layers, "128,64,32")
  expect_equal(m$params$learning_rate, 0.001)
  expect_equal(m$params$batch_size, 64L)
})

test_that("legacy comma-separated MLP widths emit valid declarative layers", {
  sub <- dsFlowerClient:::.emit_submission(
    ds.flower.model.pytorch_mlp(hidden_layers = c(128L, 64L)))
  widths <- vapply(
    Filter(function(layer) identical(layer$op, "linear"), sub$spec$layers),
    function(layer) as.character(layer$out), character(1))

  expect_identical(widths, c("128", "64", "@out"))
  expect_error(
    dsFlowerClient:::.emit_submission(
      ds.flower.model.pytorch_mlp(hidden_layers = "64,broken")),
    "positive integer"
  )
})

test_that("pytorch_logreg creates correct model", {
  m <- ds.flower.model.pytorch_logreg()
  expect_s3_class(m, "dsflower_model")
  expect_equal(m$name, "pytorch_logreg")
  expect_equal(m$framework, "pytorch")
  expect_equal(m$params$learning_rate, 0.1)
  expect_equal(m$params$batch_size, 32L)
  expect_equal(m$params$local_epochs, 1L)
})

test_that("pytorch_linear_regression creates correct model", {
  m <- ds.flower.model.pytorch_linear_regression()
  expect_s3_class(m, "dsflower_model")
  expect_equal(m$name, "pytorch_linear_regression")
  expect_equal(m$framework, "pytorch")
  expect_equal(m$params$learning_rate, 0.01)
})

test_that("pytorch_coxph creates correct model", {
  m <- ds.flower.model.pytorch_coxph()
  expect_s3_class(m, "dsflower_model")
  expect_equal(m$name, "pytorch_coxph")
  expect_equal(m$framework, "pytorch")
  expect_equal(m$params$learning_rate, 0.01)
})

test_that("pytorch_multiclass creates correct model", {
  m <- ds.flower.model.pytorch_multiclass()
  expect_s3_class(m, "dsflower_model")
  expect_equal(m$name, "pytorch_multiclass")
  expect_equal(m$framework, "pytorch")
  expect_equal(m$params$n_classes, 3L)
  expect_equal(m$params$hidden_layers, "")
})

test_that("pytorch_multiclass accepts overrides", {
  m <- ds.flower.model.pytorch_multiclass(
    hidden_layers = c(64L, 32L), n_classes = 5L
  )
  expect_equal(m$params$hidden_layers, "64,32")
  expect_equal(m$params$n_classes, 5L)
})

test_that("xgboost creates correct model", {
  m <- ds.flower.model.xgboost()
  expect_s3_class(m, "dsflower_model")
  expect_equal(m$name, "xgboost")
  expect_equal(m$framework, "xgboost")
  expect_equal(m$params$n_trees, 50L)
  expect_equal(m$params$max_depth, 3L)
  expect_equal(m$params$learning_rate, 0.3)
  expect_equal(m$params$reg_lambda, 1.0)
  expect_equal(m$params$n_bins, 64L)
  expect_equal(m$params$objective, "binary:logistic")
})

test_that("pytorch_resnet18 creates correct model", {
  m <- ds.flower.model.pytorch_resnet18()
  expect_s3_class(m, "dsflower_model")
  expect_equal(m$name, "pytorch_resnet18")
  expect_equal(m$framework, "pytorch_vision")
  expect_equal(m$params$n_classes, 2L)
  expect_equal(m$params$learning_rate, 0.001)
  expect_equal(m$params$image_size, 224L)
})

test_that("pytorch_densenet121 creates correct model", {
  m <- ds.flower.model.pytorch_densenet121()
  expect_s3_class(m, "dsflower_model")
  expect_equal(m$name, "pytorch_densenet121")
  expect_equal(m$framework, "pytorch_vision")
  expect_equal(m$params$n_classes, 2L)
  expect_equal(m$params$image_size, 224L)
})

test_that("pytorch_unet2d creates correct model", {
  m <- ds.flower.model.pytorch_unet2d()
  expect_s3_class(m, "dsflower_model")
  expect_equal(m$name, "pytorch_unet2d")
  expect_equal(m$framework, "pytorch_vision")
  expect_equal(m$params$n_classes, 1L)
  expect_equal(m$params$batch_size, 8L)
  expect_equal(m$params$image_size, 224L)
  expect_equal(m$params$base_channels, 64L)
})

test_that("pytorch_tcn creates correct model", {
  m <- ds.flower.model.pytorch_tcn()
  expect_s3_class(m, "dsflower_model")
  expect_equal(m$name, "pytorch_tcn")
  expect_equal(m$framework, "pytorch")
  expect_equal(m$params$n_channels, 1L)
  expect_equal(m$params$kernel_size, 3L)
  expect_equal(m$params$n_layers, 4L)
})

test_that("pytorch_lstm creates correct model", {
  m <- ds.flower.model.pytorch_lstm()
  expect_s3_class(m, "dsflower_model")
  expect_equal(m$name, "pytorch_lstm")
  expect_equal(m$framework, "pytorch")
  expect_equal(m$params$hidden_size, 64L)
  expect_equal(m$params$num_layers, 2L)
})
