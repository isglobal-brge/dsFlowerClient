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

  sub <- dsFlowerClient:::.emit_submission(ds.flower.model("densenet121"))
  expect_identical(sub$params$backbone, "densenet121")
  expect_true("ds.flower.model.pytorch_densenet121" %in%
                getNamespaceExports("dsFlowerClient"))
})

test_that("unsupported survival and segmentation models are not registered", {
  registered <- ds.flower.list_models()$name
  expect_false(any(c("pytorch_coxph", "pytorch_lognormal_aft",
                     "pytorch_cause_specific_cox", "pytorch_unet2d") %in%
                   registered))
  expect_error(ds.flower.model("pytorch_coxph"), "Unknown model")
  expect_error(ds.flower.model("pytorch_unet2d"), "Unknown model")
})

test_that("pytorch_tcn creates correct model", {
  m <- ds.flower.model.pytorch_tcn(input_shape = c(2L, 12L),
                                    channels = 16L, levels = 4L)
  expect_s3_class(m, "dsflower_model")
  expect_equal(m$name, "pytorch_tcn")
  expect_equal(m$framework, "pytorch")
  expect_identical(m$params$input_shape, c(2L, 12L))
  expect_identical(m$params$channels, 16L)
  expect_identical(m$params$levels, 4L)

  sub <- dsFlowerClient:::.emit_submission(m)
  expect_identical(unlist(sub$spec$layers[[1L]]$shape), c(2L, 12L))
  expect_equal(sum(vapply(sub$spec$layers, function(x) x$op == "conv1d",
                          logical(1))), 4L)
})

test_that("pytorch_lstm creates correct model", {
  m <- ds.flower.model.pytorch_lstm(
    n_tokens = 8L, n_features = 3L, hidden = 48L)
  expect_s3_class(m, "dsflower_model")
  expect_equal(m$name, "pytorch_lstm")
  expect_equal(m$framework, "pytorch")
  expect_identical(m$params$n_tokens, 8L)
  expect_identical(m$params$n_features, 3L)
  expect_identical(m$params$hidden, 48L)

  sub <- dsFlowerClient:::.emit_submission(m)
  expect_identical(sub$spec$nodes[[2L]]$op, "lstm")
  expect_identical(sub$spec$nodes[[2L]]$hidden, 48L)
})

test_that("multilabel constructor emits the registry target-width contract", {
  m <- ds.flower.model.pytorch_multilabel(n_labels = 3L)
  sub <- dsFlowerClient:::.emit_submission(m)

  expect_identical(m$params$num_labels, 3L)
  expect_identical(sub$params$num_labels, 3L)
  expect_identical(sub$loss, "multilabel_bce")
})
