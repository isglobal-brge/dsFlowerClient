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
    "Invalid parameter type"
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
  expect_equal(m$params$n_bins, 32L)
  expect_equal(m$params$objective, "binary:logistic")
})

test_that("exported model constructors never truncate public parameters", {
  expect_error(ds.flower.model.xgboost(n_trees = 3.9), "exact finite integer")
  expect_error(
    ds.flower.model.pytorch_multiclass(n_classes = 3.9),
    "exact finite integer")
  expect_error(
    ds.flower.model.pytorch_tcn(
      input_shape = c(2.9, 8.9), levels = 2L),
    "exact finite integers")
  expect_error(
    ds.flower.model.pytorch_resnet18(volumetric = 1),
    "TRUE or FALSE")
})

test_that("first-party models declare typed parameter schemas", {
  registry <- get(".dsflower_models", envir = asNamespace("dsFlowerClient"))
  for (name in ls(registry, sorted = TRUE)) {
    model <- registry[[name]]
    expect_named(model$parameter_types)
    expect_length(setdiff(names(model$defaults), names(model$parameter_types)), 0L)
  }
})

test_that("extension models must declare their complete parameter schema", {
  expect_error(
    ds.flower.register_model(
      "missing_schema", "neural",
      generate = function(params) list(kind = "sequential", layers = list()),
      loss = "bce_logits"),
    "parameter_types.*required"
  )

  registry <- get(".dsflower_models", envir = asNamespace("dsFlowerClient"))
  on.exit(rm(list = "no_params", envir = registry), add = TRUE)
  expect_silent(ds.flower.register_model(
    "no_params", "neural",
    generate = function(params) list(
      kind = "sequential",
      layers = list(list(op = "linear", out = "@out"))),
    loss = "bce_logits", parameter_types = character()))
  expect_s3_class(ds.flower.model("no_params"), "dsflower_model")

  expect_error(ds.flower.register_model(
    "missing_loss", "neural", generate = function(params) list(),
    parameter_types = character()), "must declare")
  expect_error(ds.flower.register_model(
    "ignored_loss", "trees", generate = function(params) list(), loss = "mse",
    parameter_types = character()), "loss = NULL")
  expect_error(ds.flower.register_model(
    "My-Model", "neural", generate = function(params) list(),
    loss = "bce_logits", parameter_types = character()), "canonical lowercase")
  expect_error(ds.flower.register_model(
    "logreg", "neural", generate = function(params) list(),
    loss = "bce_logits", parameter_types = character()), "built-in model alias")
})

test_that("extension optimizer and scheduler schemas may expose safe subsets", {
  registry <- get(".dsflower_models", envir = asNamespace("dsFlowerClient"))
  on.exit(rm(list = "partial_training_controls", envir = registry), add = TRUE)
  ds.flower.register_model(
    "partial_training_controls", "neural",
    generate = function(params) list(
      kind = "sequential",
      layers = list(list(op = "linear", "in" = "@in", out = "@out"))),
    loss = "bce_logits", parameter_types = c(
      optimizer = "character", scheduler = "character"))

  model <- ds.flower.model(
    "partial_training_controls", optimizer = "adam", scheduler = "cosine")
  expect_identical(names(model$params), c("optimizer", "scheduler"))
  expect_error(
    ds.flower.model("partial_training_controls", optimizer = "evil"),
    "Optimizer must be one of")
  expect_error(
    ds.flower.model("partial_training_controls", scheduler = "evil"),
    "Scheduler must be one of")
  expect_error(
    ds.flower.register_model(
      "orphan_optimizer_control", "neural",
      generate = function(params) list(kind = "sequential", layers = list()),
      loss = "bce_logits", parameter_types = c(beta1 = "number")),
    "require a declared 'optimizer'")
})

test_that("model parameter contracts are discoverable", {
  neural <- ds.flower.model_parameters("logreg")
  expect_true(all(c("optimizer", "scheduler", "learning_rate") %in%
                    neural$parameter))
  optimizer <- neural[neural$parameter == "optimizer", ]
  expect_setequal(optimizer$choices[[1L]], c("sgd", "adam", "adamw", "rmsprop"))
  expect_identical(optimizer$default[[1L]], "sgd")
  expect_identical(
    neural$default[[which(neural$parameter == "batch_size")]], 32L)
  expect_identical(
    neural$default[[which(neural$parameter == "local_epochs")]], 1L)

  trees <- ds.flower.model_parameters("xgb")
  objective <- trees[trees$parameter == "objective", ]
  expect_setequal(
    objective$choices[[1L]], c("binary:logistic", "reg:squarederror"))
  learning <- trees[trees$parameter == "learning_rate", ]
  expect_identical(learning$aliases, "eta")

  required <- ds.flower.model_parameters("pytorch_cnn")
  input_shape <- required[required$parameter == "input_shape", ]
  expect_true(input_shape$required)
  expect_null(input_shape$default[[1L]])
  expect_identical(
    required$default[[which(required$parameter == "channels")]],
    c(8L, 16L))
  expect_identical(
    ds.flower.model_parameters("pytorch_transformer")$default[[
      which(ds.flower.model_parameters("pytorch_transformer")$parameter == "d_ff")]],
    32L)
})

test_that("generic model parameters are canonicalized and validated", {
  multiclass <- ds.flower.model("pytorch_multiclass")
  expect_identical(multiclass$params$n_classes, 3L)

  trees <- ds.flower.model("xgboost", eta = 0.15)
  expect_equal(trees$params$learning_rate, 0.15)
  expect_false("eta" %in% names(trees$params))
  expect_identical(trees$params$n_bins, 32L)

  multilabel <- ds.flower.model("pytorch_multilabel", n_labels = 4L)
  expect_identical(multilabel$params$num_labels, 4L)
  expect_false("n_labels" %in% names(multilabel$params))

  optimized <- ds.flower.model(
    "pytorch_logreg", optimizer = "adam", beta1 = 0.8,
    scheduler = "cosine", scheduler_min_lr = 1e-4)
  expect_identical(optimized$params$optimizer, "adam")
  expect_equal(optimized$params$beta1, 0.8)
  expect_identical(optimized$params$scheduler, "cosine")

  overridden <- ds.flower.model("pytorch_logreg")
  registered <- dsFlowerClient:::.dsflower_get_model(overridden$name)
  registered$defaults <- overridden$params
  overridden$params <- dsFlowerClient:::.dsflower_resolve_model_params(
    registered, list(optimizer = "adam", scheduler = "step"))
  expect_identical(overridden$params$optimizer, "adam")
  expect_false(any(c("momentum", "nesterov") %in%
                     names(overridden$params)))
  expect_identical(overridden$params$scheduler, "step")
  expect_equal(overridden$params$scheduler_step_size, 1L)
  expect_error(ds.flower.model("pytorch_logreg", optimizer = "lbfgs"),
               "Optimizer must be one of")
  expect_error(
    ds.flower.model("pytorch_logreg", epsilon = 100),
    "Unknown parameter.*epsilon"
  )
  expect_error(
    ds.flower.model("xgboost", eta = 0.2, learning_rate = 0.1),
    "both alias 'eta' and canonical"
  )
  expect_error(
    ds.flower.model("pytorch_tcn", input_shape = c(2L, 12L), levels = 1.5),
    "levels.*positive_integer"
  )
})

test_that("extension schemas cannot impersonate node privacy controls", {
  expect_error(
    ds.flower.register_model(
      "bad_privacy_extension", "neural",
      generate = function(p) list(kind = "sequential", layers = list(
        list(op = "linear", out = "@out"))),
      loss = "bce_logits", parameter_types = c(epsilon = "number")),
    "node-owned privacy parameter")
  expect_error(
    ds.flower.register_model(
      "bad_default_extension", "neural",
      generate = function(p) list(kind = "sequential", layers = list(
        list(op = "linear", out = "@out"))),
      loss = "bce_logits", defaults = list(NULL),
      parameter_types = c(x = "number")),
    "uniquely named list")
})

test_that("first-party parameter caps fail before execution", {
  expect_error(ds.flower.model("pytorch_logreg", learning_rate = 11),
               "learning_rate.*10")
  expect_error(ds.flower.model("pytorch_mlp", hidden_layers = 9000L),
               "hidden_layers")
  expect_error(
    ds.flower.model("pytorch_cnn", input_shape = c(1L, 32L)),
    "channels, height, width")
  expect_error(
    ds.flower.model("pytorch_tcn", input_shape = c(1L, 16L, 16L)),
    "channels, sequence_length")
  expect_error(
    ds.flower.model("pytorch_resnet", input_shape = c(1L, 32L)),
    "channels, height, width")
  expect_error(
    ds.flower.model("pytorch_tcn", input_shape = c(1L, 4097L)),
    "input_shape")
  expect_error(
    ds.flower.model("pytorch_tcn", input_shape = c(1L, 32L), levels = 14L),
    "levels")
  expect_error(
    ds.flower.model(
      "pytorch_cnn", input_shape = c(1L, 32L, 32L), channels = rep(8L, 21L)),
    "at most 20")
  expect_error(
    ds.flower.model(
      "pytorch_cnn", input_shape = c(1L, 4L, 4L), channels = c(8L, 16L, 32L)),
    "survive")
  expect_error(ds.flower.model("xgboost", max_depth = 7L),
               "max_depth.*6")
  expect_error(ds.flower.model("xgboost", n_trees = 201L),
               "n_trees.*200")
  expect_error(ds.flower.model("pytorch_multiclass", n_classes = 1L),
               "n_classes")
  expect_error(ds.flower.model("pytorch_logreg", beta2 = 1), "beta2")
  expect_error(ds.flower.model("pytorch_logreg", momentum = 1), "momentum")
  expect_error(ds.flower.model("pytorch_logreg", optimizer_eps = 0),
               "optimizer_eps")
  expect_error(ds.flower.model("pytorch_logreg", optimizer = "sgd", beta1 = .8),
               "does not use")
  expect_error(ds.flower.model("pytorch_logreg", optimizer = "adam", momentum = .2),
               "does not use")
  expect_error(ds.flower.model("pytorch_logreg", nesterov = TRUE),
               "positive SGD momentum")
  expect_error(ds.flower.model("pytorch_logreg", scheduler_gamma = .5),
               "Scheduler 'none' does not use")
  expect_error(ds.flower.model(
    "pytorch_logreg", scheduler = "cosine", scheduler_min_lr = 1),
    "cannot exceed")
  growing <- ds.flower.model(
    "pytorch_logreg", scheduler = "exponential", scheduler_gamma = 1.1)
  expect_equal(growing$params$scheduler_gamma, 1.1)
})

test_that("bounded DP-GBDT exposes only implemented objectives and controls", {
  m <- ds.flower.model(
    "xgboost", objective = "reg:squarederror",
    margin_bounds = c(-5, 5), gradient_clip = 2)
  expect_identical(m$params$objective, "reg:squarederror")
  expect_equal(m$params$margin_bounds, c(-5, 5))
  expect_equal(m$params$gradient_clip, 2)
  expect_error(ds.flower.model("xgboost", objective = "multi:softprob"),
               "expected one of")
  expect_error(ds.flower.model("xgboost", margin_bounds = c(1, 1)),
               "numeric_interval")
})

test_that("robust regression exposes and validates its applied loss parameter", {
  model <- ds.flower.model("huber", huber_delta = 2.5,
                           optimizer = "adam")
  expect_identical(model$name, "pytorch_huber")
  expect_equal(model$params$huber_delta, 2.5)
  expect_identical(model$loss, "huber")
  expect_error(ds.flower.model("huber", huber_delta = 0), "huber_delta")
})

test_that("only the selected loss parameter is serialized", {
  mse <- dsFlowerClient:::.neural_training_config(
    ds.flower.model("pytorch_linear_regression")$params, "mse")
  huber <- dsFlowerClient:::.neural_training_config(
    ds.flower.model("huber", huber_delta = 2.5)$params, "huber")
  negbin <- dsFlowerClient:::.neural_training_config(
    ds.flower.model("pytorch_negbin", nb_dispersion = 4)$params,
    "negbin_nll")

  expect_false(any(c("nb-dispersion", "gamma-shape", "huber-delta") %in%
                     names(mse)))
  expect_equal(huber[["huber-delta"]], 2.5)
  expect_false(any(c("nb-dispersion", "gamma-shape") %in% names(huber)))
  expect_equal(negbin[["nb-dispersion"]], 4)
  expect_false(any(c("gamma-shape", "huber-delta") %in% names(negbin)))
})

test_that("retired XGBoost compatibility parameters fail explicitly", {
  expect_error(ds.flower.model.xgboost(num_class = 3L), "num_class")
  expect_error(ds.flower.model.xgboost(min_child_weight = 2),
               "min_child_weight")
  expect_error(ds.flower.model.xgboost(fixed_bin_range = 8),
               "fixed_bin_range")
  expect_error(ds.flower.model.xgboost(batch_multiclass = TRUE),
               "batch_multiclass")
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
