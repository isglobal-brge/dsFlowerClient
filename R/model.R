# Module: Model Specs
# Model specification objects for federated learning.

.model_exact_integer <- function(value, name) {
  if (!is.numeric(value) || is.logical(value)) {
    stop("'", name, "' must be one exact finite integer.", call. = FALSE)
  }
  numeric <- suppressWarnings(as.numeric(value))
  if (length(numeric) != 1L || is.na(numeric) || !is.finite(numeric) ||
      numeric != floor(numeric) || abs(numeric) > .Machine$integer.max) {
    stop("'", name, "' must be one exact finite integer.", call. = FALSE)
  }
  as.integer(numeric)
}

.model_exact_integer_vector <- function(value, name) {
  if (!is.numeric(value) || is.logical(value)) {
    stop("'", name, "' must contain exact finite integers.", call. = FALSE)
  }
  numeric <- suppressWarnings(as.numeric(value))
  if (anyNA(numeric) || any(!is.finite(numeric)) ||
      any(numeric != floor(numeric)) ||
      any(abs(numeric) > .Machine$integer.max)) {
    stop("'", name, "' must contain exact finite integers.", call. = FALSE)
  }
  as.integer(numeric)
}

.model_exact_logical <- function(value, name) {
  if (!is.logical(value) || length(value) != 1L || is.na(value)) {
    stop("'", name, "' must be TRUE or FALSE.", call. = FALSE)
  }
  value
}

#' Create a PyTorch MLP model spec
#'
#' @param hidden_layers Integer vector; hidden layer sizes.
#' @param learning_rate Numeric in \code{(0, 10]}; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs per round.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_mlp <- function(hidden_layers = c(64L, 32L),
                                         learning_rate = 0.1,
                                         batch_size = 32L,
                                         local_epochs = 1L) {
  hidden_layers <- .model_exact_integer_vector(hidden_layers, "hidden_layers")
  obj <- list(
    name      = "pytorch_mlp",
    framework = "pytorch",
    params    = list(
      hidden_layers = hidden_layers,
      learning_rate = learning_rate,
      batch_size    = .model_exact_integer(batch_size, "batch_size"),
      local_epochs  = .model_exact_integer(local_epochs, "local_epochs")
    )
  )
  class(obj) <- "dsflower_model"
  ds.flower.model(obj)
}

#' Create a PyTorch Logistic Regression model spec
#'
#' DP-SGD capable linear classifier for binary classification.
#'
#' @param learning_rate Numeric in \code{(0, 10]}; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs per round.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_logreg <- function(learning_rate = 0.1,
                                            batch_size = 32L,
                                            local_epochs = 1L) {
  obj <- list(
    name      = "pytorch_logreg",
    framework = "pytorch",
    params    = list(
      learning_rate = learning_rate,
      batch_size    = .model_exact_integer(batch_size, "batch_size"),
      local_epochs  = .model_exact_integer(local_epochs, "local_epochs")
    )
  )
  class(obj) <- "dsflower_model"
  ds.flower.model(obj)
}

#' Create a PyTorch Linear Regression model spec
#'
#' Continuous outcome prediction (MSE loss).
#'
#' @param learning_rate Numeric in \code{(0, 10]}; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs per round.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_linear_regression <- function(learning_rate = 0.01,
                                                       batch_size = 32L,
                                                       local_epochs = 1L) {
  obj <- list(
    name      = "pytorch_linear_regression",
    framework = "pytorch",
    params    = list(
      learning_rate = learning_rate,
      batch_size    = .model_exact_integer(batch_size, "batch_size"),
      local_epochs  = .model_exact_integer(local_epochs, "local_epochs")
    )
  )
  class(obj) <- "dsflower_model"
  ds.flower.model(obj)
}

#' Create a native-tight XGBoost request spec
#'
#' This constructor exposes only the typed, data-only parameter profile used by
#' the trusted native XGBoost adapter. Feature bounds and complete public cuts
#' are supplied with the training request. Privacy epsilon, delta, clipping,
#' randomness, objectives, callbacks and I/O are node-owned and cannot be set
#' here. The constructor is implemented client-side; operational availability
#' is probed afresh on every selected node at submission time. Released ensembles
#' support private external or resubstitution validation.
#' Benchmark-driven defaults use 8 trees, depth 2 and learning rate 0.25 for
#' binary classification, and 5 trees, depth 2 and learning rate 0.30 for
#' regression. Explicit values always take precedence. Seven data-independent
#' public cuts per feature are a practical starting point; callers still supply
#' every cut explicitly, and dsFlower never derives them from private data.
#'
#' @param task Either \code{"binary"} or \code{"regression"}.
#' @param n_estimators Positive integer number of boosting rounds/trees. Defaults
#'   to 8 for binary classification and 5 for regression.
#' @param max_depth Positive integer tree depth, at most 30. Defaults to 2.
#' @param learning_rate Numeric in \code{(0,1]}. Defaults to 0.25 for binary
#'   classification and 0.30 for regression.
#' @param min_child_weight Non-negative minimum child Hessian weight.
#' @param min_split_loss Non-negative minimum split loss reduction.
#' @param reg_alpha Non-negative L1 leaf regularization.
#' @param reg_lambda Positive L2 leaf regularization, which keeps the leaf
#'   denominator strictly positive.
#' @param max_delta_step Positive maximum leaf-weight step. Together with the
#'   learning rate it fixes the public leaf-value bound.
#' @return A \code{dsflower_model} request object. It does not claim that the
#'   native backend is installed or enabled.
#' @export
ds.flower.model.xgboost <- function(task = c("binary", "regression"),
                                     n_estimators = 8L,
                                     max_depth = 2L,
                                     learning_rate = 0.25,
                                     min_child_weight = 1,
                                     min_split_loss = 0,
                                     reg_alpha = 0,
                                     reg_lambda = 1,
                                     max_delta_step = 1) {
  task <- match.arg(task)
  task_defaults <- .DSFLOWER_XGBOOST_TASK_DEFAULTS[[task]]
  if (missing(n_estimators)) {
    n_estimators <- task_defaults$num_boost_round
  }
  if (missing(max_depth)) max_depth <- task_defaults$max_depth
  if (missing(learning_rate)) learning_rate <- task_defaults$learning_rate
  params <- list(
    task = task,
    num_boost_round = .model_exact_integer(n_estimators, "n_estimators"),
    max_depth = .model_exact_integer(max_depth, "max_depth"),
    learning_rate = learning_rate,
    min_child_weight = min_child_weight,
    min_split_loss = min_split_loss,
    reg_alpha = reg_alpha,
    reg_lambda = reg_lambda,
    max_delta_step = max_delta_step)
  do.call(ds.flower.model, c(list(name = "xgboost"), params))
}

#' Create a PyTorch Multi-Class Classifier model spec
#'
#' Configurable MLP or linear classifier with CrossEntropyLoss.
#'
#' @param hidden_layers Integer vector; hidden layer sizes (empty for linear).
#' @param n_classes Integer; number of output classes.
#' @param learning_rate Numeric in \code{(0, 10]}; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs per round.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_multiclass <- function(hidden_layers = integer(0),
                                                n_classes = 3L,
                                                learning_rate = 0.1,
                                                batch_size = 32L,
                                                local_epochs = 1L) {
  hidden_layers <- .model_exact_integer_vector(hidden_layers, "hidden_layers")
  obj <- list(
    name      = "pytorch_multiclass",
    framework = "pytorch",
    params    = list(
      hidden_layers = hidden_layers,
      n_classes     = .model_exact_integer(n_classes, "n_classes"),
      learning_rate = learning_rate,
      batch_size    = .model_exact_integer(batch_size, "batch_size"),
      local_epochs  = .model_exact_integer(local_epochs, "local_epochs")
    )
  )
  class(obj) <- "dsflower_model"
  ds.flower.model(obj)
}


#' Create a PyTorch ResNet-18 model spec
#'
#' Standard image classification backbone.
#'
#' @param n_classes Integer; number of output classes.
#' @param learning_rate Numeric in \code{(0, 10]}; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs per round.
#' @param image_size Integer; square resize dimension before training.
#' @param volumetric Logical; if TRUE use a true-3D backbone (MONAI) for
#'   volumetric collections. Default FALSE: the 2D backbone auto-handles both 2D
#'   images and 3D volumes (via a representative slice), the plug-and-play path.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_resnet18 <- function(n_classes = 2L,
                                              learning_rate = 0.001,
                                              batch_size = 32L,
                                              local_epochs = 1L,
                                              image_size = 224L,
                                              volumetric = FALSE) {
  obj <- list(
    name      = "pytorch_resnet18",
    framework = "pytorch_vision",
    params    = list(
      n_classes     = .model_exact_integer(n_classes, "n_classes"),
      learning_rate = learning_rate,
      batch_size    = .model_exact_integer(batch_size, "batch_size"),
      local_epochs  = .model_exact_integer(local_epochs, "local_epochs"),
      image_size    = .model_exact_integer(image_size, "image_size"),
      volumetric    = .model_exact_logical(volumetric, "volumetric")
    )
  )
  class(obj) <- "dsflower_model"
  ds.flower.model(obj)
}

#' Create a PyTorch DenseNet-121 model spec
#'
#' Image classification backbone with densely connected convolutional blocks.
#'
#' @param n_classes Integer; number of output classes.
#' @param learning_rate Numeric in \code{(0, 10]}; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs per round.
#' @param image_size Integer; square resize dimension before training.
#' @param volumetric Logical; if TRUE use a true-3D backbone (MONAI) for
#'   volumetric collections. Default FALSE: the 2D backbone auto-handles both 2D
#'   images and 3D volumes (via a representative slice), the plug-and-play path.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_densenet121 <- function(n_classes = 2L,
                                                 learning_rate = 0.001,
                                                 batch_size = 32L,
                                                 local_epochs = 1L,
                                                 image_size = 224L,
                                                 volumetric = FALSE) {
  obj <- list(
    name      = "pytorch_densenet121",
    framework = "pytorch_vision",
    params    = list(
      n_classes     = .model_exact_integer(n_classes, "n_classes"),
      learning_rate = learning_rate,
      batch_size    = .model_exact_integer(batch_size, "batch_size"),
      local_epochs  = .model_exact_integer(local_epochs, "local_epochs"),
      image_size    = .model_exact_integer(image_size, "image_size"),
      volumetric    = .model_exact_logical(volumetric, "volumetric")
    )
  )
  class(obj) <- "dsflower_model"
  ds.flower.model(obj)
}

#' Create a PyTorch TCN model spec
#'
#' Temporal Convolutional Network for time series classification.
#'
#' @param input_shape Integer vector \code{c(channels, sequence_length)} whose
#'   product must equal the staged feature count.
#' @param channels Integer; number of hidden convolution channels.
#' @param levels Integer; number of dilated TCN blocks.
#' @param n_classes Integer; number of output classes.
#' @param learning_rate Numeric in \code{(0, 10]}; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs per round.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_tcn <- function(input_shape,
                                         channels = 8L,
                                         levels = 3L,
                                         n_classes = 2L,
                                         learning_rate = 0.001,
                                         batch_size = 32L,
                                         local_epochs = 1L) {
  input_shape <- .model_exact_integer_vector(input_shape, "input_shape")
  if (length(input_shape) != 2L || anyNA(input_shape) || any(input_shape < 1L)) {
    stop("'input_shape' must be c(channels, sequence_length) with positive integers.",
         call. = FALSE)
  }
  obj <- list(
    name      = "pytorch_tcn",
    framework = "pytorch",
    params    = list(
      input_shape   = input_shape,
      channels      = .model_exact_integer(channels, "channels"),
      levels        = .model_exact_integer(levels, "levels"),
      n_classes     = .model_exact_integer(n_classes, "n_classes"),
      learning_rate = learning_rate,
      batch_size    = .model_exact_integer(batch_size, "batch_size"),
      local_epochs  = .model_exact_integer(local_epochs, "local_epochs")
    )
  )
  class(obj) <- "dsflower_model"
  ds.flower.model(obj)
}

#' Create a PyTorch LSTM model spec
#'
#' LSTM for longitudinal EHR and sequential clinical data.
#'
#' @param n_tokens Integer; number of time points per sample.
#' @param n_features Integer; number of features per time point. The product
#'   \code{n_tokens * n_features} must equal the staged feature count.
#' @param hidden Integer; LSTM hidden state size.
#' @param n_classes Integer; number of output classes.
#' @param learning_rate Numeric in \code{(0, 10]}; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs per round.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_lstm <- function(n_tokens,
                                          n_features,
                                          hidden = 32L,
                                          n_classes = 2L,
                                          learning_rate = 0.001,
                                          batch_size = 32L,
                                          local_epochs = 1L) {
  n_tokens <- .model_exact_integer(n_tokens, "n_tokens")
  n_features <- .model_exact_integer(n_features, "n_features")
  if (length(n_tokens) != 1L || is.na(n_tokens) || n_tokens < 1L ||
      length(n_features) != 1L || is.na(n_features) || n_features < 1L) {
    stop("'n_tokens' and 'n_features' must be positive integers.", call. = FALSE)
  }
  obj <- list(
    name      = "pytorch_lstm",
    framework = "pytorch",
    params    = list(
      n_tokens      = n_tokens,
      n_features    = n_features,
      hidden        = .model_exact_integer(hidden, "hidden"),
      n_classes     = .model_exact_integer(n_classes, "n_classes"),
      learning_rate = learning_rate,
      batch_size    = .model_exact_integer(batch_size, "batch_size"),
      local_epochs  = .model_exact_integer(local_epochs, "local_epochs")
    )
  )
  class(obj) <- "dsflower_model"
  ds.flower.model(obj)
}

#' Create a Poisson Regression model spec
#'
#' Count data modeling (hospital events, readmissions, adverse events).
#' Uses Poisson NLL loss with log link.
#'
#' @param hidden_layers Integer vector; hidden layer sizes (empty for linear).
#' @param learning_rate Numeric in \code{(0, 10]}; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_poisson <- function(hidden_layers = integer(0),
                                             learning_rate = 0.01,
                                             batch_size = 32L,
                                             local_epochs = 1L) {
  hidden_layers <- .model_exact_integer_vector(hidden_layers, "hidden_layers")
  obj <- list(
    name      = "pytorch_poisson",
    framework = "pytorch",
    params    = list(hidden_layers = hidden_layers,
                     learning_rate = learning_rate,
                     batch_size = .model_exact_integer(batch_size, "batch_size"),
                     local_epochs = .model_exact_integer(local_epochs, "local_epochs"))
  )
  class(obj) <- "dsflower_model"
  ds.flower.model(obj)
}

#' Create a Multi-Label Classification model spec
#'
#' Multiple binary outcomes per sample (phenotyping, multi-endpoint).
#' Uses BCEWithLogitsLoss per label. Training requires exactly \code{n_labels}
#' distinct target columns; one public two-level vocabulary is applied to each.
#'
#' @param n_labels Integer; number of label columns.
#' @param hidden_layers Integer vector; hidden layer sizes.
#' @param learning_rate Numeric in \code{(0, 10]}; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_multilabel <- function(n_labels = 2L,
                                                hidden_layers = c(64L, 32L),
                                                learning_rate = 0.1,
                                                batch_size = 32L,
                                                local_epochs = 1L) {
  hidden_layers <- .model_exact_integer_vector(hidden_layers, "hidden_layers")
  obj <- list(
    name      = "pytorch_multilabel",
    framework = "pytorch",
    params    = list(num_labels = .model_exact_integer(n_labels, "n_labels"),
                     hidden_layers = hidden_layers,
                     learning_rate = learning_rate,
                     batch_size = .model_exact_integer(batch_size, "batch_size"),
                     local_epochs = .model_exact_integer(local_epochs, "local_epochs"))
  )
  class(obj) <- "dsflower_model"
  ds.flower.model(obj)
}

#' Print a dsflower_model
#' @param x A dsflower_model object.
#' @param ... Additional arguments (ignored).
#' @return Invisibly returns x.
#' @export
print.dsflower_model <- function(x, ...) {
  cat("dsflower_model:", x$name, "(", x$framework, ")\n")
  for (nm in names(x$params)) {
    cat("  ", nm, "=", .format_r_value(x$params[[nm]]), "\n")
  }
  invisible(x)
}
