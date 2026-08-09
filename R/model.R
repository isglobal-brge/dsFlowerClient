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
