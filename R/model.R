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
  if (!length(numeric) || anyNA(numeric) || any(!is.finite(numeric)) ||
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
  # Store as comma-separated string for TOML compatibility
  # Accept both integer vector c(64, 32) and string "64,32"
  if (is.character(hidden_layers) && length(hidden_layers) == 1) {
    hl_str <- hidden_layers
  } else if (!length(hidden_layers)) {
    hl_str <- ""
  } else {
    hl_str <- paste(.model_exact_integer_vector(
      hidden_layers, "hidden_layers"), collapse = ",")
  }
  obj <- list(
    name      = "pytorch_mlp",
    framework = "pytorch",
    template  = "pytorch_mlp",
    params    = list(
      hidden_layers = hl_str,
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
    template  = "pytorch_logreg",
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
    template  = "pytorch_linear_regression",
    params    = list(
      learning_rate = learning_rate,
      batch_size    = .model_exact_integer(batch_size, "batch_size"),
      local_epochs  = .model_exact_integer(local_epochs, "local_epochs")
    )
  )
  class(obj) <- "dsflower_model"
  ds.flower.model(obj)
}

#' Create a PyTorch Cox Proportional Hazards model spec
#'
#' Survival/time-to-event analysis with partial likelihood loss.
#'
#' @note NOT runnable on the Tier-1 harness in this release. The Cox partial
#'   likelihood couples samples through the risk set, so it has no per-sample
#'   gradient and is incompatible with DP-SGD. A HookApp does not automatically
#'   restore per-sample granularity for an arbitrary coupled loss. For DP
#'   survival, a future per-sample-decomposable AFT loss could be a declarative
#'   alternative, but no survival model is registered in this release.
#' @param learning_rate Numeric in \code{(0, 10]}; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs per round.
#' @return A \code{dsflower_model} S3 object.
#' @noRd
# NOTE: not exported -- survival models are defined but not yet registered/runnable in this
# build (see model_registry.R). Kept internal until the survival track is wired + tested.
ds.flower.model.pytorch_coxph <- function(learning_rate = 0.01,
                                           batch_size = 32L,
                                           local_epochs = 1L) {
  obj <- list(
    name      = "pytorch_coxph",
    framework = "pytorch",
    template  = "pytorch_coxph",
    params    = list(
      learning_rate = learning_rate,
      batch_size    = as.integer(batch_size),
      local_epochs  = as.integer(local_epochs)
    )
  )
  class(obj) <- "dsflower_model"
  obj
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
  # Store as comma-separated string for TOML compatibility
  if (is.character(hidden_layers) && length(hidden_layers) == 1) {
    hl_str <- hidden_layers
  } else if (length(hidden_layers) > 0) {
    hl_str <- paste(.model_exact_integer_vector(
      hidden_layers, "hidden_layers"), collapse = ",")
  } else {
    hl_str <- ""
  }
  obj <- list(
    name      = "pytorch_multiclass",
    framework = "pytorch",
    template  = "pytorch_multiclass",
    params    = list(
      hidden_layers = hl_str,
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
    template  = "pytorch_resnet18",
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
    template  = "pytorch_densenet121",
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

#' Create a PyTorch U-Net 2D model spec
#'
#' Medical image segmentation (organs, tumors, lesions).
#'
#' @note NOT runnable on the Tier-1 harness in this release. The vision harness
#'   does DP linear-probing on frozen-backbone features for CLASSIFICATION;
#'   dense per-pixel segmentation is rejected at app-build time. A custom HookApp
#'   provides only complete-update output perturbation when its execution gates
#'   hold, not declarative per-pixel/per-sample DP-SGD granularity.
#' @param n_classes Integer; number of segmentation classes.
#' @param learning_rate Numeric in \code{(0, 10]}; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs per round.
#' @param image_size Integer; square resize dimension before training.
#' @param base_channels Integer; number of channels in the first U-Net block.
#' @return A \code{dsflower_model} S3 object.
#' @noRd
# NOTE: not exported -- image/vision track not yet registered/runnable in this build.
ds.flower.model.pytorch_unet2d <- function(n_classes = 1L,
                                            learning_rate = 0.001,
                                            batch_size = 8L,
                                            local_epochs = 1L,
                                            image_size = 224L,
                                            base_channels = 64L) {
  obj <- list(
    name      = "pytorch_unet2d",
    framework = "pytorch_vision",
    template  = "pytorch_unet2d",
    params    = list(
      n_classes     = as.integer(n_classes),
      learning_rate = learning_rate,
      batch_size    = as.integer(batch_size),
      local_epochs  = as.integer(local_epochs),
      image_size    = as.integer(image_size),
      base_channels = as.integer(base_channels)
    )
  )
  class(obj) <- "dsflower_model"
  obj
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
    template  = "pytorch_tcn",
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
    template  = "pytorch_lstm",
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
#' @param hidden_layers Character; comma-separated hidden layer sizes (empty = linear).
#' @param learning_rate Numeric in \code{(0, 10]}; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_poisson <- function(hidden_layers = "",
                                             learning_rate = 0.01,
                                             batch_size = 32L,
                                             local_epochs = 1L) {
  obj <- list(
    name      = "pytorch_poisson",
    framework = "pytorch",
    template  = "pytorch_poisson",
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
#' @param hidden_layers Character; comma-separated hidden layer sizes.
#' @param learning_rate Numeric in \code{(0, 10]}; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_multilabel <- function(n_labels = 2L,
                                                hidden_layers = "64,32",
                                                learning_rate = 0.1,
                                                batch_size = 32L,
                                                local_epochs = 1L) {
  obj <- list(
    name      = "pytorch_multilabel",
    framework = "pytorch",
    template  = "pytorch_multilabel",
    params    = list(num_labels = .model_exact_integer(n_labels, "n_labels"),
                     hidden_layers = hidden_layers,
                     learning_rate = learning_rate,
                     batch_size = .model_exact_integer(batch_size, "batch_size"),
                     local_epochs = .model_exact_integer(local_epochs, "local_epochs"))
  )
  class(obj) <- "dsflower_model"
  ds.flower.model(obj)
}

#' Create a Log-Normal AFT survival model spec
#'
#' Log-normal Accelerated Failure Time parametric survival model.
#' Alternative to Cox PH when proportional hazards assumption fails.
#' Predicts log(T) = X*beta + sigma*epsilon, epsilon ~ N(0,1).
#'
#' NOTE: This is specifically log-normal AFT, not the full AFT family
#' (Weibull, log-logistic, etc.).
#'
#' @param learning_rate Numeric in \code{(0, 10]}; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs.
#' @return A \code{dsflower_model} S3 object.
#' @noRd
# NOTE: not exported -- survival track not yet registered/runnable in this build.
ds.flower.model.pytorch_lognormal_aft <- function(learning_rate = 0.01,
                                         batch_size = 32L,
                                         local_epochs = 1L) {
  obj <- list(
    name      = "pytorch_lognormal_aft",
    framework = "pytorch",
    template  = "pytorch_lognormal_aft",
    params    = list(learning_rate = learning_rate,
                     batch_size = as.integer(batch_size),
                     local_epochs = as.integer(local_epochs))
  )
  class(obj) <- "dsflower_model"
  obj
}

#' Create a Cause-Specific Cox model spec
#'
#' Multi-cause survival analysis via cause-specific hazard modeling.
#' Separate Cox partial likelihood per event cause.
#'
#' NOTE: This is cause-specific hazard modeling, NOT Fine-Gray
#' sub-distribution hazards. Each cause has its own Cox PH model.
#'
#' @param n_causes Integer; number of competing event types.
#' @param learning_rate Numeric in \code{(0, 10]}; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs.
#' @return A \code{dsflower_model} S3 object.
#' @noRd
# NOTE: not exported -- survival track not yet registered/runnable in this build.
ds.flower.model.pytorch_cause_specific_cox <- function(n_causes = 2L,
                                                      learning_rate = 0.01,
                                                      batch_size = 32L,
                                                      local_epochs = 1L) {
  obj <- list(
    name      = "pytorch_cause_specific_cox",
    framework = "pytorch",
    template  = "pytorch_cause_specific_cox",
    params    = list(n_causes = as.integer(n_causes),
                     learning_rate = learning_rate,
                     batch_size = as.integer(batch_size),
                     local_epochs = as.integer(local_epochs))
  )
  class(obj) <- "dsflower_model"
  obj
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
