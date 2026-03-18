# Module: Prediction
# Apply trained federated models to new data.

#' Predict with a federated model
#'
#' Uses the global model weights from a training run or a saved model file
#' to generate predictions on new data. Supports all model types:
#' sklearn_logreg, sklearn_sgd, sklearn_ridge, and pytorch_mlp.
#'
#' @param model A \code{dsflower_run} object, or a list loaded from a saved
#'   model file (via \code{readRDS} or \code{jsonlite::fromJSON}).
#' @param newdata A data.frame or matrix with the same feature columns used
#'   during training.
#' @param type Character; \code{"response"} for predicted class (default),
#'   \code{"prob"} for probabilities.
#' @return A numeric vector of predictions.
#' @export
ds.flower.predict <- function(model, newdata, type = c("response", "prob")) {
  type <- match.arg(type)

  # Extract weights and model name from either dsflower_run or saved list
  if (inherits(model, "dsflower_run")) {
    weights <- model$weights
    model_name <- model$model
  } else if (is.list(model) && "weights" %in% names(model)) {
    weights <- model$weights
    model_name <- model$model
  } else {
    stop("'model' must be a dsflower_run or a saved model list.", call. = FALSE)
  }

  if (is.null(weights)) {
    stop("No weights available in this model.", call. = FALSE)
  }

  X <- as.matrix(newdata)

  if (model_name %in% c("sklearn_logreg", "sklearn_sgd")) {
    .predict_linear_classifier(weights, X, type)
  } else if (model_name == "sklearn_ridge") {
    .predict_ridge_classifier(weights, X, type)
  } else if (model_name == "pytorch_mlp") {
    .predict_mlp(weights, X, type)
  } else {
    stop("Unknown model type: ", model_name, call. = FALSE)
  }
}

#' @keywords internal
.predict_linear_classifier <- function(weights, X, type) {
  coef <- as.vector(weights$coef)
  intercept <- as.numeric(weights$intercept)
  logits <- as.vector(X %*% coef + intercept)
  probs <- 1 / (1 + exp(-logits))
  if (type == "prob") probs else as.integer(probs > 0.5)
}

#' @keywords internal
.predict_ridge_classifier <- function(weights, X, type) {
  coef <- as.vector(weights$coef)
  intercept <- as.numeric(weights$intercept)
  decision <- as.vector(X %*% coef + intercept)
  if (type == "prob") {
    # Ridge has no natural probability; use sigmoid of decision function
    1 / (1 + exp(-decision))
  } else {
    as.integer(decision > 0)
  }
}

#' @keywords internal
.predict_mlp <- function(weights, X, type) {
  # MLP weights come as alternating [weight, bias, weight, bias, ...]
  # Architecture: Linear -> ReLU -> ... -> Linear (output)
  param_names <- names(weights)
  if (is.null(param_names)) {
    # Unnamed: assume alternating weight/bias pairs
    n_params <- length(weights)
    param_names <- character(n_params)
    for (i in seq_len(n_params)) {
      param_names[i] <- if (i %% 2 == 1) "weight" else "bias"
    }
  }

  # Forward pass
  h <- X
  n_layers <- length(weights) %/% 2
  for (i in seq_len(n_layers)) {
    W <- weights[[(i - 1) * 2 + 1]]  # weight matrix
    b <- weights[[(i - 1) * 2 + 2]]  # bias vector

    if (is.matrix(W)) {
      h <- h %*% t(W)  # PyTorch stores weights as (out, in)
    } else {
      h <- h * as.vector(W)
    }
    h <- sweep(h, 2, as.vector(b), "+")

    # ReLU for all layers except the last
    if (i < n_layers) {
      h <- pmax(h, 0)
    }
  }

  logits <- as.vector(h)
  probs <- 1 / (1 + exp(-logits))
  if (type == "prob") probs else as.integer(probs > 0.5)
}
