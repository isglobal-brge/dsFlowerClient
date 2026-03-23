# Module: Model Specs
# Model specification objects for federated learning.

#' Create a scikit-learn Logistic Regression model spec
#'
#' @param penalty Character; regularization penalty ("l2", "l1", "none").
#' @param C Numeric; inverse regularization strength.
#' @param max_iter Integer; maximum iterations.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.sklearn_logreg <- function(penalty = "l2", C = 1.0,
                                            max_iter = 100L) {
  obj <- list(
    name      = "sklearn_logreg",
    framework = "sklearn",
    template  = "sklearn_logreg",
    params    = list(penalty = penalty, C = C, max_iter = as.integer(max_iter))
  )
  class(obj) <- "dsflower_model"
  obj
}

#' Create a scikit-learn Ridge Classifier model spec
#'
#' @param alpha Numeric; regularization strength.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.sklearn_ridge <- function(alpha = 1.0) {
  obj <- list(
    name      = "sklearn_ridge",
    framework = "sklearn",
    template  = "sklearn_ridge",
    params    = list(alpha = alpha)
  )
  class(obj) <- "dsflower_model"
  obj
}

#' Create a scikit-learn SGD Classifier model spec
#'
#' @param loss Character; loss function ("log_loss", "hinge", "modified_huber").
#' @param alpha Numeric; regularization constant.
#' @param lr_schedule Character; learning rate schedule ("optimal", "constant", "invscaling").
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.sklearn_sgd <- function(loss = "log_loss", alpha = 0.0001,
                                         lr_schedule = "optimal") {
  obj <- list(
    name      = "sklearn_sgd",
    framework = "sklearn",
    template  = "sklearn_sgd",
    params    = list(loss = loss, alpha = alpha, lr_schedule = lr_schedule)
  )
  class(obj) <- "dsflower_model"
  obj
}

#' Create a Linear SVM model spec
#'
#' Convenience constructor for a linear Support Vector Machine. Internally
#' uses the \code{sklearn_sgd} template with \code{loss = "hinge"}, which
#' is mathematically equivalent to \code{sklearn.svm.LinearSVC}.
#'
#' Only linear SVMs are supported in federated learning because kernel SVMs
#' require the full pairwise kernel matrix (all data in one place).
#'
#' @param alpha Numeric; regularization constant (analogous to 1/C in SVC).
#'   Smaller values = less regularization.
#' @param lr_schedule Character; learning rate schedule.
#' @return A \code{dsflower_model} S3 object using the sklearn_sgd template.
#' @export
ds.flower.model.sklearn_svm <- function(alpha = 0.0001,
                                         lr_schedule = "optimal") {
  obj <- list(
    name      = "sklearn_svm",
    framework = "sklearn",
    template  = "sklearn_sgd",
    params    = list(loss = "hinge", alpha = alpha, lr_schedule = lr_schedule)
  )
  class(obj) <- "dsflower_model"
  obj
}

#' Create an Elastic Net model spec
#'
#' Convenience constructor for Elastic Net regularization (L1 + L2 penalty).
#' Internally uses the \code{sklearn_sgd} template with
#' \code{penalty = "elasticnet"}. Useful for variable selection in
#' high-dimensional data (genomics, radiomics).
#'
#' @param l1_ratio Numeric; mixing parameter (0 = L2 only, 1 = L1 only).
#'   Default 0.15.
#' @param alpha Numeric; regularization constant. Default 0.0001.
#' @param loss Character; loss function. Default "log_loss" (logistic).
#' @return A \code{dsflower_model} S3 object using the sklearn_sgd template.
#' @export
ds.flower.model.sklearn_elastic_net <- function(l1_ratio = 0.15,
                                                 alpha = 0.0001,
                                                 loss = "log_loss") {
  if (l1_ratio < 0 || l1_ratio > 1) {
    stop("l1_ratio must be between 0 and 1.", call. = FALSE)
  }
  obj <- list(
    name      = "sklearn_elastic_net",
    framework = "sklearn",
    template  = "sklearn_sgd",
    params    = list(loss = loss, alpha = alpha, penalty = "elasticnet",
                     l1_ratio = l1_ratio)
  )
  class(obj) <- "dsflower_model"
  obj
}

#' Create a PyTorch MLP model spec
#'
#' @param hidden_layers Integer vector; hidden layer sizes.
#' @param learning_rate Numeric; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs per round.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_mlp <- function(hidden_layers = c(64L, 32L),
                                         learning_rate = 0.01,
                                         batch_size = 32L,
                                         local_epochs = 1L) {
  # Store as comma-separated string for TOML compatibility
  hl_str <- paste(as.integer(hidden_layers), collapse = ",")
  obj <- list(
    name      = "pytorch_mlp",
    framework = "pytorch",
    template  = "pytorch_mlp",
    params    = list(
      hidden_layers = hl_str,
      learning_rate = learning_rate,
      batch_size    = as.integer(batch_size),
      local_epochs  = as.integer(local_epochs)
    )
  )
  class(obj) <- "dsflower_model"
  obj
}

#' Create a PyTorch Logistic Regression model spec
#'
#' DP-SGD capable linear classifier for binary classification.
#'
#' @param learning_rate Numeric; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs per round.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_logreg <- function(learning_rate = 0.01,
                                            batch_size = 32L,
                                            local_epochs = 1L) {
  obj <- list(
    name      = "pytorch_logreg",
    framework = "pytorch",
    template  = "pytorch_logreg",
    params    = list(
      learning_rate = learning_rate,
      batch_size    = as.integer(batch_size),
      local_epochs  = as.integer(local_epochs)
    )
  )
  class(obj) <- "dsflower_model"
  obj
}

#' Create a PyTorch Linear Regression model spec
#'
#' Continuous outcome prediction (MSE loss).
#'
#' @param learning_rate Numeric; learning rate.
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
      batch_size    = as.integer(batch_size),
      local_epochs  = as.integer(local_epochs)
    )
  )
  class(obj) <- "dsflower_model"
  obj
}

#' Create a PyTorch Cox Proportional Hazards model spec
#'
#' Survival/time-to-event analysis with partial likelihood loss.
#'
#' @param learning_rate Numeric; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs per round.
#' @return A \code{dsflower_model} S3 object.
#' @export
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
#' @param learning_rate Numeric; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs per round.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_multiclass <- function(hidden_layers = integer(0),
                                                n_classes = 3L,
                                                learning_rate = 0.01,
                                                batch_size = 32L,
                                                local_epochs = 1L) {
  # Store as comma-separated string for TOML compatibility
  hl_str <- if (length(hidden_layers) > 0) {
    paste(as.integer(hidden_layers), collapse = ",")
  } else {
    ""
  }
  obj <- list(
    name      = "pytorch_multiclass",
    framework = "pytorch",
    template  = "pytorch_multiclass",
    params    = list(
      hidden_layers = hl_str,
      n_classes     = as.integer(n_classes),
      learning_rate = learning_rate,
      batch_size    = as.integer(batch_size),
      local_epochs  = as.integer(local_epochs)
    )
  )
  class(obj) <- "dsflower_model"
  obj
}

#' Create an XGBoost Tabular model spec (RESEARCH-ONLY)
#'
#' Gradient-boosted trees for structured data using tree bagging aggregation.
#'
#' WARNING: This template is NOT available under the 'secure' or 'secure_dp'
#' privacy profiles. Tree bagging sends complete local trees to the SuperLink,
#' exposing individual client data structure (split thresholds, leaf values).
#' This is incompatible with DataSHIELD's privacy model where the researcher
#' must not access individual-level information.
#'
#' For secure XGBoost, use \code{ds.flower.model.xgboost_secure()} instead,
#' which implements histogram-based aggregation with SecAgg+.
#'
#' @param max_depth Integer; maximum tree depth.
#' @param eta Numeric; learning rate (shrinkage).
#' @param objective Character; XGBoost objective function.
#' @param local_rounds Integer; boosting rounds per FL round.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.xgboost_tabular <- function(max_depth = 6L,
                                             eta = 0.3,
                                             objective = "binary:logistic",
                                             local_rounds = 10L) {
  obj <- list(
    name      = "xgboost_tabular",
    framework = "xgboost",
    template  = "xgboost_tabular",
    params    = list(
      max_depth    = as.integer(max_depth),
      eta          = eta,
      objective    = objective,
      local_rounds = as.integer(local_rounds)
    )
  )
  class(obj) <- "dsflower_model"
  obj
}

#' Create a PyTorch ResNet-18 model spec
#'
#' Standard image classification backbone.
#'
#' @param n_classes Integer; number of output classes.
#' @param learning_rate Numeric; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs per round.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_resnet18 <- function(n_classes = 2L,
                                              learning_rate = 0.001,
                                              batch_size = 32L,
                                              local_epochs = 1L) {
  obj <- list(
    name      = "pytorch_resnet18",
    framework = "pytorch_vision",
    template  = "pytorch_resnet18",
    params    = list(
      n_classes     = as.integer(n_classes),
      learning_rate = learning_rate,
      batch_size    = as.integer(batch_size),
      local_epochs  = as.integer(local_epochs)
    )
  )
  class(obj) <- "dsflower_model"
  obj
}

#' Create a PyTorch DenseNet-121 model spec
#'
#' Medical imaging classification (chest X-ray, etc.).
#'
#' @param n_classes Integer; number of output classes.
#' @param learning_rate Numeric; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs per round.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_densenet121 <- function(n_classes = 2L,
                                                 learning_rate = 0.001,
                                                 batch_size = 32L,
                                                 local_epochs = 1L) {
  obj <- list(
    name      = "pytorch_densenet121",
    framework = "pytorch_vision",
    template  = "pytorch_densenet121",
    params    = list(
      n_classes     = as.integer(n_classes),
      learning_rate = learning_rate,
      batch_size    = as.integer(batch_size),
      local_epochs  = as.integer(local_epochs)
    )
  )
  class(obj) <- "dsflower_model"
  obj
}

#' Create a PyTorch U-Net 2D model spec
#'
#' Medical image segmentation (organs, tumors, lesions).
#'
#' @param n_classes Integer; number of segmentation classes.
#' @param learning_rate Numeric; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs per round.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_unet2d <- function(n_classes = 1L,
                                            learning_rate = 0.001,
                                            batch_size = 8L,
                                            local_epochs = 1L) {
  obj <- list(
    name      = "pytorch_unet2d",
    framework = "pytorch_vision",
    template  = "pytorch_unet2d",
    params    = list(
      n_classes     = as.integer(n_classes),
      learning_rate = learning_rate,
      batch_size    = as.integer(batch_size),
      local_epochs  = as.integer(local_epochs)
    )
  )
  class(obj) <- "dsflower_model"
  obj
}

#' Create a PyTorch TCN model spec
#'
#' Temporal Convolutional Network for time series classification.
#'
#' @param n_channels Integer; number of input channels.
#' @param kernel_size Integer; convolution kernel size.
#' @param n_layers Integer; number of TCN blocks.
#' @param learning_rate Numeric; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs per round.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_tcn <- function(n_channels = 1L,
                                         kernel_size = 3L,
                                         n_layers = 4L,
                                         learning_rate = 0.001,
                                         batch_size = 32L,
                                         local_epochs = 1L) {
  obj <- list(
    name      = "pytorch_tcn",
    framework = "pytorch",
    template  = "pytorch_tcn",
    params    = list(
      n_channels    = as.integer(n_channels),
      kernel_size   = as.integer(kernel_size),
      n_layers      = as.integer(n_layers),
      learning_rate = learning_rate,
      batch_size    = as.integer(batch_size),
      local_epochs  = as.integer(local_epochs)
    )
  )
  class(obj) <- "dsflower_model"
  obj
}

#' Create a PyTorch LSTM model spec
#'
#' LSTM for longitudinal EHR and sequential clinical data.
#'
#' @param hidden_size Integer; LSTM hidden state size.
#' @param num_layers Integer; number of LSTM layers.
#' @param learning_rate Numeric; learning rate.
#' @param batch_size Integer; batch size.
#' @param local_epochs Integer; local training epochs per round.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.pytorch_lstm <- function(hidden_size = 64L,
                                          num_layers = 2L,
                                          learning_rate = 0.001,
                                          batch_size = 32L,
                                          local_epochs = 1L) {
  obj <- list(
    name      = "pytorch_lstm",
    framework = "pytorch",
    template  = "pytorch_lstm",
    params    = list(
      hidden_size   = as.integer(hidden_size),
      num_layers    = as.integer(num_layers),
      learning_rate = learning_rate,
      batch_size    = as.integer(batch_size),
      local_epochs  = as.integer(local_epochs)
    )
  )
  class(obj) <- "dsflower_model"
  obj
}

#' Create a Secure XGBoost model spec (Histogram Protocol)
#'
#' Secure federated XGBoost using histogram-based aggregation
#' compatible with Flower SecAgg+. The server never sees individual
#' client gradients -- only aggregated histogram sums.
#'
#' @param n_trees Integer; number of boosting rounds.
#' @param max_depth Integer; maximum tree depth.
#' @param eta Numeric; learning rate (shrinkage).
#' @param reg_lambda Numeric; L2 regularization term.
#' @param n_bins Integer; number of histogram bins.
#' @param objective Character; XGBoost objective function.
#' @return A \code{dsflower_model} S3 object.
#' @export
ds.flower.model.xgboost_secure <- function(n_trees = 10L, max_depth = 3L,
                                            eta = 0.3, reg_lambda = 1.0,
                                            n_bins = 64L,
                                            objective = "binary:logistic") {
  obj <- list(
    name      = "xgboost_secure_horizontal",
    framework = "xgboost",
    template  = "xgboost_secure_horizontal",
    params    = list(
      n_trees    = as.integer(n_trees),
      max_depth  = as.integer(max_depth),
      eta        = eta,
      reg_lambda = reg_lambda,
      n_bins     = as.integer(n_bins),
      objective  = objective
    )
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
