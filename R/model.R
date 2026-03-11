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
  obj <- list(
    name      = "pytorch_mlp",
    framework = "pytorch",
    template  = "pytorch_mlp",
    params    = list(
      hidden_layers = as.integer(hidden_layers),
      learning_rate = learning_rate,
      batch_size    = as.integer(batch_size),
      local_epochs  = as.integer(local_epochs)
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
