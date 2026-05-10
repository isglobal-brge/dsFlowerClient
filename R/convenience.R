# Module: Convenience API
# String-friendly constructors and one-shot training helpers.

.dsflower_choice_key <- function(x) {
  if (!is.character(x) || length(x) != 1L || !nzchar(x)) {
    stop("Choice must be a single non-empty character value.", call. = FALSE)
  }
  x <- trimws(tolower(x))
  x <- gsub("^ds[.]flower[.]", "", x)
  x <- gsub("[.-]+", "_", x)
  x <- gsub("^(model|strategy|privacy|task)_", "", x)
  x <- gsub("__+", "_", x)
  x
}

.dsflower_choice <- function(x, choices, what) {
  key <- .dsflower_choice_key(x)
  if (!key %in% names(choices)) {
    stop(
      "Unknown ", what, " '", x, "'. Available values: ",
      paste(sort(unique(names(choices))), collapse = ", "),
      call. = FALSE
    )
  }
  choices[[key]]
}

.dsflower_call_constructor <- function(fun_name, args) {
  fun <- get(fun_name, envir = parent.env(environment()), mode = "function")
  do.call(fun, args)
}

#' Create a model spec by name
#'
#' Convenience wrapper around the concrete \code{ds.flower.model.*}
#' constructors. Existing \code{dsflower_model} objects are returned unchanged.
#'
#' @param name Character model name or a \code{dsflower_model} object.
#' @param ... Arguments passed to the selected concrete model constructor.
#' @return A \code{dsflower_model} object.
#' @export
ds.flower.model <- function(name = "sklearn_logreg", ...) {
  if (inherits(name, "dsflower_model")) {
    dots <- list(...)
    if (length(dots)) {
      stop("'...' cannot be used when 'name' is already a dsflower_model object.",
           call. = FALSE)
    }
    return(name)
  }

  if (!is.character(name)) {
    stop("'name' must be a dsflower_model object or character model name.",
         call. = FALSE)
  }

  choices <- c(
    logreg = "ds.flower.model.sklearn_logreg",
    logistic = "ds.flower.model.sklearn_logreg",
    logistic_regression = "ds.flower.model.sklearn_logreg",
    sklearn_logreg = "ds.flower.model.sklearn_logreg",
    ridge = "ds.flower.model.sklearn_ridge",
    sklearn_ridge = "ds.flower.model.sklearn_ridge",
    sgd = "ds.flower.model.sklearn_sgd",
    sklearn_sgd = "ds.flower.model.sklearn_sgd",
    svm = "ds.flower.model.sklearn_svm",
    linear_svm = "ds.flower.model.sklearn_svm",
    sklearn_svm = "ds.flower.model.sklearn_svm",
    elastic_net = "ds.flower.model.sklearn_elastic_net",
    elasticnet = "ds.flower.model.sklearn_elastic_net",
    sklearn_elastic_net = "ds.flower.model.sklearn_elastic_net",
    mlp = "ds.flower.model.pytorch_mlp",
    pytorch_mlp = "ds.flower.model.pytorch_mlp",
    torch_mlp = "ds.flower.model.pytorch_mlp",
    pytorch_logreg = "ds.flower.model.pytorch_logreg",
    torch_logreg = "ds.flower.model.pytorch_logreg",
    pytorch_linear_regression = "ds.flower.model.pytorch_linear_regression",
    linear_regression = "ds.flower.model.pytorch_linear_regression",
    cox = "ds.flower.model.pytorch_coxph",
    coxph = "ds.flower.model.pytorch_coxph",
    pytorch_coxph = "ds.flower.model.pytorch_coxph",
    multiclass = "ds.flower.model.pytorch_multiclass",
    pytorch_multiclass = "ds.flower.model.pytorch_multiclass",
    resnet18 = "ds.flower.model.pytorch_resnet18",
    pytorch_resnet18 = "ds.flower.model.pytorch_resnet18",
    densenet121 = "ds.flower.model.pytorch_densenet121",
    pytorch_densenet121 = "ds.flower.model.pytorch_densenet121",
    unet = "ds.flower.model.pytorch_unet2d",
    unet2d = "ds.flower.model.pytorch_unet2d",
    pytorch_unet2d = "ds.flower.model.pytorch_unet2d",
    tcn = "ds.flower.model.pytorch_tcn",
    pytorch_tcn = "ds.flower.model.pytorch_tcn",
    lstm = "ds.flower.model.pytorch_lstm",
    pytorch_lstm = "ds.flower.model.pytorch_lstm",
    xgb = "ds.flower.model.xgboost",
    xgboost = "ds.flower.model.xgboost",
    poisson = "ds.flower.model.pytorch_poisson",
    pytorch_poisson = "ds.flower.model.pytorch_poisson",
    multilabel = "ds.flower.model.pytorch_multilabel",
    pytorch_multilabel = "ds.flower.model.pytorch_multilabel",
    aft = "ds.flower.model.pytorch_lognormal_aft",
    lognormal_aft = "ds.flower.model.pytorch_lognormal_aft",
    pytorch_lognormal_aft = "ds.flower.model.pytorch_lognormal_aft",
    cause_specific = "ds.flower.model.pytorch_cause_specific_cox",
    cause_specific_cox = "ds.flower.model.pytorch_cause_specific_cox",
    pytorch_cause_specific_cox = "ds.flower.model.pytorch_cause_specific_cox"
  )

  .dsflower_call_constructor(.dsflower_choice(name, choices, "model"), list(...))
}

#' Create a strategy spec by name
#'
#' Convenience wrapper around the concrete \code{ds.flower.strategy.*}
#' constructors. Existing \code{dsflower_strategy} objects are returned
#' unchanged.
#'
#' @param name Character strategy name or a \code{dsflower_strategy} object.
#' @param ... Arguments passed to the selected concrete strategy constructor.
#' @return A \code{dsflower_strategy} object.
#' @export
ds.flower.strategy <- function(name = "fedavg", ...) {
  if (inherits(name, "dsflower_strategy")) {
    dots <- list(...)
    if (length(dots)) {
      stop("'...' cannot be used when 'name' is already a dsflower_strategy object.",
           call. = FALSE)
    }
    return(name)
  }

  if (!is.character(name)) {
    stop("'name' must be a dsflower_strategy object or character strategy name.",
         call. = FALSE)
  }

  choices <- c(
    avg = "ds.flower.strategy.fedavg",
    fedavg = "ds.flower.strategy.fedavg",
    fed_average = "ds.flower.strategy.fedavg",
    fedprox = "ds.flower.strategy.fedprox",
    prox = "ds.flower.strategy.fedprox",
    fedadam = "ds.flower.strategy.fedadam",
    adam = "ds.flower.strategy.fedadam",
    fedadagrad = "ds.flower.strategy.fedadagrad",
    adagrad = "ds.flower.strategy.fedadagrad",
    fedbn = "ds.flower.strategy.fedbn"
  )

  .dsflower_call_constructor(.dsflower_choice(name, choices, "strategy"), list(...))
}

#' Create a task spec by name
#'
#' Convenience wrapper around the concrete \code{ds.flower.task.*}
#' constructors. Existing \code{dsflower_task} objects are returned unchanged.
#'
#' @param name Character task name or a \code{dsflower_task} object.
#' @return A \code{dsflower_task} object.
#' @export
ds.flower.task <- function(name = "classification") {
  if (inherits(name, "dsflower_task")) return(name)
  if (!is.character(name)) {
    stop("'name' must be a dsflower_task object or character task name.",
         call. = FALSE)
  }

  choices <- c(
    classification = "ds.flower.task.classification",
    class = "ds.flower.task.classification",
    regression = "ds.flower.task.regression",
    survival = "ds.flower.task.survival",
    segmentation = "ds.flower.task.segmentation"
  )

  .dsflower_call_constructor(.dsflower_choice(name, choices, "task"), list())
}

#' Create an automatic privacy spec
#'
#' The automatic privacy policy is resolved at run time after server
#' capabilities are inspected. It chooses \code{prefer} when all connected
#' servers report Secure Aggregation support; otherwise it chooses
#' \code{fallback}. Explicit privacy specs remain available for production
#' deployments that require a fixed policy.
#'
#' @param prefer Character privacy profile to use when SecAgg is available.
#' @param fallback Character privacy profile to use otherwise.
#' @return A \code{dsflower_privacy} object with mode = "auto".
#' @export
ds.flower.privacy.auto <- function(prefer = "clinical_default",
                                   fallback = "trusted_internal") {
  obj <- list(
    mode = "auto",
    params = list(prefer = prefer, fallback = fallback)
  )
  class(obj) <- c("dsflower_privacy_auto", "dsflower_privacy")
  obj
}

#' Create a privacy spec by name
#'
#' Convenience wrapper around the concrete \code{ds.flower.privacy.*}
#' constructors. Existing \code{dsflower_privacy} objects are returned
#' unchanged.
#'
#' @param name Character privacy profile name or a \code{dsflower_privacy}
#'   object.
#' @param ... Arguments passed to the selected concrete privacy constructor.
#' @return A \code{dsflower_privacy} object.
#' @export
ds.flower.privacy <- function(name = "clinical_default", ...) {
  if (inherits(name, "dsflower_privacy")) {
    dots <- list(...)
    if (length(dots)) {
      stop("'...' cannot be used when 'name' is already a dsflower_privacy object.",
           call. = FALSE)
    }
    return(name)
  }

  if (!is.character(name)) {
    stop("'name' must be a dsflower_privacy object or character privacy name.",
         call. = FALSE)
  }

  choices <- c(
    auto = "ds.flower.privacy.auto",
    sandbox = "ds.flower.privacy.sandbox_open",
    sandbox_open = "ds.flower.privacy.sandbox_open",
    open = "ds.flower.privacy.sandbox_open",
    trusted = "ds.flower.privacy.trusted_internal",
    trusted_internal = "ds.flower.privacy.trusted_internal",
    consortium = "ds.flower.privacy.consortium_internal",
    consortium_internal = "ds.flower.privacy.consortium_internal",
    clinical = "ds.flower.privacy.clinical_default",
    clinical_default = "ds.flower.privacy.clinical_default",
    default = "ds.flower.privacy.clinical_default",
    hardened = "ds.flower.privacy.clinical_hardened",
    clinical_hardened = "ds.flower.privacy.clinical_hardened",
    update_noise = "ds.flower.privacy.clinical_update_noise",
    clinical_update_noise = "ds.flower.privacy.clinical_update_noise",
    dp = "ds.flower.privacy.high_sensitivity_dp",
    high_sensitivity = "ds.flower.privacy.high_sensitivity_dp",
    high_sensitivity_dp = "ds.flower.privacy.high_sensitivity_dp",
    evaluation_only = "ds.flower.privacy.evaluation_only"
  )

  .dsflower_call_constructor(.dsflower_choice(name, choices, "privacy"), list(...))
}

.is_auto_privacy <- function(privacy) {
  inherits(privacy, "dsflower_privacy_auto") || identical(privacy$mode, "auto")
}

.resolve_auto_privacy <- function(privacy, caps = NULL, verbose = TRUE) {
  if (is.null(privacy)) privacy <- ds.flower.privacy.clinical_default()
  if (is.character(privacy)) privacy <- ds.flower.privacy(privacy)
  if (!inherits(privacy, "dsflower_privacy")) {
    stop("'privacy' must be a dsflower_privacy object or character privacy name.",
         call. = FALSE)
  }
  if (!.is_auto_privacy(privacy)) return(privacy)

  prefer <- privacy$params$prefer %||% "clinical_default"
  fallback <- privacy$params$fallback %||% "trusted_internal"
  has_caps <- !is.null(caps) && length(caps) > 0L
  supports_secagg <- has_caps && all(vapply(caps, function(c) {
    isTRUE(c$secure_aggregation_supported)
  }, logical(1)))

  resolved_name <- if (supports_secagg) prefer else fallback
  resolved <- ds.flower.privacy(resolved_name)

  if (isTRUE(verbose)) {
    if (supports_secagg) {
      message("Privacy auto resolved to '", resolved$mode,
              "' because all servers report Secure Aggregation support.")
    } else {
      unsupported <- if (has_caps) {
        names(caps)[!vapply(caps, function(c) {
          isTRUE(c$secure_aggregation_supported)
        }, logical(1))]
      } else {
        character(0)
      }
      suffix <- if (length(unsupported)) {
        paste0(" (no SecAgg on: ", paste(unsupported, collapse = ", "), ")")
      } else {
        " (server capabilities unavailable)"
      }
      message("Privacy auto resolved to '", resolved$mode, "'", suffix, ".")
    }
  }

  resolved
}

#' Fit a federated model in one call
#'
#' High-level convenience API for the common workflow: connect to assigned
#' DataSHIELD data, build a recipe, run Flower, clean up server-side handles,
#' and return the trained run object. Advanced users can keep using
#' \code{ds.flower.connect()}, \code{ds.flower.recipe()}, and
#' \code{ds.flower.run()} directly.
#'
#' @param conns DSI connections object.
#' @param data Optional character data source resolved by \code{ds.flower.connect()}.
#' @param resource Optional Opal resource name.
#' @param symbol Optional assigned DataSHIELD symbol. Defaults to \code{"D"}
#'   when \code{data}, \code{resource}, and \code{symbol} are all NULL.
#' @param target Character target column name, or length-two vector for
#'   survival tasks.
#' @param features Character vector of feature column names, or NULL for
#'   template-specific auto handling.
#' @param model Character model name or \code{dsflower_model} object.
#' @param model_params Named list passed to \code{ds.flower.model()} when
#'   \code{model} is a character value.
#' @param strategy Character strategy name or \code{dsflower_strategy} object.
#' @param strategy_params Named list passed to \code{ds.flower.strategy()} when
#'   \code{strategy} is a character value.
#' @param privacy Character privacy name or \code{dsflower_privacy} object.
#'   The default \code{"auto"} uses the strongest practical profile detected
#'   from the connected servers.
#' @param privacy_params Named list passed to \code{ds.flower.privacy()} when
#'   \code{privacy} is a character value.
#' @param rounds Integer number of federated rounds.
#' @param task Optional character task name or \code{dsflower_task} object.
#' @param label_set Optional imaging label-set name.
#' @param masks Optional mask asset alias for segmentation.
#' @param evaluation_only Logical; if TRUE, blocks model release.
#' @param detached Logical; passed to \code{ds.flower.run()}.
#' @param verbose Logical; print training output.
#' @param disconnect Logical; remove server-side Flower handles on exit.
#' @param run_args Named list of additional arguments passed to
#'   \code{ds.flower.run()}.
#' @return A \code{dsflower_run} object.
#' @export
ds.flower.fit <- function(conns,
                          data = NULL,
                          resource = NULL,
                          symbol = NULL,
                          target,
                          features = NULL,
                          model = "sklearn_logreg",
                          model_params = list(),
                          strategy = "fedavg",
                          strategy_params = list(),
                          privacy = "auto",
                          privacy_params = list(),
                          rounds = 5L,
                          task = NULL,
                          label_set = NULL,
                          masks = NULL,
                          evaluation_only = FALSE,
                          detached = FALSE,
                          verbose = TRUE,
                          disconnect = TRUE,
                          run_args = list()) {
  if (missing(conns) || is.null(conns)) {
    stop("'conns' is required.", call. = FALSE)
  }
  if (missing(target) || is.null(target)) {
    stop("'target' is required.", call. = FALSE)
  }
  if (!is.list(model_params) ||
      (length(model_params) > 0L && is.null(names(model_params)))) {
    stop("'model_params' must be a named list.", call. = FALSE)
  }
  if (!is.list(strategy_params) ||
      (length(strategy_params) > 0L && is.null(names(strategy_params)))) {
    stop("'strategy_params' must be a named list.", call. = FALSE)
  }
  if (!is.list(privacy_params) ||
      (length(privacy_params) > 0L && is.null(names(privacy_params)))) {
    stop("'privacy_params' must be a named list.", call. = FALSE)
  }
  if (!is.list(run_args) ||
      (length(run_args) > 0L && is.null(names(run_args)))) {
    stop("'run_args' must be a named list.", call. = FALSE)
  }

  supplied_sources <- sum(!is.null(data), !is.null(resource), !is.null(symbol))
  if (supplied_sources == 0L) {
    symbol <- "D"
  } else if (supplied_sources > 1L) {
    stop("Provide only one of 'data', 'resource', or 'symbol'.", call. = FALSE)
  }

  model_spec <- if (inherits(model, "dsflower_model")) {
    if (length(model_params)) stop("'model_params' cannot be used with a model object.", call. = FALSE)
    model
  } else {
    do.call(ds.flower.model, c(list(model), model_params))
  }

  strategy_spec <- if (inherits(strategy, "dsflower_strategy")) {
    if (length(strategy_params)) stop("'strategy_params' cannot be used with a strategy object.", call. = FALSE)
    strategy
  } else {
    do.call(ds.flower.strategy, c(list(strategy), strategy_params))
  }

  privacy_spec <- if (inherits(privacy, "dsflower_privacy")) {
    if (length(privacy_params)) stop("'privacy_params' cannot be used with a privacy object.", call. = FALSE)
    privacy
  } else {
    do.call(ds.flower.privacy, c(list(privacy), privacy_params))
  }

  task_spec <- if (is.null(task)) NULL else ds.flower.task(task)

  recipe <- ds.flower.recipe(
    model = model_spec,
    strategy = strategy_spec,
    privacy = privacy_spec,
    task = task_spec,
    num_rounds = rounds,
    target = target,
    features = features,
    label_set = label_set,
    masks = masks,
    evaluation_only = evaluation_only
  )

  flower <- ds.flower.connect(conns, data = data, resource = resource,
                              symbol = symbol)
  if (isTRUE(disconnect)) {
    on.exit(try(ds.flower.disconnect(flower), silent = TRUE), add = TRUE)
  }

  do.call(
    ds.flower.run,
    c(list(flower = flower, recipe = recipe, detached = detached,
           verbose = verbose), run_args)
  )
}
