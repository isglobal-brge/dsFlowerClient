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
ds.flower.model <- function(name = "pytorch_logreg", ...) {
  if (inherits(name, "dsflower_model")) {
    if (length(list(...))) {
      stop("'...' cannot be used when 'name' is already a dsflower_model object.",
           call. = FALSE)
    }
    return(name)
  }
  if (!is.character(name) || length(name) != 1L) {
    stop("'name' must be a dsflower_model object or a single model name.",
         call. = FALSE)
  }

  # Friendly aliases -> canonical registered names (torch + xgboost only; the
  # full model set lives in the client-side registry, extensible by derived
  # packages -- there is NO server-side catalog).
  aliases <- c(
    logreg = "pytorch_logreg", logistic = "pytorch_logreg",
    logistic_regression = "pytorch_logreg",
    mlp = "pytorch_mlp", torch_mlp = "pytorch_mlp", torch_logreg = "pytorch_logreg",
    multiclass = "pytorch_multiclass",
    linear_regression = "pytorch_linear_regression",
    poisson = "pytorch_poisson", multilabel = "pytorch_multilabel",
    resnet18 = "pytorch_resnet18", densenet121 = "pytorch_densenet121",
    xgb = "xgboost", gbdt = "xgboost")
  key <- .dsflower_choice_key(name)
  canonical <- if (key %in% names(aliases)) aliases[[key]] else key

  m <- .dsflower_get_model(canonical)   # registry lookup; errors listing available
  params <- utils::modifyList(m$defaults %||% list(), list(...))
  # Carry template/framework/loss too: the recipe task-inference, recipe print, run-record
  # metadata and prediction routing all read these. In this spec-based design template == name.
  structure(list(name = m$name, track = m$track, template = m$name,
                 framework = if (identical(m$track, "trees")) "xgboost" else "pytorch",
                 loss = m$loss, params = params),
            class = "dsflower_model")
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

  key <- .dsflower_choice_key(name)
  if (key %in% c("fedprox", "prox", "fedbn")) {
    stop("Strategy '", name, "' is not supported by the enforced-DP runtime. ",
         "Supported strategies: fedavg, fedadam, fedadagrad, fedyogi, fedavgm.",
         call. = FALSE)
  }

  choices <- c(
    avg = "ds.flower.strategy.fedavg",
    fedavg = "ds.flower.strategy.fedavg",
    fed_average = "ds.flower.strategy.fedavg",
    fedadam = "ds.flower.strategy.fedadam",
    adam = "ds.flower.strategy.fedadam",
    fedadagrad = "ds.flower.strategy.fedadagrad",
    adagrad = "ds.flower.strategy.fedadagrad",
    fedyogi = "ds.flower.strategy.fedyogi",
    yogi = "ds.flower.strategy.fedyogi",
    fedavgm = "ds.flower.strategy.fedavgm",
    avgm = "ds.flower.strategy.fedavgm"
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

  key <- .dsflower_choice_key(name)
  if (key %in% c("survival", "segmentation")) {
    stop("Task '", name, "' is not supported by the enforced-DP runtime. ",
         "Supported tasks: classification, regression.", call. = FALSE)
  }

  choices <- c(
    classification = "ds.flower.task.classification",
    class = "ds.flower.task.classification",
    regression = "ds.flower.task.regression"
  )

  .dsflower_call_constructor(.dsflower_choice(name, choices, "task"), list())
}


#' Fit a federated model in one call
#'
#' High-level convenience API for the common workflow: connect to assigned
#' DataSHIELD data, build a recipe, run Flower, clean up server-side handles,
#' and return the trained run object. Advanced users can call
#' \code{ds.flower.submit()} directly for finer control.
#'
#' @param conns DSI connections object.
#' @param data Optional character data source resolved by \code{ds.flower.connect()}.
#' @param resource Optional Opal resource name.
#' @param symbol Optional assigned DataSHIELD symbol. Defaults to \code{"D"}
#'   when \code{data}, \code{resource}, and \code{symbol} are all NULL.
#' @param target One target-column name, or exactly \code{num_labels} distinct
#'   target-column names for \code{pytorch_multilabel}.
#' @param features Character vector of feature column names, or NULL for
#'   template-specific auto handling.
#' @param model Character model name or \code{dsflower_model} object.
#' @param model_params Named list passed to \code{ds.flower.model()} when
#'   \code{model} is a character value.
#' @param torch_backend Character; requested node-side torch backend
#'   (\code{"auto"}, \code{"cpu"}, or a GPU selector).
#' @param strategy Character strategy name or \code{dsflower_strategy} object.
#' @param strategy_params Named list passed to \code{ds.flower.strategy()} when
#'   \code{strategy} is a character value.
#' @param rounds Integer number of federated rounds.
#' @param task Optional character task name or \code{dsflower_task} object.
#' @param label_set Optional imaging label-set name.
#' @param masks Reserved compatibility argument. Non-NULL values fail early
#'   because segmentation is not implemented by the enforced-DP runtime.
#' @param evaluation_only Logical; accepted for compatibility but NOT yet enforced by the
#'   enforced-DP path (the model is always released, DP-protected).
#' @param detached Logical; accepted for back-compat (unused by the enforced-DP path).
#' @param output_dir Optional character; parent directory the trained model is
#'   saved under (created if missing). Defaults to \code{./dsflower_output}.
#' @param output_name Optional character; folder/file name for this model inside
#'   \code{output_dir} (extension is added automatically). Defaults to an
#'   auto-generated model id.
#' @param silent Logical; when \code{TRUE}, suppress the training progress
#'   feedback (connection, per-round and completion messages).
#' @param verbose Logical; when \code{TRUE}, also print the raw flwr run log
#'   (debugging). The tidy per-round progress is shown regardless unless
#'   \code{silent = TRUE}.
#' @param disconnect Logical; accepted for compatibility. The submission pipeline always
#'   cleans up its server-side handles on exit regardless.
#' @param run_args Named list; accepted for back-compat (unused by the enforced-DP path).
#' @param feature_bounds Optional public feature bounds as
#'   \code{list(lower=..., upper=...)} in feature order. Appended to the signature
#'   for positional backward compatibility.
#' @param target_levels Optional ordered public classification label vocabulary.
#'   Non-numeric labels require it; missing or unknown values map to public code
#'   zero. Multilabel applies the same public two-level vocabulary independently
#'   to every target column.
#' @param target_bounds Required public \code{list(lower=..., upper=...)} for
#'   regression/count models.
#' @param allow_insecure_http Character vector of exact connection names allowed
#'   to use plaintext HTTP. Empty by default. This exception does not provide
#'   transport security; use it only behind an independently trusted network.
#' @return A \code{dsflower_run} object.
#' @export
ds.flower.fit <- function(conns,
                          data = NULL,
                          resource = NULL,
                          symbol = NULL,
                          target,
                          features = NULL,
                          model = "pytorch_logreg",
                          model_params = list(),
                          torch_backend = "auto",
                          strategy = "fedavg",
                          strategy_params = list(),
                          rounds = 5L,
                          task = NULL,
                          label_set = NULL,
                          masks = NULL,
                          output_dir = NULL,
                          output_name = NULL,
                          silent = FALSE,
                          evaluation_only = FALSE,
                          detached = FALSE,
                          verbose = FALSE,
                          disconnect = TRUE,
                          run_args = list(),
                          feature_bounds = NULL,
                          target_levels = NULL,
                          target_bounds = NULL,
                          allow_insecure_http = getOption(
                            "dsflower.dsi_allow_insecure_http", character())) {
  # Set the progress-verbosity option at the outermost entry point so it stays
  # active through every nested step, including the connection teardown that runs
  # in the submission pipeline's on.exit cleanup.
  old_opt <- options(dsflower.silent = isTRUE(silent))
  on.exit(options(old_opt), add = TRUE)

  if (missing(conns) || is.null(conns)) {
    stop("'conns' is required.", call. = FALSE)
  }
  if (missing(target) || is.null(target)) {
    stop("'target' is required.", call. = FALSE)
  }
  rounds <- suppressWarnings(as.integer(rounds))
  if (length(rounds) != 1L || is.na(rounds) || rounds < 1L) {
    stop("'rounds' must be a single positive integer.", call. = FALSE)
  }
  if (!is.list(model_params) ||
      (length(model_params) > 0L && is.null(names(model_params)))) {
    stop("'model_params' must be a named list.", call. = FALSE)
  }
  if (!is.list(strategy_params) ||
      (length(strategy_params) > 0L && is.null(names(strategy_params)))) {
    stop("'strategy_params' must be a named list.", call. = FALSE)
  }
  if (!is.list(run_args) ||
      (length(run_args) > 0L && is.null(names(run_args)))) {
    stop("'run_args' must be a named list.", call. = FALSE)
  }
  if (!is.null(masks)) {
    stop("'masks' requires segmentation, which is not supported by the ",
         "enforced-DP runtime in this release.", call. = FALSE)
  }

  supplied_sources <- sum(!is.null(data), !is.null(resource), !is.null(symbol))
  if (supplied_sources == 0L) {
    symbol <- "D"
  } else if (supplied_sources > 1L) {
    stop("Provide only one of 'data', 'resource', or 'symbol'.", call. = FALSE)
  }

  model_spec <- if (inherits(model, "dsflower_model")) {
    ds.flower.model(model)
  } else {
    do.call(ds.flower.model, c(list(name = model), model_params))
  }
  if (inherits(model, "dsflower_model") && length(model_params)) {
    model_spec$params <- utils::modifyList(model_spec$params %||% list(), model_params)
  }

  strategy_spec <- if (inherits(strategy, "dsflower_strategy")) {
    if (length(strategy_params)) {
      stop("'strategy_params' cannot be used when 'strategy' is already a ",
           "dsflower_strategy object.", call. = FALSE)
    }
    strategy
  } else {
    do.call(ds.flower.strategy, c(list(name = strategy), strategy_params))
  }

  if (!is.null(task)) {
    task_spec <- ds.flower.task(task)
    .assert_supported_task(task_spec)
    model_loss <- model_spec$loss %||% .dsflower_get_model(model_spec$name)$loss
    expected_task <- if (model_loss %in%
                         c("mse", "poisson_nll", "negbin_nll", "gamma_nll")) {
      "regression"
    } else {
      "classification"
    }
    if (!identical(task_spec$type, expected_task)) {
      stop("Task '", task_spec$type, "' is incompatible with model '",
           model_spec$name, "' (expected ", expected_task, ").", call. = FALSE)
    }
  }

  # Vision models train a head on frozen-backbone image features; all else tabular.
  data_kind <- if (model_spec$name %in% c("pytorch_resnet18", "pytorch_densenet121"))
    "image" else "tabular"

  # The submission pipeline owns connect/upload/pin/run/cleanup. The aggregation
  # strategy runs server-side (researcher SuperLink) over already-DP updates, so
  # any of the supported strategies is DP-safe post-processing (label_set/masks
  # remain back-compat no-ops for the enforced-DP tracks).
  ds.flower.submit(
    conns, model = model_spec, target = target, features = features,
    data = data, resource = resource, symbol = symbol,
    num_rounds = rounds, model_params = list(), strategy = strategy_spec,
    data_kind = data_kind, torch_backend = torch_backend,
    feature_bounds = feature_bounds,
    target_levels = target_levels, target_bounds = target_bounds,
    allow_insecure_http = allow_insecure_http,
    output_dir = output_dir, output_name = output_name,
    verbose = verbose, silent = silent)
}

#' @rdname ds.flower.fit
#' @export
ds.flower.train <- ds.flower.fit
