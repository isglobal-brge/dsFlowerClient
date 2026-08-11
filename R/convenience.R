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

.dsflower_canonical_model_name <- function(name) {
  aliases <- c(
    logreg = "pytorch_logreg", logistic = "pytorch_logreg",
    logistic_regression = "pytorch_logreg",
    mlp = "pytorch_mlp", torch_mlp = "pytorch_mlp",
    torch_logreg = "pytorch_logreg", multiclass = "pytorch_multiclass",
    linear_regression = "pytorch_linear_regression",
    huber = "pytorch_huber", robust_regression = "pytorch_huber",
    quantile = "pytorch_quantile", quantile_regression = "pytorch_quantile",
    poisson = "pytorch_poisson", multilabel = "pytorch_multilabel",
    extra_trees_classifier = "extra_trees", extratrees = "extra_trees",
    randomforest = "random_forest", rf = "random_forest",
    lightgbm_style = "lightgbm", lgbm = "lightgbm",
    catboost_style = "catboost",
    resnet18 = "pytorch_resnet18", densenet121 = "pytorch_densenet121")
  key <- .dsflower_choice_key(name)
  if (key %in% names(aliases)) aliases[[key]] else key
}

#' Create a model spec by name
#'
#' Resolves a registered model name and validates its typed parameter contract.
#' Existing \code{dsflower_model} objects are canonicalised and revalidated
#' against the current registry.
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
    canonical <- .dsflower_canonical_model_name(name$name)
    registered <- .dsflower_get_model(canonical)
    params <- .dsflower_resolve_model_params(
      registered, name$params %||% list())
    framework <- .dsflower_model_framework(registered)
    return(structure(list(
      name = registered$name, track = registered$track,
      framework = framework,
      loss = .dsflower_model_loss(registered, params), params = params),
      class = "dsflower_model"))
  }
  if (!is.character(name) || length(name) != 1L) {
    stop("'name' must be a dsflower_model object or a single model name.",
         call. = FALSE)
  }

  # Friendly aliases -> canonical registered names (declarative PyTorch; the
  # full model set lives in the client-side registry, extensible by derived
  # packages -- there is NO server-side catalog).
  canonical <- .dsflower_canonical_model_name(name)

  m <- .dsflower_get_model(canonical)   # registry lookup; errors listing available
  params <- .dsflower_resolve_model_params(m, list(...))
  framework <- .dsflower_model_framework(m)
  structure(list(name = m$name, track = m$track, framework = framework,
                 loss = .dsflower_model_loss(m, params), params = params),
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
         "Supported tasks: classification, regression, count.", call. = FALSE)
  }

  choices <- c(
    classification = "ds.flower.task.classification",
    class = "ds.flower.task.classification",
    regression = "ds.flower.task.regression",
    count = "ds.flower.task.count"
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
#'   model-specific auto handling.
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
#' @param feature_bounds Optional public feature bounds as
#'   \code{list(lower=..., upper=...)} in feature order.
#' @param feature_cuts Required for native-tight tree models: a list containing
#'   one strictly increasing vector of public cut points per feature, each
#'   strictly inside \code{feature_bounds}. Seven data-independent
#'   cuts per feature are a practical benchmark-backed starting point; they are never
#'   inferred from private node data.
#' @param target_levels Optional ordered public classification label vocabulary.
#'   Non-numeric labels require it; missing or unknown values map to public code
#'   zero. Multilabel applies the same public two-level vocabulary independently
#'   to every target column. Binary native-tree models require exactly two
#'   ordered levels.
#' @param target_bounds Required public \code{list(lower=..., upper=...)} for
#'   regression/count models.
#' @param allow_insecure_http Character vector of exact connection names allowed
#'   to use plaintext HTTP. Empty by default. This exception does not provide
#'   transport security; use it only behind an independently trusted network.
#' @param data_kind Optional input kind, \code{"tabular"} or \code{"image"}.
#'   It is inferred when the registered model supports exactly one kind; models
#'   registered for both require an explicit choice.
#' @param holdout Optional numeric test fraction strictly between zero and one,
#'   with at most six decimal places. The node assigns complete privacy units
#'   before training, trains only on the complement, and returns the final model
#'   together with one pooled differentially-private test metric release. This
#'   supports tabular neural/native-tree models and native dsFlower vision.
#' @param cross_validation Optional integer in \code{[2, 10]}. This runs a
#'   dedicated metrics-only tabular CV job for neural or binary/regression
#'   native-tree models. It releases one pooled DP OOF result and saves no fold
#'   model or prediction. When \code{rounds} is omitted, native-tree CV uses its
#'   required single round per fold; an explicit value is never overwritten.
#'   Prefer \code{ds.flower.cross_validate()} for this workflow.
#' @return A \code{dsflower_run} object, or a \code{dsflower_cv} when
#'   \code{cross_validation} is set.
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
                          output_dir = NULL,
                          output_name = NULL,
                          silent = FALSE,
                          verbose = FALSE,
                          feature_bounds = NULL,
                          feature_cuts = NULL,
                          target_levels = NULL,
                          target_bounds = NULL,
                          allow_insecure_http = getOption(
                            "dsflower.dsi_allow_insecure_http", character()),
                          data_kind = NULL,
                          holdout = NULL,
                          cross_validation = NULL) {
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
  if (missing(rounds) && !is.null(cross_validation)) {
    registered <- if (inherits(model, "dsflower_model")) {
      model
    } else {
      .dsflower_get_model(model)
    }
    if (identical(registered$track %||% NULL, "native_tree")) rounds <- 1L
  }
  if (!is.numeric(rounds) || is.logical(rounds)) {
    stop("'rounds' must be a single positive integer no greater than 500.",
         call. = FALSE)
  }
  rounds_value <- as.numeric(rounds)
  if (length(rounds_value) != 1L || is.na(rounds_value) ||
      !is.finite(rounds_value) || rounds_value < 1 ||
      rounds_value %% 1 != 0 || rounds_value > 500) {
    stop("'rounds' must be a single positive integer no greater than 500.",
         call. = FALSE)
  }
  rounds <- as.integer(rounds_value)
  if (!is.list(model_params) ||
      (length(model_params) > 0L && is.null(names(model_params)))) {
    stop("'model_params' must be a named list.", call. = FALSE)
  }
  if (!is.list(strategy_params) ||
      (length(strategy_params) > 0L && is.null(names(strategy_params)))) {
    stop("'strategy_params' must be a named list.", call. = FALSE)
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
    registered <- .dsflower_get_model(model_spec$name)
    registered$defaults <- model_spec$params %||% list()
    model_spec$params <- .dsflower_resolve_model_params(
      registered, model_params)
    model_spec$loss <- .dsflower_model_loss(registered, model_spec$params)
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
    expected_task <- if (identical(model_spec$track, "native_tree")) {
      if (identical(model_spec$params$task, "regression"))
        "regression" else "classification"
    } else if (model_loss %in% c("poisson_nll", "negbin_nll")) {
      "count"
    } else if (model_loss %in% c(
                 "mse", "huber", "quantile", "gamma_nll")) {
      "regression"
    } else {
      "classification"
    }
    if (!identical(task_spec$type, expected_task)) {
      stop("Task '", task_spec$type, "' is incompatible with model '",
           model_spec$name, "' (expected ", expected_task, ").", call. = FALSE)
    }
  }

  supported_kinds <- .dsflower_get_model(model_spec$name)$data_kinds %||% "tabular"
  if (is.null(data_kind)) {
    if (length(supported_kinds) != 1L) {
      stop("Model '", model_spec$name, "' supports more than one input kind; ",
           "set 'data_kind' explicitly to one of: ",
           paste(supported_kinds, collapse = ", "), ".", call. = FALSE)
    }
    data_kind <- supported_kinds[[1L]]
  } else if (!is.character(data_kind) || length(data_kind) != 1L ||
             is.na(data_kind) || !data_kind %in% supported_kinds) {
    stop("'data_kind' must be exactly one of the kinds supported by model '",
         model_spec$name, "': ", paste(supported_kinds, collapse = ", "), ".",
         call. = FALSE)
  }

  # The submission pipeline owns connect/upload/pin/run/cleanup. The aggregation
  # strategy runs server-side (researcher SuperLink) over already-DP updates, so
  # any supported strategy is DP-safe post-processing.
  ds.flower.submit(
    conns, model = model_spec, target = target, features = features,
    data = data, resource = resource, symbol = symbol,
    num_rounds = rounds, model_params = list(), strategy = strategy_spec,
    data_kind = data_kind, torch_backend = torch_backend,
    feature_bounds = feature_bounds,
    feature_cuts = feature_cuts,
    target_levels = target_levels, target_bounds = target_bounds,
    allow_insecure_http = allow_insecure_http,
    holdout = holdout,
    cross_validation = cross_validation,
    output_dir = output_dir, output_name = output_name,
    verbose = verbose, silent = silent)
}
