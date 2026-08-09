# Module: Client-side Model Registry
#
# dsFlower has NO server-side model catalog. The node runs whatever DP-valid
# submission it is sent; the "catalog" is purely a CLIENT-SIDE convenience that
# turns a friendly name + params into the artifact shipped in the FAB:
#   * neural track -> a model SPEC the node builds with stock torch layers
#   * native-tree track -> a built-in typed request for a trusted node adapter
# Specs are DATA, never code: nothing the researcher submits runs in the node's
# trusted interpreter (which is what makes the DP-release path unforgeable).
#
# The registry is extensible the way tidymodels/parsnip is: a derived
# dsFlowerClient extension package registers its own model collection by calling
# `ds.flower.register_model()` from its `.onLoad()`. Registration is client-side
# state only — it never adds anything server-side.

# Internal registry environment (parsnip's get_model_env() analogue).
.dsflower_models <- new.env(parent = emptyenv())

.dsflower_reserved_privacy_parameters <- c(
  "epsilon", "delta", "privacy", "noise",
  "noise_multiplier", "seed", "random_seed", "secret", "noise_root",
  "max_grad_norm", "clip_norm", "clipping_norm")

.DSFLOWER_XGBOOST_TASK_DEFAULTS <- list(
  binary = list(
    num_boost_round = 8L, max_depth = 2L, learning_rate = 0.25),
  regression = list(
    num_boost_round = 5L, max_depth = 2L, learning_rate = 0.30))

#' Register a dsFlower model generator
#'
#' Intended for dsFlowerClient extension packages: call this from your package's
#' \code{.onLoad()} to add models to the registry. The model becomes usable via
#' \code{ds.flower.model("<name>")} / \code{ds.flower.fit(..., model = "<name>")}.
#'
#' @param name Character; the model name (e.g. "pytorch_logreg").
#' @param track Character; the enforced-DP track. The public model registry
#'   currently accepts only "neural" (nn.Module + DP-SGD). Native tree engines
#'   are registered by their trusted server adapters, not extension generators.
#' @param data_kinds Character vector of supported inputs:
#'   \code{"tabular"}, \code{"image"}, or both.
#' @param generate Function of one argument \code{params} (a named list). It
#'   MUST return a named model SPEC (DATA, never code):
#'   model SPEC \code{list(kind = "sequential", layers = list(...))} the node
#'   builds with stock torch layers (end with a linear onto \code{"@out"}). The
#'   node owns the loop, loss, and optimizer.
#' @param loss Character; required for the neural track. The per-sample neural
#'   loss must come from the node
#'   allowlist: stock losses \code{bce_logits},
#'   \code{cross_entropy}, \code{mse}, \code{poisson_nll}, \code{multilabel_bce},
#'   \code{hinge} (linear SVM), \code{ordinal} (CORN); plus vetted custom per-sample
#'   losses \code{negbin_nll} (overdispersed counts), \code{gamma_nll}
#'   (positive continuous), \code{huber} (bounded robust regression), and
#'   \code{quantile} (bounded conditional-quantile regression).
#'   The node pins the actual loss; this is the client's request.
#' @param defaults Named list of default params merged under user-supplied params.
#' @param description Character or NULL; a one-line human description.
#' @param vetted Logical; informational only. Every model is node-built from the
#'   allowlisted spec vocabulary (no researcher code runs), so first- and
#'   third-party generators take the same validated path; this flag just marks
#'   first-party collections in \code{ds.flower.list_models()}.
#' @param overwrite Logical; allow replacing an existing registration.
#' @param parameter_types Required named character vector defining every accepted
#'   parameter and its basic type. Use \code{character()} for a model with no
#'   parameters. This makes misspelled or unsupported parameters fail early.
#' @param parameter_aliases Optional named character vector mapping accepted aliases
#'   to canonical parameter names (for example
#'   \code{c(eta = "learning_rate")}).
#' @param required_parameters Character vector of canonical parameters which must be
#'   present after defaults and user values are merged.
#' @param parameter_choices Optional named list of finite allowlists for declared
#'   parameters. Values outside the corresponding allowlist fail before any DSI
#'   operation.
#' @return Invisibly, the model name.
#' @export
ds.flower.register_model <- function(name, track, generate, loss = NULL,
                                     defaults = list(), description = NULL,
                                     vetted = FALSE, overwrite = FALSE,
                                     parameter_types = NULL,
                                     parameter_aliases = character(),
                                     required_parameters = character(),
                                     parameter_choices = list(),
                                     data_kinds = "tabular") {
  if (!is.character(name) || length(name) != 1L || !nzchar(name)) {
    stop("'name' must be a single non-empty string.", call. = FALSE)
  }
  if (!grepl("^[a-z][a-z0-9_]*$", name) ||
      !identical(.dsflower_canonical_model_name(name), name)) {
    stop("'name' must be a canonical lowercase identifier and must not collide ",
         "with a built-in model alias.", call. = FALSE)
  }
  track <- match.arg(track, "neural")
  if (!is.character(data_kinds) || !length(data_kinds) || anyNA(data_kinds) ||
      anyDuplicated(data_kinds) ||
      length(setdiff(data_kinds, c("tabular", "image")))) {
    stop("'data_kinds' must contain tabular, image, or both without duplicates.",
         call. = FALSE)
  }
  if (!is.function(generate)) {
    stop("'generate' must be a function(params) -> source/spec.", call. = FALSE)
  }
  if (!is.null(loss)) {
    allowed <- c("bce_logits", "cross_entropy", "mse", "poisson_nll",
                 "multilabel_bce", "hinge", "negbin_nll", "gamma_nll",
                 "huber", "quantile", "ordinal")
    if (!is.character(loss) || length(loss) != 1L || !loss %in% allowed) {
      stop("'loss' must be one of the node allowlist: ",
           paste(allowed, collapse = ", "), ".", call. = FALSE)
    }
  }
  if (identical(track, "neural") && is.null(loss)) {
    stop("Neural models must declare one trusted 'loss'.", call. = FALSE)
  }
  if (!is.list(defaults) ||
      (length(defaults) &&
       (is.null(names(defaults)) || anyNA(names(defaults)) ||
        any(!nzchar(names(defaults))) || anyDuplicated(names(defaults))))) {
    stop("'defaults' must be a uniquely named list.", call. = FALSE)
  }
  if (length(defaults) && any(vapply(defaults, is.null, logical(1)))) {
    stop("Model defaults cannot be NULL; omit optional defaults.", call. = FALSE)
  }
  if (is.null(parameter_types)) {
    stop("'parameter_types' is required; use character() for a model with no ",
         "public parameters.", call. = FALSE)
  }
  if (!is.null(parameter_types)) {
    type_names <- names(parameter_types)
    parameter_types <- as.character(parameter_types)
    names(parameter_types) <- if (length(parameter_types)) type_names else character()
    if (length(parameter_types) &&
        (is.null(names(parameter_types)) || any(!nzchar(names(parameter_types))) ||
         anyDuplicated(names(parameter_types)))) {
      stop("'parameter_types' must be a uniquely named character vector.",
           call. = FALSE)
    }
    supported_types <- c(
      "number", "positive_number", "nonnegative_number",
      "integer", "positive_integer", "positive_integer_vector",
      "hidden_layers", "logical", "character", "list",
      "numeric_interval", "numeric_intervals"
    )
    bad_types <- setdiff(parameter_types, supported_types)
    if (length(bad_types)) {
      stop("Unsupported parameter type(s): ", paste(bad_types, collapse = ", "),
           ".", call. = FALSE)
    }
    unknown_defaults <- setdiff(names(defaults), names(parameter_types))
    if (length(unknown_defaults)) {
      stop("Model defaults are missing from 'parameter_types': ",
           paste(unknown_defaults, collapse = ", "), ".", call. = FALSE)
    }
    privacy_names <- names(parameter_types)
    forbidden_privacy <- privacy_names[
      privacy_names %in% .dsflower_reserved_privacy_parameters |
        startsWith(privacy_names, "dp_") |
        startsWith(privacy_names, "privacy_")]
    if (length(forbidden_privacy)) {
      stop("Model schemas cannot expose node-owned privacy parameter(s): ",
           paste(forbidden_privacy, collapse = ", "), ".", call. = FALSE)
    }
    optimizer_children <- c(
      "momentum", "nesterov", "beta1", "beta2", "optimizer_eps",
      "amsgrad", "rmsprop_alpha")
    if (length(intersect(names(parameter_types), optimizer_children)) &&
        !"optimizer" %in% names(parameter_types)) {
      stop("Optimizer-specific parameters require a declared 'optimizer' field.",
           call. = FALSE)
    }
    scheduler_children <- c(
      "scheduler_step_size", "scheduler_gamma", "scheduler_min_lr")
    if (length(intersect(names(parameter_types), scheduler_children)) &&
        !"scheduler" %in% names(parameter_types)) {
      stop("Scheduler-specific parameters require a declared 'scheduler' field.",
           call. = FALSE)
    }
  }
  alias_names <- names(parameter_aliases)
  parameter_aliases <- as.character(parameter_aliases)
  names(parameter_aliases) <- alias_names
  if (length(parameter_aliases)) {
    if (is.null(names(parameter_aliases)) || any(!nzchar(names(parameter_aliases))) ||
        anyDuplicated(names(parameter_aliases)) || any(!nzchar(parameter_aliases))) {
      stop("'parameter_aliases' must map unique, non-empty aliases to names.",
           call. = FALSE)
    }
    if (is.null(parameter_types) ||
        length(setdiff(unname(parameter_aliases), names(parameter_types)))) {
      stop("Every parameter alias must target a declared canonical parameter.",
           call. = FALSE)
    }
    if (length(intersect(names(parameter_aliases), names(parameter_types)))) {
      stop("Parameter aliases cannot collide with canonical parameter names.",
           call. = FALSE)
    }
  }
  if (!is.character(required_parameters) || anyNA(required_parameters) ||
      any(!nzchar(required_parameters)) || anyDuplicated(required_parameters)) {
    stop("'required_parameters' must contain unique, non-empty names.",
         call. = FALSE)
  }
  if (length(required_parameters) &&
      (is.null(parameter_types) ||
       length(setdiff(required_parameters, names(parameter_types))))) {
    stop("Every required parameter must have a declared type.", call. = FALSE)
  }
  if (!is.list(parameter_choices) ||
      (length(parameter_choices) &&
       (is.null(names(parameter_choices)) || any(!nzchar(names(parameter_choices))) ||
        anyDuplicated(names(parameter_choices))))) {
    stop("'parameter_choices' must be a uniquely named list.", call. = FALSE)
  }
  unknown_choice_params <- setdiff(names(parameter_choices), names(parameter_types))
  if (length(unknown_choice_params)) {
    stop("Every parameter choice must target a declared parameter: ",
         paste(unknown_choice_params, collapse = ", "), ".", call. = FALSE)
  }
  for (key in names(parameter_choices)) {
    choices <- parameter_choices[[key]]
    if (!is.atomic(choices) || !length(choices) || anyNA(choices) ||
        anyDuplicated(choices) ||
        (is.numeric(choices) && any(!is.finite(choices)))) {
      stop("parameter_choices[['", key,
           "']] must be a non-empty finite atomic allowlist.", call. = FALSE)
    }
  }
  if (!isTRUE(overwrite) && !is.null(.dsflower_models[[name]])) {
    stop("model '", name, "' already registered; pass overwrite = TRUE to replace.",
         call. = FALSE)
  }
  .dsflower_models[[name]] <- list(
    name = name, track = track, generate = generate, loss = loss,
    defaults = defaults, parameter_types = parameter_types,
    parameter_aliases = parameter_aliases,
    required_parameters = required_parameters, description = description,
    parameter_choices = parameter_choices,
    data_kinds = data_kinds,
    available = TRUE,
    vetted = isTRUE(vetted)
  )
  invisible(name)
}

.dsflower_param_type_ok <- function(value, type) {
  scalar_number <- function(x) {
    is.numeric(x) && length(x) == 1L && !is.na(x) && is.finite(x)
  }
  scalar_integer <- function(x) {
    scalar_number(x) && x == floor(x)
  }
  switch(type,
    number = scalar_number(value),
    positive_number = scalar_number(value) && value > 0,
    nonnegative_number = scalar_number(value) && value >= 0,
    integer = scalar_integer(value),
    positive_integer = scalar_integer(value) && value > 0,
    positive_integer_vector = is.numeric(value) && length(value) > 0L &&
      !anyNA(value) && all(is.finite(value)) &&
      all(value == floor(value)) && all(value > 0),
    hidden_layers = is.numeric(value) && !is.logical(value) &&
      !anyNA(value) && all(is.finite(value)) &&
      all(value == floor(value)) && all(value > 0),
    logical = is.logical(value) && length(value) == 1L && !is.na(value),
    character = is.character(value) && length(value) == 1L &&
      !is.na(value) && nzchar(value),
    list = is.list(value),
    numeric_interval = is.numeric(value) && length(value) == 2L &&
      !anyNA(value) && all(is.finite(value)) && value[[1L]] < value[[2L]] &&
      all(abs(value) <= 1e6),
    numeric_intervals = is.list(value) && length(value) > 0L &&
      length(value) <= 65536L && all(vapply(value, function(bounds) {
        is.numeric(bounds) && length(bounds) == 2L && !anyNA(bounds) &&
          all(is.finite(bounds)) && bounds[[1L]] < bounds[[2L]] &&
          all(abs(bounds) <= 1e6)
      }, logical(1))),
    FALSE
  )
}

.dsflower_validate_parameter_limits <- function(params) {
  bounded <- function(name, lower, upper, lower_open = FALSE) {
    if (is.null(params[[name]])) return(invisible())
    value <- as.numeric(params[[name]])
    lower_ok <- if (lower_open) value > lower else value >= lower
    if (any(!is.finite(value)) || any(!lower_ok) || any(value > upper)) {
      bracket <- if (lower_open) "(" else "["
      stop("Model parameter '", name, "' must be in ", bracket, lower,
           ", ", upper, "].", call. = FALSE)
    }
  }
  bounded("learning_rate", 0, 10, lower_open = TRUE)
  bounded("batch_size", 1, 65536)
  bounded("local_epochs", 1, 1000)
  bounded("n_classes", 2, 1024)
  bounded("num_labels", 2, 1024)
  bounded("weight_decay", 0, 1000)
  bounded("l1_penalty", 0, 1000)
  bounded("nb_dispersion", 1e-6, 1e12)
  bounded("gamma_shape", 1e-6, 1e12)
  bounded("huber_delta", 1e-6, 1e6)
  bounded("quantile", 0, 1, lower_open = TRUE)
  if (!is.null(params[["quantile"]]) && params[["quantile"]] >= 1) {
    stop("Model parameter 'quantile' must be in (0, 1).", call. = FALSE)
  }
  bounded("image_size", 1, 512)
  bounded("momentum", 0, 1)
  if (!is.null(params[["momentum"]]) && params[["momentum"]] >= 1) {
    stop("Model parameter 'momentum' must be in [0, 1).", call. = FALSE)
  }
  for (name in intersect(names(params), c("beta1", "beta2", "rmsprop_alpha"))) {
    bounded(name, 0, 1)
    if (params[[name]] >= 1) {
      stop("Model parameter '", name, "' must be in [0, 1).", call. = FALSE)
    }
  }
  bounded("optimizer_eps", 0, 1, lower_open = TRUE)
  bounded("scheduler_step_size", 1, 1000)
  bounded("scheduler_gamma", 0, 10, lower_open = TRUE)
  bounded("scheduler_min_lr", 0, 10)
  bounded("gradient_clip", 0, 2e6, lower_open = TRUE)
  for (name in intersect(names(params), c(
      "channels", "n_tokens", "n_features", "d_model", "d_ff", "hidden",
      "levels"))) {
    upper <- if (identical(name, "levels")) 13 else
      if (name %in% c("channels", "n_tokens", "n_features", "d_model")) 4096 else
        8192
    bounded(name, 1, upper)
  }
  hidden <- params[["hidden_layers"]]
  if (!is.null(hidden)) {
    if (length(hidden) > 31L || any(hidden > 8192)) {
      stop("Model parameter 'hidden_layers' permits at most 31 widths, each <= 8192.",
           call. = FALSE)
    }
  }
  shape <- params[["input_shape"]]
  if (!is.null(shape) &&
      (length(shape) > 8L || any(as.numeric(shape) > 4096) ||
       prod(as.numeric(shape)) > 2^20)) {
    stop("Model parameter 'input_shape' exceeds the per-sample geometry cap.",
         call. = FALSE)
  }
  invisible(params)
}

.dsflower_complete_training_params <- function(params, declared) {
  if ("optimizer" %in% declared) {
    optimizer <- params[["optimizer"]] %||% "sgd"
    if (!is.character(optimizer) || length(optimizer) != 1L ||
        is.na(optimizer) ||
        !optimizer %in% c("sgd", "adam", "adamw", "rmsprop")) {
      stop("Optimizer must be one of: sgd, adam, adamw, rmsprop.",
           call. = FALSE)
    }
    optimizer_fields <- c(
      "momentum", "nesterov", "beta1", "beta2", "optimizer_eps",
      "amsgrad", "rmsprop_alpha")
    allowed <- switch(optimizer,
      sgd = c("momentum", "nesterov"),
      adam = c("beta1", "beta2", "optimizer_eps", "amsgrad"),
      adamw = c("beta1", "beta2", "optimizer_eps", "amsgrad"),
      rmsprop = c("momentum", "optimizer_eps", "rmsprop_alpha"))
    incompatible <- intersect(setdiff(optimizer_fields, allowed), names(params))
    if (length(incompatible)) {
      stop("Optimizer '", optimizer, "' does not use parameter(s): ",
           paste(incompatible, collapse = ", "), ".", call. = FALSE)
    }
    defaults <- switch(optimizer,
      sgd = list(momentum = 0, nesterov = FALSE),
      adam = list(beta1 = 0.9, beta2 = 0.999, optimizer_eps = 1e-8,
                  amsgrad = FALSE),
      adamw = list(beta1 = 0.9, beta2 = 0.999, optimizer_eps = 1e-8,
                   amsgrad = FALSE),
      rmsprop = list(momentum = 0, optimizer_eps = 1e-8,
                     rmsprop_alpha = 0.99))
    defaults <- defaults[intersect(names(defaults), declared)]
    params <- utils::modifyList(defaults, params)
    if (identical(optimizer, "sgd") && isTRUE(params[["nesterov"]]) &&
        as.numeric(params[["momentum"]]) <= 0) {
      stop("Nesterov acceleration requires positive SGD momentum.",
           call. = FALSE)
    }
  }

  if ("scheduler" %in% declared) {
    scheduler <- params[["scheduler"]] %||% "none"
    if (!is.character(scheduler) || length(scheduler) != 1L ||
        is.na(scheduler) ||
        !scheduler %in% c("none", "step", "exponential", "cosine")) {
      stop("Scheduler must be one of: none, step, exponential, cosine.",
           call. = FALSE)
    }
    scheduler_fields <- c(
      "scheduler_step_size", "scheduler_gamma", "scheduler_min_lr")
    scheduler_allowed <- switch(scheduler,
      none = character(), step = c("scheduler_step_size", "scheduler_gamma"),
      exponential = "scheduler_gamma", cosine = "scheduler_min_lr")
    incompatible <- intersect(
      setdiff(scheduler_fields, scheduler_allowed), names(params))
    if (length(incompatible)) {
      stop("Scheduler '", scheduler, "' does not use parameter(s): ",
           paste(incompatible, collapse = ", "), ".", call. = FALSE)
    }
    scheduler_defaults <- switch(scheduler,
      step = list(scheduler_step_size = 1L, scheduler_gamma = 0.1),
      exponential = list(scheduler_gamma = 0.1),
      cosine = list(scheduler_min_lr = 0),
      list())
    scheduler_defaults <- scheduler_defaults[
      intersect(names(scheduler_defaults), declared)]
    params <- utils::modifyList(scheduler_defaults, params)
    if (identical(scheduler, "cosine") &&
        !is.null(params[["scheduler_min_lr"]]) &&
        as.numeric(params[["scheduler_min_lr"]]) >
          as.numeric(params[["learning_rate"]] %||% 0.01)) {
      stop("scheduler_min_lr cannot exceed learning_rate.", call. = FALSE)
    }
  }
  params
}

# Resolve aliases, reject unknown names and validate public modelling parameters.
# Privacy fields deliberately never appear in a model schema.
.dsflower_resolve_model_params <- function(model, params = list()) {
  if (!is.list(params)) {
    stop("Model parameters must be supplied as a named list.", call. = FALSE)
  }
  if (length(params)) {
    keys <- names(params)
    if (is.null(keys) || any(is.na(keys) | !nzchar(keys)) || anyDuplicated(keys)) {
      stop("Model parameters must have unique, non-empty names.", call. = FALSE)
    }
    null_values <- keys[vapply(params, is.null, logical(1))]
    if (length(null_values)) {
      stop("Model parameter(s) cannot be NULL: ",
           paste(null_values, collapse = ", "),
           ". Omit a parameter to use its default.", call. = FALSE)
    }
  }

  aliases <- model$parameter_aliases %||% character()
  for (alias in intersect(names(aliases), names(params))) {
    canonical <- unname(aliases[[alias]])
    if (canonical %in% names(params)) {
      stop("Model parameters cannot set both alias '", alias,
           "' and canonical parameter '", canonical, "'.", call. = FALSE)
    }
    names(params)[names(params) == alias] <- canonical
  }

  types <- model$parameter_types
  if (!is.null(types)) {
    unknown <- setdiff(names(params), names(types))
    if (length(unknown)) {
      stop("Unknown parameter(s) for model '", model$name, "': ",
           paste(unknown, collapse = ", "), ". Allowed parameters: ",
           paste(names(types), collapse = ", "), ".", call. = FALSE)
    }
  }
  defaults <- model$defaults %||% list()
  if (identical(model$name, "xgboost")) {
    task <- params[["task"]] %||% defaults[["task"]]
    if (is.character(task) && length(task) == 1L && !is.na(task)) {
      task_defaults <- model$task_defaults[[task]]
      if (!is.null(task_defaults)) {
        defaults <- utils::modifyList(defaults, task_defaults)
      }
    }
  }
  if ("optimizer" %in% names(params)) {
    conditional <- c(
      "momentum", "nesterov", "beta1", "beta2", "optimizer_eps",
      "amsgrad", "rmsprop_alpha")
    inherited <- setdiff(conditional, names(params))
    defaults[inherited] <- NULL
  }
  if ("scheduler" %in% names(params)) {
    conditional <- c(
      "scheduler_step_size", "scheduler_gamma", "scheduler_min_lr")
    inherited <- setdiff(conditional, names(params))
    defaults[inherited] <- NULL
  }
  resolved <- utils::modifyList(defaults, params)
  resolved <- .dsflower_complete_training_params(
    resolved, names(model$parameter_types %||% character()))
  missing_required <- setdiff(model$required_parameters %||% character(),
                              names(resolved))
  if (length(missing_required)) {
    stop("Model '", model$name, "' requires parameter(s): ",
         paste(missing_required, collapse = ", "), ".", call. = FALSE)
  }
  if (!is.null(types)) {
    bad <- names(resolved)[!vapply(names(resolved), function(key) {
      .dsflower_param_type_ok(resolved[[key]], types[[key]])
    }, logical(1))]
    if (length(bad)) {
      details <- paste0(bad, " (", unname(types[bad]), ")")
      stop("Invalid parameter type/value for model '", model$name, "': ",
           paste(details, collapse = ", "), ".", call. = FALSE)
    }
  }
  choices <- model$parameter_choices %||% list()
  present_choices <- intersect(names(choices), names(resolved))
  bad_choices <- present_choices[!vapply(present_choices, function(key) {
    length(resolved[[key]]) == 1L && resolved[[key]] %in% choices[[key]]
  }, logical(1))]
  if (length(bad_choices)) {
    key <- bad_choices[[1L]]
    stop("Invalid value for model parameter '", key, "': expected one of ",
         paste(choices[[key]], collapse = ", "), ".", call. = FALSE)
  }
  .dsflower_validate_parameter_limits(resolved)
  shape_dims <- switch(model$name,
    pytorch_cnn = 3L,
    pytorch_resnet = 3L,
    pytorch_tcn = 2L,
    NULL)
  if (!is.null(shape_dims) && length(resolved$input_shape) != shape_dims) {
    expected <- if (shape_dims == 3L) "channels, height, width" else
      "channels, sequence_length"
    stop("Model '", model$name, "' requires input_shape = c(", expected,
         ").", call. = FALSE)
  }
  if (identical(model$name, "pytorch_cnn")) {
    channels <- resolved[["channels"]]
    if (length(channels) > 20L) {
      stop("Model 'pytorch_cnn' accepts at most 20 convolution channel widths.",
           call. = FALSE)
    }
    spatial <- as.numeric(resolved$input_shape[2:3])
    if (any(spatial < 2^length(channels))) {
      stop("pytorch_cnn input height and width must survive one 2x pool per ",
           "channels entry.", call. = FALSE)
    }
  }
  if (identical(model$name, "xgboost")) {
    .validate_xgboost_model_params(resolved)
  }
  resolved
}

#' List registered dsFlower models
#'
#' @return A data.frame with one row per registered model. Column
#'   \code{available} is false for a typed request surface whose trusted backend
#'   has not passed its release gate.
#' @export
ds.flower.list_models <- function() {
  names_ <- ls(.dsflower_models, sorted = TRUE)
  if (!length(names_)) {
    return(data.frame(name = character(), track = character(),
                      data_kinds = character(),
                      available = logical(),
                      loss = character(), vetted = logical(),
                      description = character(), stringsAsFactors = FALSE))
  }
  do.call(rbind, lapply(names_, function(nm) {
    m <- .dsflower_models[[nm]]
    data.frame(name = m$name, track = m$track,
               data_kinds = paste(m$data_kinds %||% "tabular", collapse = ","),
               available = isTRUE(m$available),
               loss = m$loss %||% NA_character_, vetted = m$vetted,
               description = m$description %||% NA_character_,
               stringsAsFactors = FALSE)
  }))
}

#' Inspect the public parameter contract for a registered model
#'
#' @param name Character; registered model name or friendly alias accepted by
#'   \code{ds.flower.model()}.
#' @return A data frame with one row per accepted canonical parameter and list
#'   columns \code{default} and \code{choices}. Conditional optimizer/scheduler
#'   parameters are listed even when they have no default for the selected
#'   default backend.
#' @export
ds.flower.model_parameters <- function(name) {
  if (!is.character(name) || length(name) != 1L) {
    stop("'name' must be a single model name.", call. = FALSE)
  }
  registered <- .dsflower_get_model(.dsflower_canonical_model_name(name))
  types <- registered$parameter_types %||% character()
  defaults <- .dsflower_complete_training_params(
    registered$defaults %||% list(), names(types))
  choices <- registered$parameter_choices %||% list()
  aliases <- registered$parameter_aliases %||% character()
  parameters <- names(types)
  alias_text <- vapply(parameters, function(parameter) {
    found <- names(aliases)[unname(aliases) == parameter]
    paste(found, collapse = ",")
  }, character(1))
  data.frame(
    parameter = parameters,
    type = unname(types),
    required = parameters %in% (registered$required_parameters %||% character()),
    aliases = alias_text,
    default = I(lapply(parameters, function(parameter) defaults[[parameter]])),
    choices = I(lapply(parameters, function(parameter) choices[[parameter]])),
    stringsAsFactors = FALSE)
}

#' Look up a registered model (internal)
#' @keywords internal
.dsflower_get_model <- function(name) {
  m <- .dsflower_models[[name]]
  if (is.null(m)) {
    avail <- ls(.dsflower_models, sorted = TRUE)
    stop("Unknown model '", name, "'. Registered models: ",
         paste(avail, collapse = ", "),
         ". (Extension packages add more via ds.flower.register_model().)",
         call. = FALSE)
  }
  m
}

# Resolve fields which depend on a built-in model's typed parameters.
.dsflower_model_loss <- function(model, params) {
  if (identical(model$track, "native_tree") &&
      identical(model$engine, "xgboost")) {
    return(if (identical(params$task, "regression"))
      "squared_error" else "binary_logistic")
  }
  model$loss
}

.dsflower_model_framework <- function(model) {
  if (identical(model$track, "native_tree")) return(model$engine)
  if (identical(model$data_kinds, "image")) "pytorch_vision" else "pytorch"
}

# --------------------------------------------------------------------------- #
# Built-in generators (first-party, vetted = stash-free architectures).
# --------------------------------------------------------------------------- #

# A generic feed-forward head as a declarative SPEC (DATA, never code): a list of
# allowlisted layers the NODE builds with stock torch constructors. `hidden` are
# the hidden widths (each a linear + relu); every model ends with a linear onto the
# symbolic "@out", whose width the node fixes from the pinned loss -- so output
# width is node-authoritative, never client-set. The input dim "@in" is injected
# node-side (num-features, or the frozen-backbone feature dim for vision).
.neural_mlp_spec <- function(hidden = integer(0)) {
  if (!is.numeric(hidden) || is.logical(hidden) || anyNA(hidden) ||
      any(!is.finite(hidden)) || any(hidden < 1L) ||
      any(hidden != floor(hidden))) {
    stop("hidden_layers must contain positive integer widths.", call. = FALSE)
  }
  layers <- list()
  for (w in hidden) {
    layers <- c(layers, list(list(op = "linear", out = as.integer(w)),
                             list(op = "relu")))
  }
  layers <- c(layers, list(list(op = "linear", out = "@out")))
  list(kind = "sequential", layers = layers)
}

# A small 2D-CNN as a SPEC (DATA, node-built): reshape the flat per-sample vector
# into (C,H,W), then a conv/pool stack -> adaptive pool -> flatten -> linear head.
# input_shape must multiply to the feature count (the node rejects a mismatch).
.neural_cnn_spec <- function(input_shape, channels = c(8L, 16L)) {
  if (length(input_shape) != 3L) {
    stop("pytorch_cnn input_shape must be c(channels, height, width).",
         call. = FALSE)
  }
  layers <- list(list(op = "reshape", shape = as.list(as.integer(input_shape))))
  for (ch in channels)
    layers <- c(layers, list(
      list(op = "conv2d", out_channels = as.integer(ch), kernel_size = 3L, padding = 1L),
      list(op = "relu"),
      list(op = "maxpool2d", kernel_size = 2L)))
  layers <- c(layers, list(
    list(op = "adaptiveavgpool2d", output_size = list(1L, 1L)),
    list(op = "flatten"),
    list(op = "linear", out = "@out")))
  list(kind = "sequential", layers = layers)
}

# A Temporal CNN as a SPEC (DATA, node-built): reshape into (C,L), then a dilated
# length-preserving conv1d stack (receptive field grows as 2^level) -> flatten -> head.
.neural_tcn_spec <- function(input_shape, channels = 8L, levels = 3L) {
  if (length(input_shape) != 2L) {
    stop("pytorch_tcn input_shape must be c(channels, sequence_length).",
         call. = FALSE)
  }
  layers <- list(list(op = "reshape", shape = as.list(as.integer(input_shape))))
  for (i in seq_len(levels)) {
    d <- as.integer(2L^(i - 1L))
    layers <- c(layers, list(
      list(op = "conv1d", out_channels = as.integer(channels),
           kernel_size = 3L, padding = d, dilation = d),
      list(op = "relu")))
  }
  layers <- c(layers, list(list(op = "flatten"), list(op = "linear", out = "@out")))
  list(kind = "sequential", layers = layers)
}

# A residual CNN block as a typed GRAPH (DAG) spec (DATA, node-built): reshape the flat
# vector into (C,H,W), conv -> relu -> conv, ADD the skip, global-pool -> flatten -> head.
# Demonstrates the graph language (named tensors, multi-input 'add'); the node builds a
# GraphModule from allowlisted per-sample-safe ops only. input_shape multiplies to the
# feature count (the node rejects a mismatch).
.neural_resnet_spec <- function(input_shape, channels = 8L) {
  if (length(input_shape) != 3L) {
    stop("pytorch_resnet input_shape must be c(channels, height, width).",
         call. = FALSE)
  }
  ish <- as.list(as.integer(input_shape))
  ch <- as.integer(channels)
  list(kind = "graph", output = "out", nodes = list(
    list(name = "img",  op = "reshape", `in` = list("@in"), shape = ish),
    list(name = "c1",   op = "conv2d",  `in` = list("img"), out_channels = ch, kernel_size = 3L, padding = 1L),
    list(name = "r1",   op = "relu",    `in` = list("c1")),
    list(name = "c2",   op = "conv2d",  `in` = list("r1"),  out_channels = ch, kernel_size = 3L, padding = 1L),
    list(name = "res",  op = "add",     `in` = list("c1", "c2")),
    list(name = "pool", op = "adaptiveavgpool2d", `in` = list("res"), output_size = list(1L, 1L)),
    list(name = "flat", op = "flatten", `in` = list("pool")),
    list(name = "out",  op = "linear",  `in` = list("flat"), out = "@out")))
}

# A Transformer ENCODER block as a typed GRAPH (DATA, node-built) -- self-attention
# (Q/K/V linears + matmul + softmax over TOKENS, never the batch) + residual + LayerNorm
# + FFN, built ENTIRELY from per-sample-safe primitives (no Opacus DPMultiheadAttention,
# no custom code). n_tokens * d_model must equal the feature count.
.neural_transformer_spec <- function(n_tokens, d_model, d_ff = 32L) {
  T <- as.integer(n_tokens); d <- as.integer(d_model); ff <- as.integer(d_ff)
  list(kind = "graph", output = "out", nodes = list(
    list(name = "x",   op = "reshape",   `in` = list("@in"), shape = list(T, d)),
    list(name = "q",   op = "linear",    `in` = list("x"),   out = d),
    list(name = "k",   op = "linear",    `in` = list("x"),   out = d),
    list(name = "v",   op = "linear",    `in` = list("x"),   out = d),
    list(name = "kt",  op = "transpose", `in` = list("k"),   dims = list(0L, 1L)),
    list(name = "sc",  op = "matmul",    `in` = list("q", "kt")),
    list(name = "a",   op = "softmax",   `in` = list("sc"),  axis = 1L),
    list(name = "ctx", op = "matmul",    `in` = list("a", "v")),
    list(name = "res", op = "add",       `in` = list("x", "ctx")),
    list(name = "n1",  op = "layernorm", `in` = list("res")),
    list(name = "f1",  op = "linear",    `in` = list("n1"),  out = ff),
    list(name = "fa",  op = "relu",      `in` = list("f1")),
    list(name = "f2",  op = "linear",    `in` = list("fa"),  out = d),
    list(name = "res2", op = "add",      `in` = list("n1", "f2")),
    list(name = "n2",  op = "layernorm", `in` = list("res2")),
    list(name = "flat", op = "flatten",  `in` = list("n2")),
    list(name = "out", op = "linear",    `in` = list("flat"), out = "@out")))
}

# An LSTM/GRU sequence model as a typed GRAPH (DATA, node-built): reshape the flat vector
# into (n_tokens, n_features), run a sanitized Opacus DP-RNN over time, take the last
# hidden state -> head. Recurrence is within a sample (over time), never across the batch.
# n_tokens * n_features must equal the feature count.
.neural_seq_spec <- function(n_tokens, n_features, hidden = 32L, kind = "lstm") {
  list(kind = "graph", output = "out", nodes = list(
    list(name = "x",   op = "reshape", `in` = list("@in"),
         shape = list(as.integer(n_tokens), as.integer(n_features))),
    list(name = "h",   op = kind, `in` = list("x"), hidden = as.integer(hidden)),
    list(name = "out", op = "linear", `in` = list("h"), out = "@out")))
}

# Output width is decided NODE-SIDE from the pinned loss (model_spec.output_width):
# bce_logits -> 1 (binary) or one-vs-rest; cross_entropy / hinge -> num-classes;
# ordinal -> K-1 cumulative thresholds; multilabel_bce -> num-labels;
# mse / poisson_nll / negbin_nll / gamma_nll -> 1. The spec just targets "@out".

#' Register the first-party model collection (called from .onLoad)
#' @keywords internal
.dsflower_register_builtins <- function(overwrite = TRUE) {
  reg <- function(...) ds.flower.register_model(..., vetted = TRUE, overwrite = overwrite)
  neural_common <- c(
    learning_rate = "positive_number", batch_size = "positive_integer",
    local_epochs = "positive_integer", weight_decay = "nonnegative_number",
    l1_penalty = "nonnegative_number", optimizer = "character",
    momentum = "nonnegative_number", nesterov = "logical",
    beta1 = "nonnegative_number", beta2 = "nonnegative_number",
    optimizer_eps = "positive_number", amsgrad = "logical",
    rmsprop_alpha = "nonnegative_number", scheduler = "character",
    scheduler_step_size = "positive_integer",
    scheduler_gamma = "positive_number",
    scheduler_min_lr = "nonnegative_number"
  )
  neural_defaults <- list(
    learning_rate = 0.01, batch_size = 32L, local_epochs = 1L,
    weight_decay = 0, l1_penalty = 0,
    optimizer = "sgd", scheduler = "none")
  neural_choices <- list(
    optimizer = c("sgd", "adam", "adamw", "rmsprop"),
    scheduler = c("none", "step", "exponential", "cosine"))
  neural_reg <- function(name, track, generate, loss, defaults = list(), ...) {
    reg(name, track, generate = generate, loss = loss,
        defaults = utils::modifyList(neural_defaults, defaults),
        parameter_choices = neural_choices, ...)
  }
  with_common <- function(...) c(neural_common, ...)
  class_aliases <- c(num_classes = "n_classes")

  # ---- neural: tabular (output width comes from the loss, node-side) ----
  # Classification family (bounded losses): default learning_rate = 0.1. On
  # STANDARDIZED features (which the submit pipeline now provides) the old 0.01 is
  # far too timid -- it barely moves off init in a handful of rounds (validated:
  # lr=0.01 -> ~0.52, lr=0.1 -> ~0.90 on breast_cancer with DP eps=3). Regression /
  # count losses keep the 0.01 fallback (their step size tracks the raw target scale).
  neural_reg("pytorch_logreg", "neural",
      generate = function(p) .neural_mlp_spec(integer(0)),
      loss = "bce_logits", defaults = list(learning_rate = 0.1),
      parameter_types = neural_common,
      description = "Logistic regression (single linear layer).")
  neural_reg("pytorch_mlp", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% c(64L, 32L)),
      loss = "bce_logits", defaults = list(hidden_layers = c(64L, 32L), learning_rate = 0.1),
      parameter_types = with_common(hidden_layers = "hidden_layers"),
      description = "Multilayer perceptron classifier.")
  neural_reg("pytorch_multiclass", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% integer(0)),
      loss = "cross_entropy",
      defaults = list(hidden_layers = integer(0), n_classes = 3L,
                      learning_rate = 0.1),
      parameter_types = with_common(hidden_layers = "hidden_layers",
                                    n_classes = "positive_integer"),
      parameter_aliases = class_aliases,
      description = "Multiclass classifier (softmax cross-entropy).")
  neural_reg("pytorch_linear_regression", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% integer(0)),
      loss = "mse", defaults = list(hidden_layers = integer(0)),
      parameter_types = with_common(hidden_layers = "hidden_layers"),
      description = "Linear / MLP regression (MSE).")
  neural_reg("pytorch_huber", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% integer(0)),
      loss = "huber",
      defaults = list(hidden_layers = integer(0), huber_delta = 1.0),
      parameter_types = with_common(
        hidden_layers = "hidden_layers", huber_delta = "positive_number"),
      description = "Robust bounded-target regression (per-sample Huber loss).")
  neural_reg("pytorch_quantile", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% integer(0)),
      loss = "quantile",
      defaults = list(hidden_layers = integer(0), quantile = 0.5),
      parameter_types = with_common(
        hidden_layers = "hidden_layers", quantile = "positive_number"),
      description = "Bounded conditional-quantile regression (per-sample pinball loss).")
  neural_reg("pytorch_poisson", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% integer(0)),
      loss = "poisson_nll", defaults = list(hidden_layers = integer(0)),
      parameter_types = with_common(hidden_layers = "hidden_layers"),
      description = "Poisson regression (count outcomes).")
  neural_reg("pytorch_multilabel", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% integer(0)),
      loss = "multilabel_bce",
      defaults = list(num_labels = 2L, hidden_layers = c(64L, 32L),
                      learning_rate = 0.1, batch_size = 32L, local_epochs = 1L),
      parameter_types = with_common(num_labels = "positive_integer",
                                    hidden_layers = "hidden_layers"),
      parameter_aliases = c(n_labels = "num_labels"),
      description = "Multilabel classifier (independent BCE per label).")
  neural_reg("pytorch_svm", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% integer(0)),
      loss = "hinge", defaults = list(
        hidden_layers = integer(0), n_classes = 2L, learning_rate = 0.1),
      parameter_types = with_common(hidden_layers = "hidden_layers",
                                    n_classes = "positive_integer"),
      parameter_aliases = class_aliases,
      description = "Linear SVM (multiclass hinge / MultiMarginLoss).")
  neural_reg("pytorch_negbin", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% integer(0)),
      loss = "negbin_nll",
      defaults = list(hidden_layers = integer(0), nb_dispersion = 1.0),
      parameter_types = with_common(hidden_layers = "hidden_layers",
                                    nb_dispersion = "positive_number"),
      description = "Negative-binomial regression (overdispersed counts).")
  neural_reg("pytorch_gamma", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% integer(0)),
      loss = "gamma_nll",
      defaults = list(hidden_layers = integer(0), gamma_shape = 1.0),
      parameter_types = with_common(hidden_layers = "hidden_layers",
                                    gamma_shape = "positive_number"),
      description = "Gamma regression (positive continuous: cost, length-of-stay, concentration).")
  neural_reg("pytorch_ordinal", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% integer(0)),
      loss = "ordinal", defaults = list(hidden_layers = integer(0), n_classes = 3L, learning_rate = 0.1),
      parameter_types = with_common(hidden_layers = "hidden_layers",
                                    n_classes = "positive_integer"),
      parameter_aliases = class_aliases,
      description = "Ordinal regression (CORN cumulative-threshold tasks).")
  neural_reg("pytorch_ridge", "neural",
      generate = function(p) .neural_mlp_spec(integer(0)),
      loss = "mse", defaults = list(weight_decay = 1.0),
      parameter_types = neural_common,
      description = "Ridge regression (linear + L2 penalty).")
  neural_reg("pytorch_lasso", "neural",
      generate = function(p) .neural_mlp_spec(integer(0)),
      loss = "mse", defaults = list(l1_penalty = 0.01),
      parameter_types = neural_common,
      description = "Lasso regression (linear + L1 penalty).")
  neural_reg("pytorch_elasticnet", "neural",
      generate = function(p) .neural_mlp_spec(integer(0)),
      loss = "mse", defaults = list(weight_decay = 1.0, l1_penalty = 0.01),
      parameter_types = neural_common,
      description = "Elastic-net regression (linear + L1 + L2 penalties).")

  # ---- neural: convolutional (reshape flat features -> conv stack, node-built) ----
  neural_reg("pytorch_cnn", "neural",
      generate = function(p) .neural_cnn_spec(
        p$input_shape %||% stop("pytorch_cnn needs model_params$input_shape = c(C,H,W) multiplying to the feature count"),
        p$channels %||% c(8L, 16L)),
      loss = "cross_entropy", defaults = list(
        n_classes = 2L, channels = c(8L, 16L)),
      parameter_types = with_common(
        input_shape = "positive_integer_vector",
        channels = "positive_integer_vector", n_classes = "positive_integer"),
      parameter_aliases = class_aliases, required_parameters = "input_shape",
      description = "2D CNN (reshape -> conv/pool stack -> head).")
  neural_reg("pytorch_tcn", "neural",
      generate = function(p) .neural_tcn_spec(
        p$input_shape %||% stop("pytorch_tcn needs model_params$input_shape = c(C,L) multiplying to the feature count"),
        p$channels %||% 8L, p$levels %||% 3L),
      loss = "cross_entropy",
      defaults = list(n_classes = 2L, channels = 8L, levels = 3L,
                      learning_rate = 0.001, batch_size = 32L,
                      local_epochs = 1L),
      parameter_types = with_common(
        input_shape = "positive_integer_vector", channels = "positive_integer",
        levels = "positive_integer", n_classes = "positive_integer"),
      parameter_aliases = class_aliases, required_parameters = "input_shape",
      description = "Temporal CNN (dilated conv1d stack over a sequence).")
  neural_reg("pytorch_resnet", "neural",
      generate = function(p) .neural_resnet_spec(
        p$input_shape %||% stop("pytorch_resnet needs model_params$input_shape = c(C,H,W) multiplying to the feature count"),
        p$channels %||% 8L),
      loss = "cross_entropy", defaults = list(n_classes = 2L, channels = 8L),
      parameter_types = with_common(
        input_shape = "positive_integer_vector", channels = "positive_integer",
        n_classes = "positive_integer"),
      parameter_aliases = class_aliases, required_parameters = "input_shape",
      description = "Residual CNN block (typed-graph DAG: conv->conv->skip-add->pool->head).")
  neural_reg("pytorch_transformer", "neural",
      generate = function(p) .neural_transformer_spec(
        p$n_tokens %||% stop("pytorch_transformer needs model_params$n_tokens"),
        p$d_model  %||% stop("pytorch_transformer needs model_params$d_model (n_tokens*d_model = feature count)"),
        p$d_ff %||% 32L),
      loss = "cross_entropy", defaults = list(n_classes = 2L, d_ff = 32L),
      parameter_types = with_common(
        n_tokens = "positive_integer", d_model = "positive_integer",
        d_ff = "positive_integer", n_classes = "positive_integer"),
      parameter_aliases = class_aliases,
      required_parameters = c("n_tokens", "d_model"),
      description = "Transformer encoder block (self-attention + FFN, typed-graph DAG from primitives).")
  neural_reg("pytorch_lstm", "neural",
      generate = function(p) .neural_seq_spec(
        p$n_tokens   %||% stop("pytorch_lstm needs model_params$n_tokens"),
        p$n_features %||% stop("pytorch_lstm needs model_params$n_features (n_tokens*n_features = feature count)"),
        p$hidden %||% 32L, "lstm"),
      loss = "cross_entropy",
      defaults = list(n_classes = 2L, hidden = 32L,
                      learning_rate = 0.001, batch_size = 32L,
                      local_epochs = 1L),
      parameter_types = with_common(
        n_tokens = "positive_integer", n_features = "positive_integer",
        hidden = "positive_integer", n_classes = "positive_integer"),
      parameter_aliases = class_aliases,
      required_parameters = c("n_tokens", "n_features"),
      description = "LSTM sequence model (sanitized Opacus DPLSTM, typed-graph DAG).")
  neural_reg("pytorch_gru", "neural",
      generate = function(p) .neural_seq_spec(
        p$n_tokens   %||% stop("pytorch_gru needs model_params$n_tokens"),
        p$n_features %||% stop("pytorch_gru needs model_params$n_features"),
        p$hidden %||% 32L, "gru"),
      loss = "cross_entropy", defaults = list(n_classes = 2L, hidden = 32L),
      parameter_types = with_common(
        n_tokens = "positive_integer", n_features = "positive_integer",
        hidden = "positive_integer", n_classes = "positive_integer"),
      parameter_aliases = class_aliases,
      required_parameters = c("n_tokens", "n_features"),
      description = "GRU sequence model (sanitized Opacus DPGRU, typed-graph DAG).")

  # ---- neural: vision head (frozen backbone is node-resident; the spec is the
  #      trainable head, with @in injected node-side from the backbone feature dim).
  #      local() forces a fresh nm per iteration (no lazy loop-variable capture). ----
  for (nm in c("pytorch_resnet18", "pytorch_densenet121")) local({
    nm <- nm
    neural_reg(nm, "neural",
        generate = function(p) .neural_mlp_spec(integer(0)),
        loss = "cross_entropy",
        defaults = list(n_classes = 2L,
                        volumetric = FALSE, learning_rate = 0.001,
                        batch_size = 32L, local_epochs = 1L, image_size = 224L),
        parameter_types = with_common(
          n_classes = "positive_integer", volumetric = "logical",
          image_size = "positive_integer"),
        parameter_aliases = class_aliases,
        data_kinds = "image",
        description = paste0("Vision classifier head on a frozen ", nm, " backbone."))
  })

  # Native engines are first-party node adapters, never extension generators.
  # The public object is useful for constructing and hashing the exact request,
  # but availability remains false until the native backend release gates pass.
  .dsflower_models[["xgboost"]] <- list(
    name = "xgboost", track = "native_tree", engine = "xgboost",
    generate = function(params) params, loss = NULL,
    defaults = c(list(task = "binary"),
      .DSFLOWER_XGBOOST_TASK_DEFAULTS$binary, list(
      min_child_weight = 1,
      min_split_loss = 0, reg_alpha = 0, reg_lambda = 1,
      max_delta_step = 1)),
    task_defaults = .DSFLOWER_XGBOOST_TASK_DEFAULTS,
    parameter_types = c(
      task = "character", num_boost_round = "positive_integer",
      max_depth = "positive_integer", learning_rate = "positive_number",
      min_child_weight = "nonnegative_number",
      min_split_loss = "nonnegative_number",
      reg_alpha = "nonnegative_number", reg_lambda = "positive_number",
      max_delta_step = "positive_number"),
    parameter_aliases = c(
      n_estimators = "num_boost_round", eta = "learning_rate",
      gamma = "min_split_loss", alpha = "reg_alpha",
      lambda = "reg_lambda"),
    required_parameters = character(),
    parameter_choices = list(task = c("binary", "regression")),
    description = paste0(
      "Native-tight DP XGBoost request with task-aware defaults ",
      "(backend not yet available)."),
    data_kinds = "tabular", available = FALSE, vetted = TRUE)

  invisible(TRUE)
}
