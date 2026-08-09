# Module: Local Hyperparameter Optimization
# Optuna suggests parameters in one ephemeral local process. The objective is
# evaluated as an R function and is never serialized or sent to data nodes.

.DSFLOWER_LOCAL_HPO_PROTOCOL <- "dsflower-local-hpo-v1"

.hpo_scalar_number <- function(value, name) {
  if (!is.numeric(value) || is.logical(value) || length(value) != 1L ||
      is.na(value) || !is.finite(value)) {
    stop("'", name, "' must be one finite number.", call. = FALSE)
  }
  as.numeric(value)
}

.hpo_exact_integer <- function(value, name, minimum, maximum) {
  value <- .hpo_scalar_number(value, name)
  if (value != floor(value) || value < minimum || value > maximum) {
    minimum_label <- format(minimum, scientific = FALSE, trim = TRUE)
    maximum_label <- format(maximum, scientific = FALSE, trim = TRUE)
    stop("'", name, "' must be one integer in [", minimum_label, ", ",
         maximum_label, "].", call. = FALSE)
  }
  value
}

.hpo_scalar_logical <- function(value, name) {
  if (!is.logical(value) || length(value) != 1L || is.na(value)) {
    stop("'", name, "' must be TRUE or FALSE.", call. = FALSE)
  }
  value
}

.hpo_step_fits <- function(lower, upper, step) {
  ratio <- (upper - lower) / step
  is.finite(ratio) &&
    abs(ratio - round(ratio)) <= 1e-10 * max(1, abs(ratio))
}

#' Define a bounded floating-point HPO dimension
#'
#' @param lower,upper Finite numeric endpoints with `lower < upper`.
#' @param log Whether Optuna samples on a logarithmic scale. Logarithmic bounds
#'   must be positive and cannot be combined with `step`.
#' @param step Optional positive discretization step. It must divide the range
#'   exactly so Optuna never silently changes the upper endpoint.
#' @return A typed local HPO dimension for `ds.flower.hpo()`.
#' @rdname ds.flower.hpo.dimensions
#' @export
ds.flower.hpo.float <- function(lower, upper, log = FALSE, step = NULL) {
  lower <- .hpo_scalar_number(lower, "lower")
  upper <- .hpo_scalar_number(upper, "upper")
  log <- .hpo_scalar_logical(log, "log")
  if (lower >= upper) {
    stop("'lower' must be strictly smaller than 'upper'.", call. = FALSE)
  }
  if (log && lower <= 0) {
    stop("Logarithmic float bounds must be positive.", call. = FALSE)
  }
  if (!is.null(step)) {
    step <- .hpo_scalar_number(step, "step")
    if (log) {
      stop("'step' cannot be combined with logarithmic sampling.",
           call. = FALSE)
    }
    if (step <= 0 || !.hpo_step_fits(lower, upper, step)) {
      stop("'step' must be positive and divide the float range exactly.",
           call. = FALSE)
    }
  }
  structure(
    list(type = "float", lower = lower, upper = upper, log = log, step = step),
    class = "dsflower_hpo_dimension"
  )
}

#' Define a bounded integer HPO dimension
#'
#' @param lower,upper Integer endpoints with `lower < upper`.
#' @param log Whether Optuna samples on a logarithmic scale. This requires a
#'   positive lower endpoint and `step = 1`.
#' @param step Positive integer step that divides the range exactly.
#' @return A typed local HPO dimension for `ds.flower.hpo()`.
#' @rdname ds.flower.hpo.dimensions
#' @export
ds.flower.hpo.integer <- function(lower, upper, log = FALSE, step = 1L) {
  lower <- .hpo_exact_integer(
    lower, "lower", -.Machine$integer.max, .Machine$integer.max)
  upper <- .hpo_exact_integer(
    upper, "upper", -.Machine$integer.max, .Machine$integer.max)
  step <- .hpo_exact_integer(step, "step", 1, .Machine$integer.max)
  log <- .hpo_scalar_logical(log, "log")
  if (lower >= upper) {
    stop("'lower' must be strictly smaller than 'upper'.", call. = FALSE)
  }
  if (log && (lower < 1 || step != 1)) {
    stop("Logarithmic integer sampling requires lower >= 1 and step = 1.",
         call. = FALSE)
  }
  if (!.hpo_step_fits(lower, upper, step)) {
    stop("'step' must divide the integer range exactly.", call. = FALSE)
  }
  structure(
    list(type = "integer", lower = lower, upper = upper, log = log,
         step = step),
    class = "dsflower_hpo_dimension"
  )
}

#' Define a categorical HPO dimension
#'
#' @param values A non-empty atomic vector of unique finite values. Character,
#'   logical, integer and numeric choices are supported.
#' @return A typed local HPO dimension for `ds.flower.hpo()`.
#' @rdname ds.flower.hpo.dimensions
#' @export
ds.flower.hpo.categorical <- function(values) {
  if (!is.atomic(values) || is.factor(values) || is.raw(values) ||
      !length(values) || anyNA(values) || anyDuplicated(values) ||
      !(is.character(values) || is.logical(values) || is.numeric(values))) {
    stop("'values' must be a non-empty atomic vector of unique scalar choices.",
         call. = FALSE)
  }
  if (is.numeric(values) && any(!is.finite(values))) {
    stop("Numeric categorical choices must be finite.", call. = FALSE)
  }
  structure(
    list(type = "categorical", values = values),
    class = "dsflower_hpo_dimension"
  )
}

.hpo_request <- function(space, n_trials, direction, seed) {
  if (!is.list(space) || !length(space)) {
    stop("'space' must be a non-empty named list of HPO dimensions.",
         call. = FALSE)
  }
  parameter_names <- names(space)
  if (is.null(parameter_names) || anyNA(parameter_names) ||
      any(!nzchar(parameter_names)) || anyDuplicated(parameter_names)) {
    stop("'space' must have non-empty, unique parameter names.", call. = FALSE)
  }
  valid <- vapply(space, inherits, logical(1), what = "dsflower_hpo_dimension")
  if (!all(valid)) {
    stop("Every search-space value must be created by a ds.flower.hpo.*() ",
         "dimension constructor.", call. = FALSE)
  }
  direction <- match.arg(direction, c("minimize", "maximize"))
  n_trials <- .hpo_exact_integer(
    n_trials, "n_trials", 1, 1000000)
  seed <- .hpo_exact_integer(seed, "seed", 0, 4294967295)

  parameter_names <- sort(parameter_names, method = "radix")
  wire_space <- stats::setNames(lapply(parameter_names, function(name) {
    dimension <- unclass(space[[name]])
    if (identical(dimension$type, "categorical")) {
      dimension$values <- unname(as.list(dimension$values))
    }
    dimension
  }), parameter_names)
  list(
    protocol = .DSFLOWER_LOCAL_HPO_PROTOCOL,
    direction = direction,
    seed = seed,
    n_trials = n_trials,
    space = wire_space
  )
}

.local_hpo_python_cmd <- function() {
  primary <- tryCatch(.client_python_cmd(), error = function(e) "")
  candidates <- unique(c(
    primary,
    unname(Sys.which(c("python3", "python")))
  ))
  candidates <- candidates[nzchar(candidates)]
  probe <- "import optuna,sys;sys.exit(0 if optuna.__version__=='4.8.0' else 1)"
  for (python in candidates) {
    result <- tryCatch(
      processx::run(
        python, c("-I", "-c", probe), error_on_status = FALSE,
        timeout = 10
      ),
      error = function(e) NULL
    )
    if (!is.null(result) && identical(result$status, 0L)) return(python)
  }
  stop("Local HPO requires Optuna 4.8.0 in the dsFlowerClient Python ",
       "environment. Reinstall dsFlowerClient or install 'optuna==4.8.0' ",
       "into that environment.", call. = FALSE)
}

.hpo_bounded_message <- function(value, fallback) {
  if (!is.character(value) || length(value) != 1L || is.na(value) ||
      !nzchar(value)) {
    value <- fallback
  }
  value <- gsub("[\r\n]+", " ", value)
  substr(value, 1L, 512L)
}

.hpo_protocol_timeout <- function() {
  value <- getOption("dsflower.hpo_protocol_timeout_secs", 60)
  value <- .hpo_scalar_number(value, "dsflower.hpo_protocol_timeout_secs")
  if (value <= 0) {
    stop("'dsflower.hpo_protocol_timeout_secs' must be positive.",
         call. = FALSE)
  }
  value
}

.hpo_read_event <- function(process) {
  deadline <- Sys.time() + .hpo_protocol_timeout()
  repeat {
    lines <- process$read_output_lines()
    if (length(lines)) {
      if (length(lines) != 1L) {
        stop("The local HPO backend emitted an invalid protocol response.",
             call. = FALSE)
      }
      event <- tryCatch(
        jsonlite::fromJSON(lines[[1]], simplifyVector = FALSE),
        error = function(e) NULL
      )
      if (!is.list(event) || !is.character(event$event) ||
          length(event$event) != 1L) {
        stop("The local HPO backend emitted malformed JSON.", call. = FALSE)
      }
      if (identical(event$event, "error")) {
        message <- .hpo_bounded_message(
          event$message, "unspecified backend error")
        stop("Local HPO backend error: ", message, call. = FALSE)
      }
      return(event)
    }
    if (!process$is_alive()) {
      stderr <- .hpo_bounded_message(
        paste(process$read_all_error_lines(), collapse = " "),
        "the backend exited without a response")
      stop("Local HPO backend failed: ", stderr, call. = FALSE)
    }
    if (Sys.time() >= deadline) {
      stop("Local HPO backend protocol timeout.", call. = FALSE)
    }
    process$poll_io(100L)
  }
}

.hpo_write <- function(process, value) {
  line <- as.character(jsonlite::toJSON(
    value, auto_unbox = TRUE, null = "null", digits = NA
  ))
  process$write_input(paste0(line, "\n"))
  invisible(NULL)
}

.hpo_trial_params <- function(event, expected_number, parameter_names) {
  if (!identical(event$event, "trial") ||
      !is.numeric(event$number) || length(event$number) != 1L ||
      event$number != expected_number || !is.list(event$params) ||
      !identical(names(event$params), parameter_names)) {
    stop("The local HPO backend returned an invalid trial.", call. = FALSE)
  }
  event$params
}

#' Optimize an explicit objective locally with Optuna
#'
#' Runs one sequential Optuna TPE study entirely on the researcher's machine.
#' Optuna uses ephemeral in-memory storage. For each suggestion this function
#' calls `objective(params)` in the current R process; the function itself
#' is never serialized, uploaded, or sent to a data node. It must return one
#' finite numeric value.
#'
#' The objective may explicitly call `ds.flower.fit()` and a suitable
#' private validation API. Each such call remains an ordinary independent
#' server-enforced training or validation release. HPO does not weaken or alter
#' node privacy policy. `n_trials` is only the geometry of this local call:
#' it is not a rate limit, quota, catalog permission, or persistent budget.
#'
#' Repeating a study with Optuna 4.8.0, the same seed, search space, and
#' sequence of objective values reproduces its suggestions. Reproducibility of
#' the returned values additionally depends on the objective itself.
#' Static typed spaces are intentional: conditional search logic would require
#' exposing Python Trial semantics to R or serializing code, neither of which
#' is part of this data-only local bridge.
#'
#' @param objective Local R function accepting one named parameter list and
#'   returning one finite numeric scalar.
#' @param space Non-empty named list of dimensions created with
#'   `ds.flower.hpo.float()`, `ds.flower.hpo.integer()`, or
#'   `ds.flower.hpo.categorical()`. Parameter names are unrestricted by a
#'   model catalog.
#' @param n_trials Positive integer number of trials in this local study, at
#'   most one million. This per-process memory ceiling is not a historical
#'   quota or a limit on the number of HPO calls.
#' @param direction Either `"minimize"` or `"maximize"`.
#' @param seed Integer in `[0, 2^32 - 1]` controlling only local Optuna
#'   suggestions. It is unrelated to node-owned privacy randomness.
#' @return A `dsflower_hpo` object with the best value and parameters,
#'   complete local trial history, seed, direction, and Optuna version.
#' @export
ds.flower.hpo <- function(objective, space, n_trials = 20L,
                          direction = c("minimize", "maximize"), seed = 0L) {
  if (!is.function(objective)) {
    stop("'objective' must be a local R function.", call. = FALSE)
  }
  request <- .hpo_request(space, n_trials, direction, seed)
  helper <- system.file(
    "python", "local_hpo_helper.py", package = "dsFlowerClient"
  )
  if (!nzchar(helper) || !file.exists(helper)) {
    stop("The packaged local HPO helper is missing.", call. = FALSE)
  }
  process <- processx::process$new(
    command = .local_hpo_python_cmd(),
    args = c("-u", "-I", "-X", "utf8", helper),
    stdin = "|", stdout = "|", stderr = "|",
    cleanup = TRUE, cleanup_tree = TRUE
  )
  on.exit(tryCatch(
    if (process$is_alive()) process$kill_tree(),
    error = function(e) NULL
  ), add = TRUE)
  .hpo_write(process, request)

  parameter_names <- names(request$space)
  values <- numeric(request$n_trials)
  params <- vector("list", request$n_trials)
  for (index in seq_len(request$n_trials)) {
    event <- .hpo_read_event(process)
    params[[index]] <- .hpo_trial_params(
      event, expected_number = index - 1, parameter_names = parameter_names)
    value <- tryCatch(
      objective(params[[index]]),
      error = function(e) {
        message <- .hpo_bounded_message(
          conditionMessage(e), "unspecified objective error")
        stop("Local HPO objective failed in trial ", index - 1, ": ",
             message, call. = FALSE)
      }
    )
    values[[index]] <- .hpo_scalar_number(value, "objective result")
    .hpo_write(process, list(
      event = "value", number = index - 1, value = values[[index]]
    ))
  }

  complete <- .hpo_read_event(process)
  if (!identical(complete$event, "complete") ||
      !identical(names(complete), c("event", "optuna_version")) ||
      !is.character(complete$optuna_version) ||
      length(complete$optuna_version) != 1L ||
      !identical(complete$optuna_version, "4.8.0")) {
    stop("The local HPO backend returned an invalid completion record.",
         call. = FALSE)
  }
  process$wait(timeout = 5000L)
  if (!identical(process$get_exit_status(), 0L)) {
    stop("The local HPO backend did not exit cleanly.", call. = FALSE)
  }

  history <- lapply(seq_len(request$n_trials), function(index) {
    list(number = index - 1, value = values[[index]], params = params[[index]])
  })
  best_index <- if (identical(request$direction, "minimize")) {
    which.min(values)
  } else {
    which.max(values)
  }
  result <- list(
    direction = request$direction,
    seed = request$seed,
    n_trials = request$n_trials,
    sampler = "TPESampler",
    optuna_version = complete$optuna_version,
    best_number = as.integer(best_index - 1L),
    best_value = values[[best_index]],
    best_params = params[[best_index]],
    trials = history
  )
  class(result) <- "dsflower_hpo"
  result
}

#' Print a local HPO result
#' @param x A `dsflower_hpo` object.
#' @param ... Ignored.
#' @return Invisibly returns `x`.
#' @export
print.dsflower_hpo <- function(x, ...) {
  cat("dsflower_hpo\n")
  cat("  Backend:  Optuna ", x$optuna_version, " / ", x$sampler,
      " / in-memory\n", sep = "")
  cat("  Direction: ", x$direction, "\n", sep = "")
  cat("  Trials:    ", x$n_trials, " (local execution geometry)\n", sep = "")
  cat("  Best:      trial ", x$best_number, ", value ",
      format(x$best_value, digits = 8), "\n", sep = "")
  invisible(x)
}
