# Module: flwr CLI Integration
# Controls Flower runs via the flwr CLI.

#' Start a Flower run
#'
#' Fetches the model template from the server, builds a Flower App from
#' the recipe, then invokes \code{flwr run} against the running SuperLink.
#'
#' @param recipe A \code{dsflower_recipe} object.
#' @param conns DSI connections object. Used to fetch the model template
#'   from the server. If NULL, uses the connections stored during
#'   \code{ds.flower.nodes.init}.
#' @param app_dir Character; path to a pre-built app directory (optional).
#' @param run_config Named list; additional run config overrides.
#' @param verbose Logical; print flwr output (default TRUE).
#' @return A \code{dsflower_run} object with weights, history, and predictions.
#' @export
ds.flower.run.start <- function(recipe, conns = NULL, app_dir = NULL,
                                 run_config = list(), verbose = TRUE) {
  .require_flwr_cli()

  if (!inherits(recipe, "dsflower_recipe")) {
    stop("'recipe' must be a dsflower_recipe object.", call. = FALSE)
  }

  # Check SuperLink is running
  sl_info <- .dsflower_client_env$.superlink
  if (is.null(sl_info) || is.null(sl_info$process) || !sl_info$process$is_alive()) {
    stop("No SuperLink is running. Call ds.flower.superlink.start() first.",
         call. = FALSE)
  }

  # Results directory for model weights and metrics
  results_dir <- file.path(tempdir(), "dsflower_results",
                           format(Sys.time(), "%Y%m%d_%H%M%S"))
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  # Get connections for template fetching
  if (is.null(conns)) {
    conns <- .dsflower_client_env$.conns
  }
  if (is.null(conns)) {
    stop("'conns' is required to fetch templates from the server.", call. = FALSE)
  }

  # Set min_fit_clients and min_available_clients from number of connections
  n_clients <- length(conns)
  recipe$strategy$params$min_fit_clients <- as.integer(n_clients)
  recipe$strategy$params$min_available_clients <- as.integer(n_clients)

  # Build app if no pre-built dir provided
  if (is.null(app_dir)) {
    app_dir <- .build_flower_app(recipe, conns = conns,
                                  results_dir = results_dir)
  }

  # Build command: flwr run <app_dir> dsflower --stream
  # The "dsflower" arg refers to the named connection in config.toml
  args <- c("run", app_dir, "dsflower", "--stream")

  # Add run_config overrides
  for (nm in names(run_config)) {
    val <- run_config[[nm]]
    args <- c(args, "-c", paste0(nm, "=", val))
  }

  # Run via processx with FLWR_HOME pointing to our private config
  result <- processx::run(
    command = "flwr",
    args = args,
    env = c("current", FLWR_HOME = sl_info$flwr_home),
    error_on_status = FALSE,
    timeout = 3600  # 1 hour max
  )

  # Clean ANSI escape codes
  clean_stdout <- gsub("\033\\[[0-9;]*m", "", result$stdout)
  clean_stderr <- gsub("\033\\[[0-9;]*m", "", result$stderr)

  if (verbose) {
    if (nchar(clean_stdout) > 0) message(clean_stdout)
  }

  run_id <- .parse_run_id(clean_stdout)

  # Read saved weights and history from results dir
  weights <- .read_model_weights(results_dir)
  history <- .read_training_history(results_dir)

  structure(
    list(
      run_id      = run_id,
      status      = result$status,
      num_rounds  = recipe$num_rounds,
      model       = recipe$model$name,
      strategy    = recipe$strategy$name,
      weights     = weights,
      history     = history,
      results_dir = results_dir,
      app_dir     = app_dir,
      stdout      = clean_stdout,
      stderr      = clean_stderr
    ),
    class = "dsflower_run"
  )
}

#' List Flower runs
#'
#' Invokes \code{flwr list} to list runs.
#'
#' @return Character; output of flwr list.
#' @export
ds.flower.run.list <- function() {
  .require_flwr_cli()
  sl_info <- .dsflower_client_env$.superlink
  env <- if (!is.null(sl_info)) c("current", FLWR_HOME = sl_info$flwr_home) else "current"
  result <- processx::run("flwr", args = c("list"), env = env,
                          error_on_status = FALSE)
  result$stdout
}

#' Get Flower run logs
#'
#' Invokes \code{flwr log} for a specific run.
#'
#' @param run_id Character; the run ID.
#' @return Character; log output.
#' @export
ds.flower.run.logs <- function(run_id) {
  .require_flwr_cli()
  sl_info <- .dsflower_client_env$.superlink
  env <- if (!is.null(sl_info)) c("current", FLWR_HOME = sl_info$flwr_home) else "current"
  result <- processx::run("flwr", args = c("log", run_id), env = env,
                          error_on_status = FALSE)
  result$stdout
}

#' Stop a Flower run
#'
#' Invokes \code{flwr stop} for a specific run.
#'
#' @param run_id Character; the run ID.
#' @return Character; output of flwr stop.
#' @export
ds.flower.run.stop <- function(run_id) {
  .require_flwr_cli()
  sl_info <- .dsflower_client_env$.superlink
  env <- if (!is.null(sl_info)) c("current", FLWR_HOME = sl_info$flwr_home) else "current"
  result <- processx::run("flwr", args = c("stop", run_id), env = env,
                          error_on_status = FALSE)
  result$stdout
}

#' Parse run ID from flwr output
#'
#' @param stdout Character; stdout from flwr run.
#' @return Character; the run ID, or NULL.
#' @keywords internal
.parse_run_id <- function(stdout) {
  if (is.null(stdout) || !nzchar(stdout)) return(NULL)
  # Try various patterns Flower might use
  m <- regmatches(stdout, regexec("run[_-]?id[: =]+([a-zA-Z0-9_-]+)", stdout,
                                   ignore.case = TRUE))[[1]]
  if (length(m) >= 2) return(m[2])

  # Fallback: look for UUID-like pattern
  m2 <- regmatches(stdout, regexec(
    "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    stdout))[[1]]
  if (length(m2) >= 1) return(m2[1])

  # Fallback: numeric run ID
  m3 <- regmatches(stdout, regexec("run[_ ]*(\\d+)", stdout,
                                    ignore.case = TRUE))[[1]]
  if (length(m3) >= 2) return(m3[2])

  NULL
}

#' Read saved model weights from results directory
#' @param results_dir Character; path to the results directory.
#' @return A list of numeric arrays (one per parameter), or NULL.
#' @keywords internal
.read_model_weights <- function(results_dir) {
  path <- file.path(results_dir, "global_model.json")
  if (!file.exists(path)) return(NULL)

  raw <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  shapes <- raw[["__shapes__"]]
  round <- raw[["__round__"]]

  # Reconstruct numpy arrays as R matrices/vectors
  param_names <- setdiff(names(raw), c("__shapes__", "__round__"))
  param_names <- param_names[order(as.integer(param_names))]

  params <- lapply(seq_along(param_names), function(i) {
    vals <- unlist(raw[[param_names[i]]])
    shape <- unlist(shapes[[i]])
    if (length(shape) == 1) {
      array(vals, dim = shape)
    } else if (length(shape) == 2) {
      matrix(vals, nrow = shape[1], ncol = shape[2], byrow = TRUE)
    } else {
      array(vals, dim = shape)
    }
  })

  names(params) <- c("coef", "intercept")[seq_along(params)]
  attr(params, "round") <- round
  params
}

#' Read training history from results directory
#' @param results_dir Character; path to the results directory.
#' @return A data.frame with columns round, loss, n_clients, n_failures, or NULL.
#' @keywords internal
.read_training_history <- function(results_dir) {
  path <- file.path(results_dir, "history.json")
  if (!file.exists(path)) return(NULL)

  raw <- jsonlite::fromJSON(path, simplifyVector = TRUE)
  as.data.frame(raw)
}

#' Print a dsflower_run
#' @param x A dsflower_run object.
#' @param ... Additional arguments (ignored).
#' @export
print.dsflower_run <- function(x, ...) {
  cat("Federated Learning Run\n")
  cat("  Model:    ", x$model, "\n")
  cat("  Strategy: ", x$strategy, "\n")
  cat("  Rounds:   ", x$num_rounds, "\n")
  cat("  Status:   ", if (x$status == 0) "success" else "failed", "\n")

  if (!is.null(x$history)) {
    cat("\n  Loss per round:\n")
    for (i in seq_len(nrow(x$history))) {
      cat(sprintf("    round %d: %.6f (%d clients)\n",
        x$history$round[i], x$history$loss[i], x$history$n_clients[i]))
    }
  }

  if (!is.null(x$weights)) {
    cat("\n  Global model weights: ",
        length(x$weights), " parameter arrays\n")
    for (nm in names(x$weights)) {
      w <- x$weights[[nm]]
      cat(sprintf("    %s: %s\n", nm, paste(dim(w) %||% length(w), collapse = " x ")))
    }
  }

  cat("\n  Use ds.flower.save_model() to save the trained model.\n")
  invisible(x)
}

#' Save the global model from a training run
#'
#' Saves the federated model weights to a file. Supported formats:
#' \code{.rds} (R native), \code{.json} (portable).
#'
#' @param run A \code{dsflower_run} object.
#' @param path Character; file path to save to.
#' @return Invisible path.
#' @export
ds.flower.save_model <- function(run, path) {
  if (!inherits(run, "dsflower_run")) {
    stop("'run' must be a dsflower_run object.", call. = FALSE)
  }
  if (is.null(run$weights)) {
    stop("No model weights available in this run.", call. = FALSE)
  }

  model_data <- list(
    weights  = run$weights,
    history  = run$history,
    model    = run$model,
    strategy = run$strategy,
    rounds   = run$num_rounds,
    run_id   = run$run_id
  )

  ext <- tolower(tools::file_ext(path))
  if (ext == "rds") {
    saveRDS(model_data, path)
  } else if (ext == "json") {
    jsonlite::write_json(model_data, path, auto_unbox = TRUE, digits = 10)
  } else {
    stop("Unsupported format '.", ext, "'. Use .rds or .json.", call. = FALSE)
  }

  message("Model saved to ", path)
  invisible(path)
}
