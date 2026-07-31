# Module: flwr CLI Integration
# Controls Flower runs via the flwr CLI.

#' Start a Flower run
#'
#' Fetches the model template from the server, builds a Flower App from
#' the recipe, then invokes \code{flwr run} against the running SuperLink.
#' Model weights and training history are automatically saved to
#' \code{output_dir} after training completes.
#'
#' @param recipe A \code{dsflower_recipe} object.
#' @param conns DSI connections object. Used to fetch the model template
#'   from the server. If NULL, uses the connections stored during
#'   \code{ds.flower.nodes.init}.
#' @param app_dir Character; path to a pre-built app directory (optional).
#' @param run_config Named list; additional run config overrides.
#' @param output_dir Character; persistent directory for model output.
#'   Defaults to \code{"dsflower_output/<timestamp>"} in the working directory.
#' @param output_name Optional name for the persisted model artifact.
#' @param results_dir Character; temporary directory where the Flower ServerApp
#'   writes model artefacts. Usually generated automatically.
#' @param symbol Character; server-side Flower handle symbol. Low-level callers
#'   that initialise the default handle can leave this as \code{"flower"}.
#' @param verbose Logical; print flwr output (default TRUE).
#' @param silent Logical; suppress progress feedback.
#' @return A \code{dsflower_run} object with weights, history, and predictions.
#' @export
ds.flower.run.start <- function(recipe, conns = NULL, app_dir = NULL,
                                 run_config = list(), output_dir = NULL,
                                 output_name = NULL,
                                 results_dir = NULL,
                                 symbol = "flower",
                                 verbose = FALSE, silent = FALSE) {
  .require_flwr_cli()

  old_opt <- options(dsflower.silent = isTRUE(silent))
  on.exit(options(old_opt), add = TRUE)

  if (!inherits(recipe, "dsflower_recipe")) {
    stop("'recipe' must be a dsflower_recipe object.", call. = FALSE)
  }

  # Check SuperLink is running
  sl_info <- .dsflower_client_env$.superlink
  if (is.null(sl_info) || is.null(sl_info$process) || !sl_info$process$is_alive()) {
    stop("No SuperLink is running. Call ds.flower.superlink.start() first.",
         call. = FALSE)
  }

  # Results directory for model weights and metrics.
  if (is.null(results_dir)) {
    results_dir <- file.path(tempdir(), "dsflower_results",
                             format(Sys.time(), "%Y%m%d_%H%M%S"))
  }
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

  # DP (epsilon/delta/clipping + mechanism) is decided + enforced by each node; the
  # client carries no privacy spec. Always include every node each round.
  recipe$strategy$params$fraction_fit <- 1.0

  # The Tier-1 harness is a torch app; ensure its runtime client-side (the
  # ServerApp builds the initial model).
  .ensure_client_framework("pytorch")

  # The app dir is always pre-built by the submission pipeline (a dsflower_runner
  # FAB, content-hash-pinned against the node-resident canonical runner). There is
  # no legacy in-builder fallback.
  if (is.null(app_dir)) {
    stop("'app_dir' is required: build the submission app first ",
         "(ds.flower.submit() / ds.flower.fit() do this for you).", call. = FALSE)
  }

  # Build command: flwr run <app_dir> dsflower --stream
  args <- c("run", app_dir, "dsflower", "--stream")

  # Add run_config overrides
  for (nm in names(run_config)) {
    val <- run_config[[nm]]
    args <- c(args, "-c", paste0(nm, "=", val))
  }

  # Run via venv flwr with FLWR_HOME pointing to our private config.
  # Trees are a single federated round (each site trains its full DP-GBDT locally,
  # then the boosters are bagged), so num_rounds does not apply to that track.
  is_trees <- identical(recipe$model$track, "trees")
  eff_rounds <- if (is_trees) 1L else recipe$num_rounds
  .dsf_msg("Training ", recipe$model$template, " across ", n_clients,
           " site(s) for ", eff_rounds, " round(s)...")
  flwr_cmd <- .client_flwr_cmd()
  # The trees track has no flwr [ROUND] stream to parse, so announce its one
  # round here; the neural/egress tracks stream [ROUND k/N] during training.
  if (is_trees)
    .dsf_msg("  Round 1/1: each site trains its DP-GBDT locally, then the boosters are bagged...")
  # PYTHONUNBUFFERED so the flwr child flushes its [ROUND k/N] log lines as they
  # happen (block-buffered pipes otherwise hold them until exit, which hides the
  # per-round progress under a non-interactive front-end such as knitr).
  result <- .run_flwr_with_artifact_watchdog(
    command = flwr_cmd,
    args = args,
    env = .client_venv_env(extra = c(FLWR_HOME = sl_info$flwr_home,
                                      PYTHONUNBUFFERED = "1")),
    results_dir = results_dir,
    num_rounds = recipe$num_rounds,
    expect_artifacts = !isTRUE(recipe$evaluation_only)
  )

  # Clean ANSI escape codes
  clean_stdout <- gsub("\033\\[[0-9;]*m", "", result$stdout)
  clean_stderr <- gsub("\033\\[[0-9;]*m", "", result$stderr)

  if (verbose) {
    if (nchar(clean_stdout) > 0) message(clean_stdout)
  }

  run_id <- .parse_run_id(clean_stdout)

  # Read saved weights and history from results dir.
  # The ServerApp writes files asynchronously -- retry briefly if not found.
  weights <- .read_model_weights(results_dir)
  if (is.null(weights) && result$status == 0L) {
    Sys.sleep(2)
    weights <- .read_model_weights(results_dir)
  }
  history <- .read_training_history(results_dir)
  runtime_status <- .flower_runtime_status(
    cli_status = result$status,
    stdout = clean_stdout,
    stderr = clean_stderr,
    weights = weights,
    history = history,
    expect_artifacts = !isTRUE(recipe$evaluation_only)
  )

  if (!is.null(weights) || !is.null(history)) {
    n_done <- if (is.data.frame(history) && nrow(history)) nrow(history) else recipe$num_rounds
    .dsf_msg("Training complete: ", n_done, " round(s) aggregated across ",
             n_clients, " site(s) with strategy '", recipe$strategy$name,
             "'. Differential privacy was enforced by the nodes.")
  }

  # Generate a unique model ID for identification
  model_id <- paste0(
    recipe$model$template, "_",
    recipe$strategy$name, "_",
    recipe$num_rounds, "r_",
    format(Sys.time(), "%Y%m%d_%H%M%S")
  )

  # Auto-save to persistent output_dir. The caller may set the parent path
  # (output_dir) and/or the model's folder name (output_name); both default
  # sensibly. Missing directories are created, and the final path is made
  # absolute so prediction resolves the model regardless of the working dir.
  saved_path <- NULL
  if (!is.null(weights) || !is.null(history)) {
    parent    <- if (is.null(output_dir)) file.path(".", "dsflower_output") else output_dir
    save_name <- if (is.null(output_name)) model_id else output_name
    output_dir <- file.path(parent, save_name)
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    output_dir <- normalizePath(output_dir, winslash = "/", mustWork = FALSE)

    model_data <- list(
      model_id   = model_id,
      weights    = weights,
      history    = history,
      model      = recipe$model$name,
      template   = recipe$model$template,
      framework  = recipe$model$framework,
      strategy   = recipe$strategy$name,
      privacy    = "server-enforced-dp",
      num_rounds = recipe$num_rounds,
      target_levels = recipe$target_levels %||% NULL,
      target_bounds = recipe$target_bounds %||% NULL,
      run_id     = run_id,
      created_at = Sys.time(),
      n_clients  = length(conns)
    )
    saved_path <- file.path(output_dir, paste0(save_name, ".rds"))
    saveRDS(model_data, saved_path)

    # Copy all output files (JSON, native format, history)
    output_files <- list.files(results_dir, full.names = TRUE)
    for (f in output_files) {
      file.copy(f, file.path(output_dir, basename(f)), overwrite = TRUE)
    }

    # Write metadata for listing/identification
    meta <- list(
      model_id   = model_id,
      model      = recipe$model$name,
      template   = recipe$model$template,
      strategy   = recipe$strategy$name,
      privacy    = "server-enforced-dp",
      num_rounds = recipe$num_rounds,
      n_clients  = length(conns),
      features   = recipe$features,   # training feature order, so predict can align newdata
      # Public preprocessing constants used by the node. New runs record lower/upper
      # so prediction can repeat the same clip-to-bounds + affine transform. Keep the
      # legacy mean/SD fields readable for models trained before public bounds existed.
      feature_means = recipe$feature_means,
      feature_sds   = recipe$feature_sds,
      feature_lower = recipe$feature_lower,
      feature_upper = recipe$feature_upper,
      target_levels = recipe$target_levels %||% NULL,
      target_bounds = recipe$target_bounds %||% NULL,
      created_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
      status     = if (runtime_status == 0L) "success" else "failed"
    )
    jsonlite::write_json(meta, file.path(output_dir, "metadata.json"),
                         auto_unbox = TRUE, pretty = TRUE)

    .dsf_msg("Model saved to ", output_dir)
  }

  structure(
    list(
      model_id    = model_id,
      run_id      = run_id,
      status      = runtime_status,
      cli_status  = result$status,
      num_rounds  = recipe$num_rounds,
      model       = recipe$model$name,
      strategy    = recipe$strategy$name,
      weights     = weights,
      history     = history,
      output_dir  = output_dir,
      saved_path  = saved_path,
      results_dir = results_dir,
      app_dir     = app_dir,
      stdout      = clean_stdout,
      stderr      = clean_stderr
    ),
    class = "dsflower_run"
  )
}

.flwr_run_timeout_secs <- function() {
  opt <- getOption("dsflower.run_timeout_secs", NULL)
  env <- Sys.getenv("DSFLOWER_RUN_TIMEOUT_SECS", unset = NA_character_)
  raw <- opt %||% if (!is.na(env) && nzchar(env)) env else 3600
  val <- suppressWarnings(as.numeric(raw))
  if (is.na(val) || val <= 0) 3600 else val
}

.flwr_artifact_watchdog_secs <- function() {
  opt <- getOption("dsflower.artifact_watchdog_grace_secs", NULL)
  env <- Sys.getenv("DSFLOWER_ARTIFACT_WATCHDOG_GRACE_SECS", unset = NA_character_)
  raw <- opt %||% if (!is.na(env) && nzchar(env)) env else 10
  val <- suppressWarnings(as.numeric(raw))
  if (is.na(val) || val < 0) 10 else val
}

.flwr_weight_read_max_bytes <- function() {
  opt <- getOption("dsflower.weight_read_max_bytes", NULL)
  env <- Sys.getenv("DSFLOWER_WEIGHT_READ_MAX_BYTES", unset = NA_character_)
  raw <- opt %||% if (!is.na(env) && nzchar(env)) env else 50 * 1024^2
  val <- suppressWarnings(as.numeric(raw))
  if (is.na(val) || val < 0) 50 * 1024^2 else val
}

.model_artifact_exists <- function(results_dir) {
  any(file.exists(file.path(
    results_dir,
    c("global_model.json", "global_model.skipped.json",
      "model.pt", "model.npz")
  )))
}

.training_artifacts_complete <- function(results_dir, num_rounds,
                                         expect_artifacts = TRUE) {
  history <- .read_training_history(results_dir)
  if (is.null(history) || !NROW(history)) return(FALSE)

  if ("round" %in% names(history)) {
    rounds <- suppressWarnings(as.integer(history$round))
    if (!any(rounds >= as.integer(num_rounds), na.rm = TRUE)) return(FALSE)
  }

  if (!isTRUE(expect_artifacts)) return(TRUE)
  .model_artifact_exists(results_dir)
}

# Emit a status/progress line to the user unless silence was requested. Progress
# is opt-out (ds.flower.fit(..., silent = TRUE)); the option is set by the entry
# points and read here so deep helpers need no extra argument threading.
.dsf_msg <- function(...) {
  if (!isTRUE(getOption("dsflower.silent", FALSE))) message(...)
}

# Parse freshly streamed flwr lines and announce each round as it starts. The
# ServerApp logs "[ROUND k/N]" per federated round (neural/egress tracks); the
# one-round trees track has no such marker and relies on the start/complete
# lines instead. `seen` tracks already-announced rounds across poll iterations.
.emit_round_progress <- function(lines, num_rounds, seen) {
  if (!length(lines) || isTRUE(getOption("dsflower.silent", FALSE))) return(invisible())
  for (ln in lines) {
    m <- regmatches(ln, regexpr("\\[ROUND[[:space:]]+[0-9]+", ln))
    if (!length(m)) next
    k <- suppressWarnings(as.integer(regmatches(m, regexpr("[0-9]+", m))))
    if (is.na(k) || k %in% seen$rounds) next
    seen$rounds <- c(seen$rounds, k)
    message("  Round ", k, "/", num_rounds, ": aggregating updates from the sites...")
  }
  invisible()
}

.stop_flwr_run <- function(flwr_cmd, run_id, env) {
  if (is.null(run_id) || !nzchar(run_id)) return(invisible(FALSE))
  tryCatch({
    processx::run(flwr_cmd, args = c("stop", run_id), env = env,
                  error_on_status = FALSE, timeout = 15)
    TRUE
  }, error = function(e) FALSE)
}

.run_flwr_with_artifact_watchdog <- function(command, args, env,
                                             results_dir, num_rounds,
                                             expect_artifacts = TRUE) {
  timeout <- .flwr_run_timeout_secs()
  grace <- .flwr_artifact_watchdog_secs()
  deadline <- Sys.time() + timeout
  artifact_deadline <- NULL
  completed_from_artifacts <- FALSE
  run_id <- NULL
  stdout <- character()
  stderr <- character()

  proc <- processx::process$new(
    command = command,
    args = args,
    env = env,
    stdout = "|",
    stderr = "|",
    cleanup = TRUE,
    cleanup_tree = TRUE
  )

  seen <- new.env(parent = emptyenv()); seen$rounds <- integer(0)

  repeat {
    out <- proc$read_output_lines()
    err <- proc$read_error_lines()
    if (length(out)) stdout <- c(stdout, out)
    if (length(err)) stderr <- c(stderr, err)
    .emit_round_progress(c(out, err), num_rounds, seen)

    if (is.null(run_id)) {
      run_id <- .parse_run_id(paste(c(stdout, stderr), collapse = "\n"))
    }

    if (proc$is_alive() &&
        .training_artifacts_complete(results_dir, num_rounds, expect_artifacts)) {
      completed_from_artifacts <- TRUE
      if (is.null(artifact_deadline)) {
        artifact_deadline <- Sys.time() + grace
        .stop_flwr_run(command, run_id, env)
      } else if (Sys.time() >= artifact_deadline) {
        proc$kill_tree()
      }
    }

    if (!proc$is_alive()) break

    if (Sys.time() >= deadline) {
      .stop_flwr_run(command, run_id, env)
      Sys.sleep(1)
      if (proc$is_alive()) proc$kill_tree()
      stderr <- c(stderr, paste0("dsFlower run timeout after ", timeout, " seconds."))
      break
    }

    # If the DSI tunnel is active, carry a batch of Fleet-API bytes between the
    # SuperLink and the nodes; the aggregate round-trip paces the loop. Otherwise
    # just idle one second.
    if (!.tunnel_pump()) Sys.sleep(1)
  }

  out <- proc$read_all_output_lines()
  err <- proc$read_all_error_lines()
  if (length(out)) stdout <- c(stdout, out)
  if (length(err)) stderr <- c(stderr, err)
  .emit_round_progress(c(out, err), num_rounds, seen)  # catch any buffered tail

  status <- proc$get_exit_status()
  if (is.null(status)) status <- if (Sys.time() >= deadline) 124L else 0L
  if (isTRUE(completed_from_artifacts)) status <- 0L

  list(
    status = as.integer(status),
    stdout = paste(stdout, collapse = "\n"),
    stderr = paste(stderr, collapse = "\n")
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
  extra <- if (!is.null(sl_info)) c(FLWR_HOME = sl_info$flwr_home) else NULL
  result <- processx::run(.client_flwr_cmd(), args = c("list"),
                          env = .client_venv_env(extra = extra),
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
  extra <- if (!is.null(sl_info)) c(FLWR_HOME = sl_info$flwr_home) else NULL
  result <- processx::run(.client_flwr_cmd(), args = c("log", run_id),
                          env = .client_venv_env(extra = extra),
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
  extra <- if (!is.null(sl_info)) c(FLWR_HOME = sl_info$flwr_home) else NULL
  result <- processx::run(.client_flwr_cmd(), args = c("stop", run_id),
                          env = .client_venv_env(extra = extra),
                          error_on_status = FALSE)
  result$stdout
}

#' Compute SHA-256 hash of the Python files in the built app
#'
#' Must match the server-side \code{.compute_template_hash()} algorithm.
#'
#' @param app_dir Character; path to the built app directory.
#' @param template_name Character; template name (subdirectory name).
#' @return Character; hex-encoded SHA-256 hash.
#' @keywords internal
.compute_app_hash <- function(app_dir, template_name) {
  pkg_dir <- file.path(app_dir, template_name)
  py_files <- sort(list.files(pkg_dir, pattern = "\\.py$",
                               full.names = FALSE))

  blob <- raw(0)
  for (fname in py_files) {
    content <- readBin(file.path(pkg_dir, fname), "raw",
                       file.info(file.path(pkg_dir, fname))$size)
    blob <- c(blob, charToRaw(fname), charToRaw("\n"), content, as.raw(0x00))
  }

  digest::digest(blob, algo = "sha256", serialize = FALSE)
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

.parse_flower_exit_code <- function(text) {
  if (is.null(text) || !nzchar(text)) return(NULL)
  m <- regmatches(text, regexec("Exit Code:\\s*([0-9]+)", text))[[1]]
  if (length(m) >= 2L) return(as.integer(m[[2L]]))
  NULL
}

.flower_runtime_status <- function(cli_status, stdout = "", stderr = "",
                                   weights = NULL, history = NULL,
                                   expect_artifacts = TRUE) {
  cli_status <- as.integer(cli_status %||% 0L)
  combined <- paste(stdout %||% "", stderr %||% "", sep = "\n")
  flower_exit <- .parse_flower_exit_code(combined)
  has_server_error <- grepl(
    paste(c(
      "ServerApp raised an exception",
      "ClientApp raised an exception",
      "An unhandled exception occurred",
      "Traceback \\(most recent call last\\)"
    ), collapse = "|"),
    combined
  )

  if (!is.null(flower_exit) && flower_exit != 0L) {
    return(flower_exit)
  }
  if (cli_status != 0L) {
    return(cli_status)
  }
  if (has_server_error) {
    return(1L)
  }
  if (isTRUE(expect_artifacts) && is.null(weights) && is.null(history)) {
    return(1L)
  }
  0L
}

#' Read saved model weights from results directory
#' @param results_dir Character; path to the results directory.
#' @return A list of numeric arrays (one per parameter), or NULL.
#' @keywords internal
.read_model_weights <- function(results_dir) {
  path <- file.path(results_dir, "global_model.json")
  if (!file.exists(path)) return(NULL)
  size <- file.info(path)$size
  max_bytes <- .flwr_weight_read_max_bytes()
  if (is.finite(size) && !is.na(size) && size > max_bytes) {
    message("Skipping in-memory weight load for large global_model.json (",
            round(size / 1024^2, 1), " MiB). Native artifacts remain in ",
            results_dir, ".")
    return(NULL)
  }

  raw <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  if (identical(raw[["model_type"]], "xgboost")) {
    return(raw)
  }
  shapes <- raw[["__shapes__"]]
  round <- raw[["__round__"]]
  if (is.null(shapes)) return(NULL)

  # Reconstruct numpy arrays as R matrices/vectors
  param_names <- setdiff(names(raw), c("__shapes__", "__round__"))
  param_names <- param_names[order(as.integer(param_names))]

  params <- lapply(seq_along(param_names), function(i) {
    vals <- unlist(raw[[param_names[i]]])
    shape <- unlist(shapes[[i]])
    if (is.null(shape) || length(shape) == 0 || any(shape == 0)) {
      # Scalar or empty tensor (e.g. BatchNorm num_batches_tracked)
      vals
    } else if (length(shape) == 1) {
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
#' @return A data.frame with columns round, n_failures and (when available)
#'   n_examples, or NULL. Loss is not released by the nodes (disclosure backstop).
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
  cat("  Model ID: ", x$model_id, "\n")
  cat("  Model:    ", x$model, "\n")
  cat("  Strategy: ", x$strategy, "\n")
  cat("  Rounds:   ", x$num_rounds, "\n")
  cat("  Status:   ", if (x$status == 0) "success" else "failed", "\n")

  # Per-round summary. Loss is intentionally not released by the nodes (a
  # disclosure backstop: only a bucketed example count leaves), so we report
  # what the history actually carries. Measure utility with ds.flower.predict.
  if (!is.null(x$history) && is.data.frame(x$history) && nrow(x$history)) {
    cat("\n  Rounds:\n")
    has_ex <- "n_examples" %in% names(x$history)
    for (i in seq_len(nrow(x$history))) {
      ex <- if (has_ex) x$history$n_examples[i] else NA
      cat(sprintf("    round %d aggregated%s\n", x$history$round[i],
        if (!is.na(ex)) sprintf(" (~%s examples)", ex) else ""))
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

  if (!is.null(x$output_dir)) {
    cat("\n  Output: ", x$output_dir, "\n")
  }
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
    model_id = run$model_id,
    weights  = run$weights,
    history  = run$history,
    model    = run$model,
    strategy = run$strategy,
    rounds   = run$num_rounds,
    run_id   = run$run_id,
    saved_at = Sys.time()
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

#' List saved models
#'
#' Scans the output directory for previously saved models and returns
#' a summary data.frame with metadata for each.
#'
#' @param base_dir Character; base directory to scan.
#'   Defaults to \code{"./dsflower_output"}.
#' @return A data.frame with columns: model_id, model, template, strategy,
#'   privacy, num_rounds, n_clients, created_at, status, path.
#' @export
ds.flower.models <- function(base_dir = file.path(".", "dsflower_output")) {
  if (!dir.exists(base_dir)) {
    return(data.frame(
      model_id = character(0), model = character(0),
      template = character(0), strategy = character(0),
      privacy = character(0), num_rounds = integer(0),
      n_clients = integer(0), created_at = character(0),
      status = character(0), path = character(0),
      stringsAsFactors = FALSE
    ))
  }

  subdirs <- list.dirs(base_dir, recursive = FALSE, full.names = TRUE)
  rows <- list()
  # Coerce any metadata field to a single scalar: %||% only replaces NULL, so an empty
  # (length-0) field from a partial/old metadata.json would make data.frame() error with
  # "differing number of rows". This keeps the row (NA for the missing field) instead.
  s1 <- function(x, default) { x <- x %||% default; if (length(x) == 0L) default else x[[1]] }

  for (d in subdirs) {
    meta_path <- file.path(d, "metadata.json")
    if (!file.exists(meta_path)) next
    meta <- tryCatch(
      jsonlite::fromJSON(meta_path, simplifyVector = TRUE),
      error = function(e) NULL
    )
    if (is.null(meta)) next
    rows[[length(rows) + 1L]] <- data.frame(
      model_id   = s1(meta$model_id, basename(d)),
      model      = s1(meta$model, NA_character_),
      template   = s1(meta$template, NA_character_),
      strategy   = s1(meta$strategy, NA_character_),
      privacy    = s1(meta$privacy, NA_character_),
      num_rounds = as.integer(s1(meta$num_rounds, NA_integer_)),
      n_clients  = as.integer(s1(meta$n_clients, NA_integer_)),
      created_at = s1(meta$created_at, NA_character_),
      status     = s1(meta$status, NA_character_),
      path       = d,
      stringsAsFactors = FALSE
    )
  }

  if (length(rows) == 0L) {
    return(data.frame(
      model_id = character(0), model = character(0),
      template = character(0), strategy = character(0),
      privacy = character(0), num_rounds = integer(0),
      n_clients = integer(0), created_at = character(0),
      status = character(0), path = character(0),
      stringsAsFactors = FALSE
    ))
  }

  do.call(rbind, rows)
}

#' Load a saved model
#'
#' Reads model weights and metadata from a previously saved output directory
#' or \code{.rds} file.
#'
#' @param path Character; path to the model directory or \code{.rds} file.
#' @return A list with model_id, weights, history, and metadata.
#' @export
ds.flower.load_model <- function(path) {
  if (dir.exists(path)) {
    rds_path <- file.path(path, "model.rds")
    if (!file.exists(rds_path)) {
      # The bundle is named after the model (output_name.rds); fall back to the
      # single .rds in the directory when the legacy model.rds is absent.
      cand <- list.files(path, pattern = "\\.rds$", full.names = TRUE)
      if (length(cand) != 1L)
        stop("No model .rds found in ", path, call. = FALSE)
      rds_path <- cand
    }
    return(readRDS(rds_path))
  }

  if (file.exists(path)) {
    ext <- tolower(tools::file_ext(path))
    if (ext == "rds") return(readRDS(path))
    if (ext == "json") {
      return(jsonlite::fromJSON(path, simplifyVector = FALSE))
    }
    stop("Unsupported format: ", ext, call. = FALSE)
  }

  stop("Path not found: ", path, call. = FALSE)
}

#' Delete a saved model
#'
#' Removes a model output directory and all its contents.
#'
#' @param path Character; path to the model directory.
#' @param confirm Logical; if TRUE (default), ask for confirmation.
#' @return Invisible TRUE.
#' @export
ds.flower.delete_model <- function(path, confirm = TRUE) {
  if (!dir.exists(path)) {
    stop("Directory not found: ", path, call. = FALSE)
  }

  meta_path <- file.path(path, "metadata.json")
  if (!file.exists(meta_path)) {
    stop("Not a dsFlower model directory (no metadata.json): ", path,
         call. = FALSE)
  }

  if (confirm && interactive()) {
    meta <- tryCatch(
      jsonlite::fromJSON(meta_path, simplifyVector = TRUE),
      error = function(e) list(model_id = basename(path))
    )
    ans <- readline(paste0("Delete model '", meta$model_id, "' at ",
                           path, "? [y/N] "))
    if (!tolower(trimws(ans)) %in% c("y", "yes")) {
      message("Cancelled.")
      return(invisible(FALSE))
    }
  }

  unlink(path, recursive = TRUE)
  message("Deleted: ", path)
  invisible(TRUE)
}
