# Module: flwr CLI Integration
# Controls Flower runs via the flwr CLI.

#' Start a Flower run
#'
#' Invokes \code{flwr run} for a pre-built Flower App against the running
#' SuperLink. Model weights and training history are automatically saved to
#' \code{output_dir} after training completes.
#'
#' @param recipe A \code{dsflower_recipe} object.
#' @param conns DSI connections object used to determine the required site count.
#'   If NULL, uses the connections stored during
#'   \code{ds.flower.nodes.init}.
#' @param app_dir Required character path to a pre-built app directory. The
#'   high-level \code{ds.flower.submit()} and \code{ds.flower.fit()} pipelines
#'   build and supply it automatically.
#' @param run_config Named list; additional run config overrides.
#' @param output_dir Character; persistent directory for model output.
#'   Defaults to \code{"dsflower_output/<timestamp>"} in the working directory.
#' @param output_name Optional name for the persisted model artifact.
#' @param results_dir Character; temporary directory where the Flower ServerApp
#'   writes model artefacts. Usually generated automatically.
#' @param symbol Character; server-side Flower handle symbol. Low-level callers
#'   that initialise the default handle can leave this as \code{"flower"}.
#' @param verbose Logical; print flwr output (default FALSE).
#' @param silent Logical; suppress progress feedback.
#' @return A \code{dsflower_run} object with run status and identifiers, model
#'   metadata, weights, history, output paths, and captured CLI output.
#' @export
ds.flower.run.start <- function(recipe, conns = NULL, app_dir = NULL,
                                 run_config = list(), output_dir = NULL,
                                 output_name = NULL,
                                 results_dir = NULL,
                                 symbol = "flower",
                                 verbose = FALSE, silent = FALSE) {
  if (!inherits(recipe, "dsflower_recipe")) {
    stop("'recipe' must be a dsflower_recipe object.", call. = FALSE)
  }

  .require_flwr_cli()

  old_opt <- options(dsflower.silent = isTRUE(silent))
  on.exit(options(old_opt), add = TRUE)

  # Check SuperLink is running
  sl_info <- .dsflower_client_env$.superlink
  if (is.null(sl_info) || is.null(sl_info$process) || !sl_info$process$is_alive()) {
    stop("No SuperLink is running. Call ds.flower.superlink.start() first.",
         call. = FALSE)
  }

  # Results directory for model weights and metrics.
  if (is.null(results_dir)) {
    results_root <- file.path(tempdir(), "dsflower_results")
    dir.create(results_root, recursive = TRUE, showWarnings = FALSE)
    results_dir <- tempfile(
      pattern = paste0(format(Sys.time(), "%Y%m%d_%H%M%S"), "_"),
      tmpdir = results_root)
  } else if (dir.exists(results_dir) &&
             length(list.files(results_dir, all.files = TRUE, no.. = TRUE))) {
    stop("'results_dir' must be empty; stale Flower artifacts are unsafe.",
         call. = FALSE)
  }
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  # Resolve the connections used for this federated run.
  if (is.null(conns)) {
    conns <- .dsflower_client_env$.conns
  }
  if (is.null(conns)) {
    stop("'conns' is required to run the federated app.", call. = FALSE)
  }

  # Set min_fit_clients and min_available_clients from number of connections
  n_clients <- length(conns)
  recipe$strategy$params$min_fit_clients <- as.integer(n_clients)
  recipe$strategy$params$min_available_clients <- as.integer(n_clients)

  # DP (epsilon/delta/clipping + mechanism) is decided + enforced by each node; the
  # client carries no privacy spec. Always include every node each round.
  recipe$strategy$params$fraction_fit <- 1.0

  # Neural ServerApps build the initial torch model. The native-tree app has a
  # separate dependency-light entrypoint and must not install or import torch.
  native_tree <- identical(recipe$model$track %||% NULL, "native_tree")
  cv_job <- !is.null(recipe$cross_validation_contract)
  if (cv_job && native_tree) {
    stop("Cross-validation requires a neural recipe.", call. = FALSE)
  }
  if (cv_job &&
      (!is.character(recipe$cross_validation_job_sha256) ||
       length(recipe$cross_validation_job_sha256) != 1L ||
       is.na(recipe$cross_validation_job_sha256) ||
       !grepl("^[0-9a-f]{64}$", recipe$cross_validation_job_sha256) ||
       !identical(as.integer(recipe$cross_validation_n_nodes),
                  as.integer(n_clients)))) {
    stop("Cross-validation recipe has no exact public job provenance pin.",
         call. = FALSE)
  }
  if (!native_tree) .ensure_client_framework("pytorch")

  # The app dir is always pre-built by the submission pipeline (a dsflower_runner
  # FAB, content-hash-pinned against the node-resident canonical runner).
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
  eff_rounds <- recipe$num_rounds
  if (cv_job) {
    .dsf_msg("Cross-validating ", recipe$model$name, " across ", n_clients,
             " site(s), ", recipe$cross_validation_contract$folds,
             " fold(s) and ", eff_rounds, " round(s) per fold...")
  } else {
    .dsf_msg("Training ", recipe$model$name, " across ", n_clients,
             " site(s) for ", eff_rounds, " round(s)...")
  }
  flwr_cmd <- .client_flwr_cmd()
  # PYTHONUNBUFFERED so the flwr child flushes its [ROUND k/N] log lines as they
  # happen (block-buffered pipes otherwise hold them until exit, which hides the
  # per-round progress under a non-interactive front-end such as knitr).
  result <- .run_flwr_with_artifact_watchdog(
    command = flwr_cmd,
    args = args,
    env = .client_venv_env(extra = c(FLWR_HOME = sl_info$flwr_home,
                                      PYTHONUNBUFFERED = "1")),
    results_dir = results_dir,
    num_rounds = eff_rounds,
    expect_artifacts = TRUE,
    cross_validation = cv_job
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
  holdout_release <- .read_holdout_result(results_dir)
  cv_release <- .read_cross_validation_result(results_dir)
  if (cv_job) {
    unexpected <- setdiff(
      list.files(results_dir, all.files = TRUE, no.. = TRUE), "cv.json")
    if (result$status != 0L || is.null(cv_release) || !is.null(weights) ||
        !is.null(history) || !is.null(holdout_release) || length(unexpected)) {
      stop("Federated cross-validation failed or produced a forbidden fold ",
           "artifact; no result was accepted.", call. = FALSE)
    }
    if (!identical(
        cv_release$cv_contract_sha256,
        recipe$cross_validation_contract$sha256) ||
        !identical(
          as.integer(cv_release$folds),
          as.integer(recipe$cross_validation_contract$folds)) ||
        !identical(as.integer(cv_release$n_nodes), as.integer(n_clients)) ||
        !identical(cv_release$cv_job_sha256,
                   recipe$cross_validation_job_sha256)) {
      stop("Federated cross-validation output does not match its submitted ",
           "public contract.", call. = FALSE)
    }
    saved_path <- NULL
    persisted_dir <- NULL
    if (!is.null(output_dir) || !is.null(output_name)) {
      parent <- output_dir %||% file.path(".", "dsflower_output")
      save_name <- output_name %||% paste0(
        recipe$model$name, "_cv_", format(Sys.time(), "%Y%m%d_%H%M%S"))
      persisted_dir <- file.path(parent, save_name)
      if (dir.exists(persisted_dir) && length(list.files(
          persisted_dir, all.files = TRUE, no.. = TRUE))) {
        stop("Cross-validation output directory must be empty.", call. = FALSE)
      }
      dir.create(persisted_dir, recursive = TRUE, showWarnings = FALSE)
      persisted_dir <- normalizePath(
        persisted_dir, winslash = "/", mustWork = FALSE)
      saved_path <- file.path(persisted_dir, "cv.json")
      temporary <- tempfile(pattern = ".cv-", tmpdir = persisted_dir)
      on.exit(unlink(temporary, force = TRUE), add = TRUE)
      copied <- file.copy(
        file.path(results_dir, "cv.json"), temporary, overwrite = FALSE)
      if (!isTRUE(copied) || !file.rename(temporary, saved_path)) {
        stop("Could not persist the pooled cross-validation result.",
             call. = FALSE)
      }
    }
    .dsf_msg("Cross-validation complete: one pooled DP OOF metric release; ",
             "no fold models, predictions, or site metrics were saved.")
    return(structure(list(
      run_id = run_id,
      status = 0L,
      cli_status = as.integer(result$status),
      model = recipe$model$name,
      task = cv_release$task,
      folds = as.integer(cv_release$folds),
      rounds = as.integer(eff_rounds),
      n_nodes = as.integer(cv_release$n_nodes),
      metrics = cv_release$metrics,
      contract = recipe$cross_validation_contract,
      job_sha256 = cv_release$cv_job_sha256,
      results_dir = results_dir,
      output_dir = persisted_dir,
      saved_path = saved_path,
      stdout = clean_stdout,
      stderr = clean_stderr
    ), class = "dsflower_cv"))
  }
  if (!is.null(cv_release)) {
    stop("Training produced an unexpected cross-validation transcript.",
         call. = FALSE)
  }
  expects_holdout <- !is.null(recipe$holdout_contract)
  if (!expects_holdout && !is.null(holdout_release)) {
    stop("Training produced an unexpected holdout transcript.", call. = FALSE)
  }
  native_history_available <- !is.data.frame(history) || !nrow(history) ||
    !("available" %in% names(history)) ||
    any(as.logical(history$available), na.rm = TRUE)
  native_release <- NULL
  if (native_tree && identical(as.integer(result$status), 0L) &&
      native_history_available) {
    native_release <- .native_tree_release_metadata(recipe, results_dir)
  }
  runtime_status <- .flower_runtime_status(
    cli_status = result$status,
    stdout = clean_stdout,
    stderr = clean_stderr,
    weights = weights,
    history = history,
    expect_artifacts = TRUE
  )

  if (runtime_status != 0L) {
    stop("Federated training failed (status ", runtime_status,
         "); no model was accepted or saved.", call. = FALSE)
  }
  if (expects_holdout && is.null(holdout_release)) {
    stop("Atomic holdout produced no pooled test metric release; no model was ",
         "accepted or saved.", call. = FALSE)
  }
  if (expects_holdout && !.atomic_holdout_commit_complete(
      history, eff_rounds, results_dir)) {
    stop("Atomic holdout has no complete commit marker; no model was accepted ",
         "or saved.", call. = FALSE)
  }

  available_rounds <- if (is.data.frame(history) && nrow(history)) {
    if ("available" %in% names(history)) {
      keep <- !is.na(history$available) & as.logical(history$available)
      if ("round" %in% names(history)) {
        as.integer(history$round[keep])
      } else {
        which(keep)
      }
    } else {
      seq_len(nrow(history))
    }
  } else {
    integer()
  }
  available <- length(available_rounds) > 0L ||
    (!is.null(weights) && is.null(history))
  if (!is.null(weights) || !is.null(history)) {
    n_done <- if (is.data.frame(history) && nrow(history)) nrow(history) else eff_rounds
    if (available) {
      n_available <- if (length(available_rounds)) length(available_rounds) else n_done
      .dsf_msg("Training complete: ", n_available,
               " private round(s) available across ", n_clients,
               " site(s) with strategy '", recipe$strategy$name,
               "'. Differential privacy was enforced by the nodes.")
    } else {
      .dsf_msg("Training request completed, but no private model release is ",
               "available under the node privacy policy. The request was not ",
               "blocked and no public fallback was saved as a trained model.")
    }
  }

  # Generate a unique model ID for identification
  model_id <- paste0(
    recipe$model$name, "_",
    recipe$strategy$name, "_",
    eff_rounds, "r_",
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
      framework  = recipe$model$framework,
      track      = recipe$model$track %||% NULL,
      model_spec = recipe$model_spec %||% NULL,
      model_params = recipe$model_params %||% list(),
      loss_name  = recipe$loss_name %||% NULL,
      data_kind  = recipe$data_kind %||% "tabular",
      available  = available,
      available_rounds = as.integer(available_rounds),
      strategy   = recipe$strategy$name,
      privacy    = "server-enforced-dp",
      num_rounds = eff_rounds,
      requested_num_rounds = recipe$num_rounds,
      target_levels = recipe$target_levels %||% NULL,
      target_bounds = recipe$target_bounds %||% NULL,
      native_tree_request_b64 = recipe$native_tree_request_b64 %||% NULL,
      native_tree_request_sha256 =
        recipe$native_tree_request_sha256 %||% NULL,
      public_schema_sha256 = recipe$public_schema_sha256 %||% NULL,
      holdout = holdout_release$metrics %||% NULL,
      holdout_task = holdout_release$task %||% NULL,
      holdout_contract = recipe$holdout_contract %||% NULL,
      artifact = native_release$artifact %||% NULL,
      sanitization = native_release$sanitization %||% NULL,
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
    meta <- if (native_tree && available) {
      list(
        track = "native_tree",
        engine = recipe$model$engine,
        task = recipe$model$task,
        data_kind = recipe$data_kind %||% "tabular",
        features = recipe$features,
        feature_lower = recipe$feature_lower,
        feature_upper = recipe$feature_upper,
        target_levels = recipe$target_levels %||% NULL,
        target_bounds = recipe$target_bounds %||% NULL,
        native_tree_request_b64 = recipe$native_tree_request_b64,
        native_tree_request_sha256 = recipe$native_tree_request_sha256,
        public_schema_sha256 = recipe$public_schema_sha256,
        artifact = native_release$artifact,
        sanitization = native_release$sanitization)
    } else {
      list(
        model_id   = model_id,
        model      = recipe$model$name,
        framework  = recipe$model$framework,
        track      = recipe$model$track %||% NULL,
        model_spec = recipe$model_spec %||% NULL,
        model_params = recipe$model_params %||% list(),
        loss_name  = recipe$loss_name %||% NULL,
        data_kind  = recipe$data_kind %||% "tabular",
        available  = available,
        available_rounds = as.integer(available_rounds),
        strategy   = recipe$strategy$name,
        privacy    = "server-enforced-dp",
        num_rounds = eff_rounds,
        requested_num_rounds = recipe$num_rounds,
        n_clients  = length(conns),
        features   = recipe$features,
        feature_lower = recipe$feature_lower,
        feature_upper = recipe$feature_upper,
        target_levels = recipe$target_levels %||% NULL,
        target_bounds = recipe$target_bounds %||% NULL,
        created_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
        holdout = holdout_release$metrics %||% NULL,
        holdout_contract = recipe$holdout_contract %||% NULL,
        status     = if (!available) "unavailable" else if (runtime_status == 0L) {
          "success"
        } else {
          "failed"
        })
    }
    jsonlite::write_json(meta, file.path(output_dir, "metadata.json"),
                         auto_unbox = TRUE, pretty = TRUE,
                         null = if (native_tree && available) "null" else "list")

    .dsf_msg("Model saved to ", output_dir)
  }

  structure(
    list(
      model_id    = model_id,
      run_id      = run_id,
      status      = runtime_status,
      cli_status  = result$status,
      num_rounds  = eff_rounds,
      requested_num_rounds = recipe$num_rounds,
      model       = recipe$model$name,
      data_kind   = recipe$data_kind %||% "tabular",
      available   = available,
      available_rounds = as.integer(available_rounds),
      strategy    = recipe$strategy$name,
      weights     = weights,
      history     = history,
      output_dir  = output_dir,
      saved_path  = saved_path,
      results_dir = results_dir,
      app_dir     = app_dir,
      stdout      = clean_stdout,
      stderr      = clean_stderr,
      holdout     = holdout_release$metrics %||% NULL,
      holdout_task = holdout_release$task %||% NULL,
      holdout_contract = recipe$holdout_contract %||% NULL
    ),
    class = "dsflower_run"
  )
}

.read_holdout_result <- function(results_dir) {
  path <- file.path(results_dir, "holdout.json")
  if (!file.exists(path)) return(NULL)
  info <- file.info(path)
  if (is.na(info$size) || info$size < 1 || info$size > 8 * 1024^2) {
    stop("Holdout output failed the pooled-only privacy contract.",
         call. = FALSE)
  }
  value <- tryCatch(
    jsonlite::fromJSON(path, simplifyVector = FALSE),
    error = function(e) NULL)
  allowed <- c("pooled_only", "privacy", "method", "task", "n_nodes",
               "metrics")
  required_metric <- if (is.list(value) && length(value$task) == 1L) {
    switch(as.character(value$task),
           binary = "accuracy", multiclass = "accuracy",
           ordinal = "accuracy", multilabel = "macro_f1",
           regression = "mae", count = "mae", NA_character_)
  } else {
    NA_character_
  }
  primary_task <- if (is.list(value) && length(value$task) == 1L) {
    as.character(value$task)
  } else {
    ""
  }
  primary_metric <- if (is.list(value) && is.list(value$metrics) &&
      !is.na(required_metric) && required_metric %in% names(value$metrics)) {
    value$metrics[[required_metric]]
  } else {
    NULL
  }
  plausible_primary <- is.numeric(primary_metric) &&
    length(primary_metric) == 1L && is.finite(primary_metric) &&
    primary_metric >= 0 &&
    (primary_task %in% c("regression", "count") || primary_metric <= 1)
  if (!is.list(value) || is.null(names(value)) || anyDuplicated(names(value)) ||
      !setequal(names(value), allowed) ||
      !identical(value$pooled_only, TRUE) ||
      !identical(value$privacy, "node-dp-pooled-postprocessing") ||
      !identical(value$method, "holdout") ||
      length(value$task) != 1L || !value$task %in% c(
        "binary", "multiclass", "ordinal", "multilabel",
        "regression", "count") ||
      length(value$n_nodes) != 1L || !is.numeric(value$n_nodes) ||
      !is.finite(value$n_nodes) || value$n_nodes < 1 ||
      value$n_nodes != floor(value$n_nodes) || !is.list(value$metrics) ||
      is.null(names(value$metrics)) || anyDuplicated(names(value$metrics)) ||
      is.na(required_metric) || !required_metric %in% names(value$metrics) ||
      !plausible_primary ||
      any(c("per_node", "predictions", "folds") %in% names(value$metrics))) {
    stop("Holdout output failed the pooled-only privacy contract.",
         call. = FALSE)
  }
  value
}

.read_cross_validation_result <- function(results_dir) {
  path <- file.path(results_dir, "cv.json")
  if (!file.exists(path)) return(NULL)
  info <- file.info(path)
  if (is.na(info$size) || info$size < 1 || info$size > 8 * 1024^2) {
    stop("Cross-validation output failed the pooled-only privacy contract.",
         call. = FALSE)
  }
  value <- tryCatch(
    jsonlite::fromJSON(path, simplifyVector = FALSE),
    error = function(e) NULL)
  allowed <- c("pooled_only", "privacy", "method", "task", "n_nodes",
               "folds", "cv_contract_sha256", "cv_job_sha256", "metrics")
  required_metric <- if (is.list(value) && length(value$task) == 1L) {
    switch(as.character(value$task),
           binary = "accuracy", multiclass = "accuracy",
           ordinal = "accuracy", multilabel = "macro_f1",
           regression = "mae", count = "mae", NA_character_)
  } else {
    NA_character_
  }
  primary_task <- if (is.list(value) && length(value$task) == 1L) {
    as.character(value$task)
  } else {
    ""
  }
  primary_metric <- if (is.list(value) && is.list(value$metrics) &&
      !is.na(required_metric) && required_metric %in% names(value$metrics)) {
    value$metrics[[required_metric]]
  } else {
    NULL
  }
  plausible_primary <- is.numeric(primary_metric) &&
    length(primary_metric) == 1L && is.finite(primary_metric) &&
    primary_metric >= 0 &&
    (primary_task %in% c("regression", "count") || primary_metric <= 1)
  safe_metric_tree <- function(item) {
    if (is.null(item)) return(TRUE)
    if (is.numeric(item)) return(all(is.finite(item)))
    if (!is.list(item)) return(FALSE)
    keys <- names(item) %||% character()
    if (any(keys %in% c(
        "per_node", "per_site", "predictions", "models", "fold",
        "folds", "per_fold", "fold_metrics"))) return(FALSE)
    all(vapply(item, safe_metric_tree, logical(1)))
  }
  if (!is.list(value) || is.null(names(value)) || anyDuplicated(names(value)) ||
      !setequal(names(value), allowed) ||
      !identical(value$pooled_only, TRUE) ||
      !identical(value$privacy, "node-dp-pooled-postprocessing") ||
      !identical(value$method, "cross_validation") ||
      length(value$task) != 1L || !value$task %in% c(
        "binary", "multiclass", "ordinal", "multilabel",
        "regression", "count") ||
      length(value$n_nodes) != 1L || !is.numeric(value$n_nodes) ||
      !is.finite(value$n_nodes) || value$n_nodes < 1 ||
      value$n_nodes != floor(value$n_nodes) ||
      length(value$folds) != 1L || !is.numeric(value$folds) ||
      !is.finite(value$folds) || value$folds < 2 || value$folds > 10 ||
      value$folds != floor(value$folds) ||
      length(value$cv_contract_sha256) != 1L ||
      !is.character(value$cv_contract_sha256) ||
      is.na(value$cv_contract_sha256) ||
      !grepl("^[0-9a-f]{64}$", value$cv_contract_sha256) ||
      length(value$cv_job_sha256) != 1L ||
      !is.character(value$cv_job_sha256) ||
      is.na(value$cv_job_sha256) ||
      !grepl("^[0-9a-f]{64}$", value$cv_job_sha256) ||
      !is.list(value$metrics) ||
      is.null(names(value$metrics)) || anyDuplicated(names(value$metrics)) ||
      is.na(required_metric) || !required_metric %in% names(value$metrics) ||
      !plausible_primary ||
      !safe_metric_tree(value$metrics)) {
    stop("Cross-validation output failed the pooled-only privacy contract.",
         call. = FALSE)
  }
  value
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
  native_files <- vapply(
    .NATIVE_TREE_RELEASE_SPECS, `[[`, character(1), "artifact_file")
  any(file.exists(file.path(
    results_dir,
    c("global_model.json", "global_model.skipped.json",
      "model.pt", "model.npz", native_files,
      "validation.json")
  )))
}

.atomic_neural_model_exists <- function(results_dir) {
  path <- file.path(results_dir, "model.pt")
  info <- file.info(path)
  file.exists(path) && !is.na(info$isdir) && !isTRUE(info$isdir) &&
    !is.na(info$size) && info$size > 0
}

.atomic_holdout_commit_complete <- function(history, num_rounds, results_dir) {
  expected <- seq_len(as.integer(num_rounds))
  if (!is.data.frame(history) || nrow(history) != length(expected) ||
      !all(c("round", "available") %in% names(history))) {
    return(FALSE)
  }
  rounds <- suppressWarnings(as.integer(history$round))
  available <- suppressWarnings(as.logical(history$available))
  identical(rounds, expected) &&
    all(!is.na(available) & available) &&
    .atomic_neural_model_exists(results_dir)
}

.native_tree_release_metadata <- function(recipe, results_dir) {
  request <- .validate_native_tree_request_wire(
    recipe$native_tree_request_b64,
    recipe$native_tree_request_sha256)
  engine <- recipe$model$engine %||% NULL
  task <- recipe$model$task %||% NULL
  if (!is.character(engine) || length(engine) != 1L || is.na(engine) ||
      !engine %in% .NATIVE_TREE_ENGINES ||
      !identical(request$value$engine, engine) ||
      !identical(request$value$task, recipe$model$task) ||
      !identical(request$value$public_schema$sha256,
                 recipe$public_schema_sha256)) {
    stop("Native-tree recipe no longer matches its canonical request.",
         call. = FALSE)
  }
  release_spec <- .native_tree_release_spec(engine)
  path <- file.path(results_dir, release_spec$artifact_file)
  info <- file.info(path)
  if (!file.exists(path) || is.na(info$isdir) || isTRUE(info$isdir) ||
      is.na(info$size) || info$size < 1 ||
      info$size > .NATIVE_TREE_ENSEMBLE_MAX_BYTES) {
    stop("Native-tree training produced no bounded sanitized ensemble artifact.",
         call. = FALSE)
  }
  bytes <- readBin(path, what = "raw", n = as.integer(info$size))
  artifact <- list(
    file = release_spec$artifact_file,
    format = release_spec$artifact_format,
    size_bytes = as.integer(length(bytes)),
    sha256 = digest::digest(bytes, algo = "sha256", serialize = FALSE))
  sanitization <- .native_tree_sanitization_attestation(engine)
  candidate <- list(
    public_schema_sha256 = recipe$public_schema_sha256,
    artifact = artifact,
    sanitization = sanitization)
  .validate_native_tree_ensemble_artifact(
    candidate, results_dir, task, engine)
  .validate_native_tree_prediction_profile(
    results_dir,
    recipe$native_tree_request_b64,
    recipe$native_tree_request_sha256,
    recipe$public_schema_sha256,
    task, engine,
    artifact)
  list(artifact = artifact,
       sanitization = sanitization)
}

.training_artifacts_complete <- function(results_dir, num_rounds,
                                         expect_artifacts = TRUE,
                                         cross_validation = FALSE) {
  if (isTRUE(cross_validation)) {
    path <- file.path(results_dir, "cv.json")
    info <- file.info(path)
    return(file.exists(path) && !is.na(info$isdir) && !isTRUE(info$isdir) &&
             !is.na(info$size) && info$size > 0 && info$size <= 8 * 1024^2)
  }
  history <- .read_training_history(results_dir)
  if (is.null(history) || !NROW(history)) return(FALSE)

  if ("round" %in% names(history)) {
    rounds <- suppressWarnings(as.integer(history$round))
    if (!any(rounds >= as.integer(num_rounds), na.rm = TRUE)) return(FALSE)
  }

  if (!isTRUE(expect_artifacts)) return(TRUE)
  if ("available" %in% names(history) &&
      !any(as.logical(history$available), na.rm = TRUE)) return(TRUE)
  .model_artifact_exists(results_dir)
}

# Emit a status/progress line to the user unless silence was requested. Progress
# is opt-out (ds.flower.fit(..., silent = TRUE)); the option is set by the entry
# points and read here so deep helpers need no extra argument threading.
.dsf_msg <- function(...) {
  if (!isTRUE(getOption("dsflower.silent", FALSE))) message(...)
}

# Parse freshly streamed flwr lines and announce each round as it starts. The
# ServerApp logs "[ROUND k/N]" per federated round. `seen` tracks already-
# announced rounds across poll iterations.
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
                                             expect_artifacts = TRUE,
                                             cross_validation = FALSE) {
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
        .training_artifacts_complete(
          results_dir, num_rounds, expect_artifacts,
          cross_validation = cross_validation)) {
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
      "An exception was raised when attempting to load .*ClientApp",
      paste0("Received[[:space:]]+[0-9]+[[:space:]]+results and[[:space:]]+",
             "[1-9][0-9]*[[:space:]]+failures"),
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
    message("Skipping in-memory weight load for large ", basename(path), " (",
            round(size / 1024^2, 1), " MiB). Native artifacts remain in ",
            results_dir, ".")
    return(NULL)
  }

  raw <- jsonlite::fromJSON(path, simplifyVector = FALSE)
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

  names(params) <- if (length(params) == 2L) {
    c("coef", "intercept")
  } else {
    paste0("parameter_", seq_along(params) - 1L)
  }
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
  cat("  Release:  ", if (isFALSE(x$available)) "unavailable" else "available", "\n")
  if (!is.null(x$holdout)) {
    cat("  Holdout:  pooled DP metrics available in $holdout\n")
  }

  # Per-round summary. Loss is intentionally not released by the nodes (a
  # disclosure backstop: only a bucketed example count leaves), so we report
  # what the history actually carries. Measure utility with ds.flower.predict.
  if (!is.null(x$history) && is.data.frame(x$history) && nrow(x$history)) {
    cat("\n  Rounds:\n")
    has_ex <- "n_examples" %in% names(x$history)
    for (i in seq_len(nrow(x$history))) {
      ex <- if (has_ex) x$history$n_examples[i] else NA
      released <- if ("available" %in% names(x$history)) {
        isTRUE(x$history$available[i])
      } else TRUE
      cat(sprintf("    round %d %s%s\n", x$history$round[i],
        if (released) "aggregated" else "release unavailable",
        if (released && !is.na(ex)) sprintf(" (~%s examples)", ex) else ""))
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
#' Saves the federated model metadata to a file, together with a
#' sibling \code{<filename>.assets} directory containing the native model and
#' its public reconstruction metadata. Move both entries together. Supported
#' formats are \code{.rds} (R native) and \code{.json}.
#'
#' @param run A \code{dsflower_run} object.
#' @param path Character; unused \code{.rds} or \code{.json} file path.
#' @return Invisible path.
#' @export
ds.flower.save_model <- function(run, path) {
  if (!inherits(run, "dsflower_run")) {
    stop("'run' must be a dsflower_run object.", call. = FALSE)
  }
  if (isFALSE(run$available)) {
    stop("No private model release is available in this run.", call. = FALSE)
  }
  if (!is.character(path) || length(path) != 1L || is.na(path) || !nzchar(path)) {
    stop("'path' must be one non-empty .rds or .json file path.", call. = FALSE)
  }

  ext <- tolower(tools::file_ext(path))
  if (!ext %in% c("rds", "json")) {
    stop("Unsupported format '.", ext, "'. Use .rds or .json.", call. = FALSE)
  }
  parent <- dirname(path)
  if (!dir.exists(parent) && !dir.create(parent, recursive = TRUE)) {
    stop("Cannot create model destination directory: ", parent, call. = FALSE)
  }
  parent <- normalizePath(parent, winslash = "/", mustWork = TRUE)
  path <- file.path(parent, basename(path))
  asset_name <- paste0(basename(path), ".assets")
  asset_path <- file.path(parent, asset_name)
  if (file.exists(path) || file.exists(asset_path) || dir.exists(asset_path)) {
    stop("Model destination already exists: ", path,
         " (or its .assets directory).", call. = FALSE)
  }

  source_dir <- run$output_dir %||% NULL
  if (!is.character(source_dir) || length(source_dir) != 1L ||
      is.na(source_dir) || !dir.exists(source_dir)) {
    stop("The run's persisted model directory is unavailable; cannot create ",
         "a portable model bundle.", call. = FALSE)
  }
  source_dir <- normalizePath(source_dir, winslash = "/", mustWork = TRUE)
  native_asset_files <- unique(unlist(lapply(
    .NATIVE_TREE_RELEASE_SPECS,
    function(spec) c(spec$artifact_file, spec$profile_file)),
    use.names = FALSE))
  native_model_files <- vapply(
    .NATIVE_TREE_RELEASE_SPECS, `[[`, character(1), "artifact_file")
  asset_files <- c(
    "metadata.json", "history.json", "model.pt", "model.npz",
    "global_model.json", "global_model.skipped.json",
    native_asset_files
  )
  present <- asset_files[file.exists(file.path(source_dir, asset_files))]
  model_files <- c(
    "model.pt", "model.npz", "global_model.json",
    native_model_files
  )
  if (!"metadata.json" %in% present || !any(model_files %in% present)) {
    stop("The run directory does not contain a complete public model bundle.",
         call. = FALSE)
  }
  if (any(native_model_files %in% present)) {
    .resolve_validation_contract(source_dir, 32L)
  }

  model_data <- list(
    model_id = run$model_id,
    weights  = run$weights,
    history  = run$history,
    model    = run$model,
    strategy = run$strategy,
    rounds   = run$num_rounds,
    run_id   = run$run_id,
    saved_at = Sys.time(),
    artifact_bundle = asset_name
  )

  staged_assets <- tempfile(".dsflower-assets-", tmpdir = parent)
  staged_model <- tempfile(
    ".dsflower-model-", tmpdir = parent, fileext = paste0(".", ext))
  dir.create(staged_assets)
  installed_assets <- FALSE
  on.exit({
    if (file.exists(staged_model)) unlink(staged_model)
    if (dir.exists(staged_assets)) unlink(staged_assets, recursive = TRUE)
    if (installed_assets && !file.exists(path) && dir.exists(asset_path)) {
      unlink(asset_path, recursive = TRUE)
    }
  }, add = TRUE)
  copied <- file.copy(
    file.path(source_dir, present), staged_assets, overwrite = FALSE)
  if (!all(copied)) {
    stop("Failed to copy the native model bundle.", call. = FALSE)
  }

  if (ext == "rds") {
    saveRDS(model_data, staged_model)
  } else {
    jsonlite::write_json(
      model_data, staged_model, auto_unbox = TRUE, digits = 10)
  }
  if (!file.rename(staged_assets, asset_path)) {
    stop("Failed to install the native model bundle.", call. = FALSE)
  }
  installed_assets <- TRUE
  if (!file.rename(staged_model, path)) {
    stop("Failed to install the saved model file.", call. = FALSE)
  }

  message("Model saved to ", path, " with assets in ", asset_path)
  invisible(path)
}

#' List saved models
#'
#' Scans the output directory for previously saved models and returns
#' a summary data.frame with metadata for each.
#'
#' @param base_dir Character; base directory to scan.
#'   Defaults to \code{"./dsflower_output"}.
#' @return A data.frame with columns: model_id, model, strategy,
#'   privacy, num_rounds, n_clients, created_at, status, path.
#' @export
ds.flower.models <- function(base_dir = file.path(".", "dsflower_output")) {
  if (!dir.exists(base_dir)) {
    return(data.frame(
      model_id = character(0), model = character(0),
      strategy = character(0),
      privacy = character(0), num_rounds = integer(0),
      n_clients = integer(0), created_at = character(0),
      status = character(0), path = character(0),
      stringsAsFactors = FALSE
    ))
  }

  subdirs <- list.dirs(base_dir, recursive = FALSE, full.names = TRUE)
  rows <- list()
  # Coerce any metadata field to a single scalar: %||% only replaces NULL, so an empty
  # (length-0) field from a partial metadata.json would make data.frame() error with
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
      strategy = character(0),
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
#' Reads model weights and metadata from a previously saved output directory,
#' \code{.rds} file, or \code{.json} file. Files created by
#' \code{ds.flower.save_model()} must remain beside their sibling
#' \code{<filename>.assets} directory.
#'
#' @param path Character; path to a model directory, \code{.rds}, or
#'   \code{.json} file.
#' @return A list with model_id, weights, history, and metadata.
#' @export
ds.flower.load_model <- function(path) {
  if (!is.character(path) || length(path) != 1L || is.na(path) || !nzchar(path)) {
    stop("'path' must be one non-empty file or directory path.", call. = FALSE)
  }
  attach_source <- function(value, source_dir) {
    if (!is.list(value)) {
      stop("Saved dsFlower model must contain a list.", call. = FALSE)
    }
    bundle <- value[["artifact_bundle"]]
    if (!is.null(bundle)) {
      if (!is.character(bundle) || length(bundle) != 1L || is.na(bundle) ||
          !nzchar(bundle) || grepl("[\\\\/]", bundle) ||
          bundle %in% c(".", "..")) {
        stop("Saved dsFlower model has an invalid artifact bundle name.",
             call. = FALSE)
      }
      source_dir <- file.path(source_dir, bundle)
      if (!dir.exists(source_dir)) {
        stop("Saved model artifact bundle is missing: ", source_dir,
             call. = FALSE)
      }
    }
    value$source_dir <- normalizePath(
      source_dir, winslash = "/", mustWork = TRUE)
    value
  }
  if (dir.exists(path)) {
    rds_path <- file.path(path, "model.rds")
    if (!file.exists(rds_path)) {
      # Runs name the bundle after output_name; accept that single .rds file.
      cand <- list.files(path, pattern = "\\.rds$", full.names = TRUE)
      if (length(cand) != 1L)
        stop("No model .rds found in ", path, call. = FALSE)
      rds_path <- cand
    }
    return(attach_source(readRDS(rds_path), path))
  }

  if (file.exists(path)) {
    ext <- tolower(tools::file_ext(path))
    if (ext == "rds") return(attach_source(readRDS(path), dirname(path)))
    if (ext == "json") {
      return(attach_source(
        jsonlite::fromJSON(path, simplifyVector = FALSE), dirname(path)))
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
