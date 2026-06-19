# Module: Auto-managed run
# Wraps run.start with automatic SuperLink, ensure, prepare, and cleanup.

# Client-side param bounds (mirrors server .TEMPLATE_PARAM_SCHEMA)
.PARAM_BOUNDS <- list(
  # PyTorch common
  learning_rate = list(min = 1e-8,  max = 1.0),
  batch_size    = list(min = 1L,    max = 100000L),
  local_epochs  = list(min = 1L,    max = 1000L),
  # Architecture
  n_classes     = list(min = 1L,    max = 10000L),
  n_labels      = list(min = 1L,    max = 10000L),
  n_causes      = list(min = 2L,    max = 100L),
  hidden_size   = list(min = 1L,    max = 4096L),
  num_layers    = list(min = 1L,    max = 20L),
  n_channels    = list(min = 1L,    max = 100L),
  kernel_size   = list(min = 1L,    max = 100L),
  n_layers      = list(min = 1L,    max = 50L),
  # sklearn
  alpha         = list(min = 1e-10, max = 100),
  eta0          = list(min = 1e-8,  max = 10),
  C             = list(min = 1e-10, max = 1000),
  max_iter      = list(min = 1L,    max = 100000L),
  l1_ratio      = list(min = 0,     max = 1),
  # XGBoost
  n_trees       = list(min = 1L,    max = 1000L),
  max_depth     = list(min = 1L,    max = 30L),
  n_bins        = list(min = 2L,    max = 1024L),
  eta           = list(min = 1e-6,  max = 1.0),
  reg_lambda    = list(min = 0,     max = 1000),
  local_rounds  = list(min = 1L,    max = 10000L)
)

# Strategy param bounds
.STRATEGY_BOUNDS <- list(
  fraction_fit      = list(min = 0, max = 1),
  fraction_evaluate = list(min = 0, max = 1),
  proximal_mu       = list(min = 0, max = 100),
  eta               = list(min = 1e-8, max = 10),
  tau               = list(min = 1e-10, max = 10)
)

#' Validate model hyperparameters against bounds
#' @keywords internal
.validate_model_params <- function(model) {
  params <- model$params
  if (is.null(params)) return(invisible(TRUE))
  for (nm in names(params)) {
    bounds <- .PARAM_BOUNDS[[nm]]
    if (is.null(bounds)) next
    val <- params[[nm]]
    if (is.null(val)) next
    val <- suppressWarnings(as.numeric(val))
    if (is.na(val)) next
    if (!is.null(bounds$min) && val < bounds$min)
      stop("Parameter '", nm, "' = ", val, " is below minimum (", bounds$min,
           ") for model '", model$name, "'.", call. = FALSE)
    if (!is.null(bounds$max) && val > bounds$max)
      stop("Parameter '", nm, "' = ", val, " exceeds maximum (", bounds$max,
           ") for model '", model$name, "'.", call. = FALSE)
  }
  invisible(TRUE)
}

#' Validate strategy parameters against bounds
#' @keywords internal
.validate_strategy_params <- function(strategy) {
  params <- strategy$params
  if (is.null(params)) return(invisible(TRUE))
  for (nm in names(params)) {
    bounds <- .STRATEGY_BOUNDS[[nm]]
    if (is.null(bounds)) next
    val <- params[[nm]]
    if (is.null(val)) next
    val <- suppressWarnings(as.numeric(val))
    if (is.na(val)) next
    if (!is.null(bounds$min) && val < bounds$min)
      stop("Strategy parameter '", nm, "' = ", val, " is below minimum (",
           bounds$min, ") for strategy '", strategy$name, "'.", call. = FALSE)
    if (!is.null(bounds$max) && val > bounds$max)
      stop("Strategy parameter '", nm, "' = ", val, " exceeds maximum (",
           bounds$max, ") for strategy '", strategy$name, "'.", call. = FALSE)
  }
  invisible(TRUE)
}

#' Run federated learning (auto-managed)
#'
#' Automatically handles SuperLink startup, SuperNode ensure, data preparation,
#' training, and cleanup. The researcher only needs a connection and a recipe.
#'
#' For advanced control (custom ports, persistent SuperLink, etc.), use the
#' low-level functions: \code{ds.flower.superlink.start()},
#' \code{ds.flower.nodes.ensure()}, \code{ds.flower.run.start()}.
#'
#' @param flower A \code{dsflower_connection} from \code{ds.flower.connect()},
#'   or NULL to use the last connection.
#' @param recipe A \code{dsflower_recipe} object.
#' @param detached Logical; use detached SuperLink (default FALSE).
#' @param verbose Logical; print training output (default TRUE).
#' @return A \code{dsflower_run} object.
#' @export
ds.flower.run <- function(flower, recipe, detached = FALSE,
                           verbose = TRUE) {
  if (missing(flower) || is.null(flower))
    stop("'flower' connection handle required. Use: ds.flower.run(flower, recipe)",
         call. = FALSE)
  if (!inherits(flower, "dsflower_connection"))
    stop("'flower' must be a dsflower_connection from ds.flower.connect().",
         call. = FALSE)

  conns <- flower$conns
  symbol <- flower$symbol
  n_clients <- length(conns)

  # Resolve template from model (never require the user to pass it)
  template_name <- recipe$model$template

  # Client-side hyperparameter validation (server schema)
  .validate_model_params(recipe$model)
  .validate_strategy_params(recipe$strategy)

  # DP is always on (local DP, no SecAgg); the privacy spec is just the
  # (epsilon, delta, clipping) budget, read from the node manifest at train time.
  privacy <- .resolve_privacy(recipe$privacy)
  recipe$privacy <- privacy

  # Resolve target and label_set from recipe
  target_column <- recipe$target_column %||% recipe$target
  label_set <- recipe$label_set

  # Resolve features
  feature_columns <- recipe$feature_columns %||% recipe$features
  if (is.null(feature_columns)) feature_columns <- character(0)

  recipe$strategy$params$min_fit_clients <- as.integer(n_clients)
  recipe$strategy$params$min_available_clients <- as.integer(n_clients)
  # Always include every node each round (fixed sampling) -- non-disclosive default.
  recipe$strategy$params$fraction_fit <- 1.0

  # Step 1: Prepare (only if config changed since last prepare)
  current_hash <- digest::digest(list(
    target_column, feature_columns, label_set, template_name,
    recipe$task$type, privacy$epsilon, privacy$delta, recipe$masks),
    algo = "sha256")

  needs_prepare <- is.null(flower$prepare_hash) ||
                   !identical(flower$prepare_hash, current_hash)

  if (needs_prepare) {
    ds.flower.nodes.prepare(conns, symbol,
      target_column   = target_column,
      feature_columns = if (length(feature_columns) > 0) feature_columns else NULL,
      run_config      = list("task-type" = recipe$task$type),
      privacy         = privacy,
      template_name   = template_name,
      label_set       = label_set)
    flower$prepare_hash <- current_hash
    .dsflower_client_env$.connection <- flower
  }

  # Build the harness app before starting remote SuperNodes, so the build is
  # outside the highest-resource phase of the run. The harness is a torch app.
  .ensure_client_framework("pytorch")
  prebuilt_results_dir <- file.path(tempdir(), "dsflower_results",
                                    format(Sys.time(), "%Y%m%d_%H%M%S"))
  dir.create(prebuilt_results_dir, recursive = TRUE, showWarnings = FALSE)
  prebuilt_app_dir <- .build_flower_app(
    recipe,
    conns = conns,
    results_dir = prebuilt_results_dir
  )

  # Step 2: SuperLink. All transport is the DSI tunnel (researcher-initiated, no
  # open ports), so we always run a local SuperLink — there is no remote
  # coordinator / NAT-traversal path.
  started_superlink <- FALSE
  sl_status <- ds.flower.superlink.status()
  if (!isTRUE(sl_status$running)) {
    ds.flower.superlink.start(detached = detached)
    started_superlink <- TRUE
  }

  cleanup_nodes <- function() {
    tryCatch(
      ds.flower.nodes.cleanup(conns, symbol),
      error = function(e) {
        warning("Failed to clean up dsFlower SuperNodes: ",
                conditionMessage(e), call. = FALSE)
      }
    )
  }
  nodes_ensured <- FALSE

  # Step 3: Ensure SuperNodes
  tryCatch(
    {
      ds.flower.nodes.ensure(conns, symbol, template_name = template_name)
      nodes_ensured <- TRUE
    },
    error = function(e) {
      cleanup_nodes()
      if (started_superlink) ds.flower.superlink.stop()
      stop(e)
    }
  )

  # Step 4: Run training
  run <- tryCatch(
    ds.flower.run.start(
      recipe, conns,
      app_dir = prebuilt_app_dir,
      results_dir = prebuilt_results_dir,
      symbol = symbol,
      verbose = verbose
    ),
    error = function(e) {
      if (nodes_ensured) cleanup_nodes()
      if (started_superlink && !detached) ds.flower.superlink.stop()
      stop(e)
    }
  )

  # Step 5: Stop remote SuperNodes, then stop SuperLink if we started it.
  # SuperNodes are per-run processes; leaving them alive after SuperLink stops
  # causes reconnection loops and consumes CPU/RAM in Rock containers.
  if (nodes_ensured) cleanup_nodes()

  if (started_superlink && !detached) {
    ds.flower.superlink.stop()
  }

  run
}

#' One-shot federated learning (simplest path)
#'
#' Connects, prepares, trains, and cleans up in a single call.
#' For the simplest possible researcher experience.
#'
#' @param conns DSI connections object.
#' @param data Character; data source (resource name or symbol).
#' @param recipe A \code{dsflower_recipe} object.
#' @param ... Additional arguments passed to \code{ds.flower.run()}.
#' @return A \code{dsflower_run} object.
#' @export
ds.flower.train <- function(conns, data, recipe, ...) {
  flower <- ds.flower.connect(conns, data)
  on.exit(try(ds.flower.disconnect(flower), silent = TRUE), add = TRUE)
  ds.flower.run(flower, recipe, ...)
}
