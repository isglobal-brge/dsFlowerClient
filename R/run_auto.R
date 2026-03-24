# Module: Auto-managed run
# Wraps run.start with automatic SuperLink, ensure, prepare, and cleanup.

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
ds.flower.run <- function(flower = NULL, recipe, detached = FALSE,
                           verbose = TRUE) {
  if (is.null(flower)) {
    flower <- .dsflower_client_env$.connection
  }
  if (is.null(flower) || !inherits(flower, "dsflower_connection")) {
    stop("No connection. Call ds.flower.connect() first.", call. = FALSE)
  }

  conns <- flower$conns
  symbol <- flower$symbol

  # Resolve template from model (never require the user to pass it)
  template_name <- recipe$model$template

  # Resolve privacy (recipe wins, else clinical_default)
  privacy <- recipe$privacy
  if (is.null(privacy)) privacy <- ds.flower.privacy.clinical_default()

  # Resolve target and label_set from recipe
  target_column <- recipe$target_column %||% recipe$target
  label_set <- recipe$label_set

  # Resolve features
  feature_columns <- recipe$feature_columns %||% recipe$features
  if (is.null(feature_columns)) feature_columns <- character(0)

  # Step 1: Prepare (if not already done)
  tryCatch({
    ds.flower.nodes.prepare(conns, symbol,
      target_column   = target_column,
      feature_columns = if (length(feature_columns) > 0) feature_columns else NULL,
      privacy         = privacy,
      template_name   = template_name,
      label_set       = label_set)
  }, error = function(e) {
    # If prepare fails because already prepared, continue
    if (!grepl("already prepared|staging", conditionMessage(e), ignore.case = TRUE))
      stop(e)
  })

  # Step 2: SuperLink (start if not running)
  started_superlink <- FALSE
  sl_status <- ds.flower.superlink.status()
  if (!isTRUE(sl_status$running)) {
    ds.flower.superlink.start(detached = detached)
    started_superlink <- TRUE
  }

  # Step 3: Ensure SuperNodes
  tryCatch(
    ds.flower.nodes.ensure(conns, symbol, template_name = template_name),
    error = function(e) {
      if (started_superlink) ds.flower.superlink.stop()
      stop(e)
    }
  )

  # Step 4: Run training
  run <- tryCatch(
    ds.flower.run.start(recipe, conns, verbose = verbose),
    error = function(e) {
      if (started_superlink && !detached) ds.flower.superlink.stop()
      stop(e)
    }
  )

  # Step 5: Stop SuperLink (only if we started it and not detached)
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
  ds.flower.run(flower, recipe, ...)
}
