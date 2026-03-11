# Module: flwr CLI Integration
# Controls Flower runs via the flwr CLI.

#' Start a Flower run
#'
#' Builds a Flower App from the recipe, then invokes \code{flwr run} against
#' the running SuperLink. The SuperLink must have been started with
#' \code{ds.flower.superlink.start()} beforehand.
#'
#' @param recipe A \code{dsflower_recipe} object.
#' @param app_dir Character; path to a pre-built app directory (optional).
#' @param run_config Named list; additional run config overrides.
#' @param verbose Logical; print flwr output (default TRUE).
#' @return A list with run_id, app_dir, and output.
#' @export
ds.flower.run.start <- function(recipe, app_dir = NULL,
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

  # Build app if no pre-built dir provided
  if (is.null(app_dir)) {
    app_dir <- .build_flower_app(recipe)
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

  if (verbose) {
    if (nchar(result$stdout) > 0) message(result$stdout)
    if (nchar(result$stderr) > 0) message(result$stderr)
  }

  run_id <- .parse_run_id(result$stdout)

  list(
    run_id   = run_id,
    app_dir  = app_dir,
    status   = result$status,
    stdout   = result$stdout,
    stderr   = result$stderr
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
