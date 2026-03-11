# Module: SuperLink Lifecycle
# Manages the Flower SuperLink process on the researcher's machine.

#' Start a Flower SuperLink
#'
#' Spawns a \code{flower-superlink} process using processx with a private
#' FLWR_HOME directory. Writes a \code{config.toml} so that \code{flwr run}
#' can connect to this SuperLink.
#'
#' @param insecure Logical; use insecure mode (default TRUE).
#' @param fleet_port Integer; port for the Fleet API (default 9092).
#'   SuperNodes connect here.
#' @param control_port Integer; port for the Control API (default 9093).
#'   \code{flwr run} connects here.
#' @param serverappio_port Integer; port for the ServerAppIO API (default 9091).
#' @return Invisible list with process info.
#' @export
ds.flower.superlink.start <- function(insecure = TRUE,
                                       fleet_port = 9092L,
                                       control_port = 9093L,
                                       serverappio_port = 9091L) {
  .require_flwr_cli()

  # Check if already running
  existing <- .dsflower_client_env$.superlink
  if (!is.null(existing) && !is.null(existing$process) &&
      existing$process$is_alive()) {
    message("SuperLink is already running (PID: ", existing$process$get_pid(), ")")
    return(invisible(existing))
  }

  # Private FLWR_HOME
  flwr_home <- file.path(tempdir(), "dsflower_superlink")
  dir.create(flwr_home, recursive = TRUE, showWarnings = FALSE)

  # Build args for flower-superlink
  args <- character(0)
  if (insecure) args <- c(args, "--insecure")
  args <- c(args,
    "--fleet-api-address", paste0("0.0.0.0:", fleet_port),
    "--control-api-address", paste0("0.0.0.0:", control_port),
    "--serverappio-api-address", paste0("0.0.0.0:", serverappio_port)
  )

  # Log path
  log_path <- file.path(flwr_home, "superlink.log")

  # Spawn
  proc <- processx::process$new(
    command = "flower-superlink",
    args = args,
    stdout = log_path,
    stderr = "2>&1",
    cleanup = TRUE,
    cleanup_tree = TRUE,
    env = c("current", FLWR_HOME = flwr_home)
  )

  # Write config.toml so flwr run can find this SuperLink
  config_toml <- paste0(
    "[superlink]\n",
    'default = "dsflower"\n\n',
    "[superlink.dsflower]\n",
    'address = "127.0.0.1:', control_port, '"\n',
    "insecure = true\n"
  )
  writeLines(config_toml, file.path(flwr_home, "config.toml"))

  fleet_address   <- paste0("127.0.0.1:", fleet_port)
  control_address <- paste0("127.0.0.1:", control_port)

  info <- list(
    process          = proc,
    pid              = proc$get_pid(),
    fleet_address    = fleet_address,
    control_address  = control_address,
    fleet_port       = fleet_port,
    control_port     = control_port,
    serverappio_port = serverappio_port,
    flwr_home        = flwr_home,
    log_path         = log_path,
    started_at       = Sys.time()
  )

  .dsflower_client_env$.superlink <- info
  message("SuperLink started (PID: ", info$pid, ")")
  message("  Fleet API (SuperNodes): ", fleet_address)
  message("  Control API (flwr run): ", control_address)
  invisible(info)
}

#' Stop the Flower SuperLink
#'
#' Sends SIGTERM, waits, then SIGKILL if needed. Cleans up temp files.
#'
#' @return Invisible TRUE.
#' @export
ds.flower.superlink.stop <- function() {
  info <- .dsflower_client_env$.superlink
  if (is.null(info) || is.null(info$process)) {
    message("No SuperLink is running.")
    return(invisible(TRUE))
  }

  proc <- info$process
  if (proc$is_alive()) {
    proc$signal(15L)  # SIGTERM
    proc$wait(timeout = 5000)
    if (proc$is_alive()) {
      proc$kill()
    }
  }

  # Cleanup temp directory
  if (!is.null(info$flwr_home) && dir.exists(info$flwr_home)) {
    unlink(info$flwr_home, recursive = TRUE)
  }

  .dsflower_client_env$.superlink <- NULL
  message("SuperLink stopped.")
  invisible(TRUE)
}

#' Get SuperLink status
#'
#' @return A named list with running, pid, fleet_address, control_address,
#'   ports, started_at.
#' @export
ds.flower.superlink.status <- function() {
  info <- .dsflower_client_env$.superlink
  if (is.null(info) || is.null(info$process)) {
    return(list(
      running         = FALSE,
      pid             = NULL,
      fleet_address   = NULL,
      control_address = NULL,
      ports           = NULL,
      started_at      = NULL
    ))
  }

  list(
    running         = info$process$is_alive(),
    pid             = info$pid,
    fleet_address   = info$fleet_address,
    control_address = info$control_address,
    ports           = list(
      fleet       = info$fleet_port,
      control     = info$control_port,
      serverappio = info$serverappio_port
    ),
    started_at      = info$started_at
  )
}
