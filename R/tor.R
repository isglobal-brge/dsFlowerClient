# Module: Tor transport (zero-account, zero-key NAT traversal)
# Publishes the local SuperLink as a Tor hidden service (.onion) and brings the
# data nodes onto Tor so their SuperNodes reach it -- no account, no auth key,
# no public host. The most "comfortable" transport: zero researcher setup.

#' @keywords internal
.client_tor_dir <- function() {
  d <- file.path(Sys.getenv("HOME", "~"), ".dsflower", "tor")
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}

#' @keywords internal
.client_tor_bin <- function() {
  pth <- Sys.which("tor")
  if (nzchar(pth)) return(pth)
  stop("tor not found. Install Tor (brew install tor / apt install tor) or it ",
       "will be auto-provisioned in a future release.", call. = FALSE)
}

#' Publish a local port as a Tor v3 hidden service, return the .onion address
#' @keywords internal
.client_tor_hidden_service <- function(target_port, timeout = 120) {
  d <- .client_tor_dir()
  hs <- file.path(d, "hs"); dir.create(hs, showWarnings = FALSE); Sys.chmod(hs, "0700")
  logf <- file.path(d, "tor.log")
  torrc <- file.path(d, "torrc")
  writeLines(c(
    "SocksPort 0",
    # Single onion service: we only need NAT traversal + encryption, NOT
    # anonymity for the SuperLink, so drop the service side from 3 hops to 1
    # (end-to-end ~6 -> ~3-4 hops, ~halves RTT). Requires SocksPort 0.
    "HiddenServiceNonAnonymousMode 1",
    "HiddenServiceSingleHopMode 1",
    paste0("DataDirectory ", file.path(d, "data")),
    paste0("HiddenServiceDir ", hs),   # persistent key -> stable, pre-publishable .onion
    "HiddenServiceVersion 3",
    paste0("HiddenServicePort ", target_port, " 127.0.0.1:", target_port),
    paste0("Log notice file ", logf)
  ), torrc)
  running <- !is.null(.dsflower_client_env$.tor_proc) &&
    inherits(.dsflower_client_env$.tor_proc, "process") &&
    tryCatch(.dsflower_client_env$.tor_proc$is_alive(), error = function(e) FALSE)
  if (!running) {
    p <- processx::process$new(.client_tor_bin(), c("-f", torrc),
      stdout = file.path(d, "tor.out"), stderr = "2>&1",
      cleanup = FALSE, cleanup_tree = FALSE)
    .dsflower_client_env$.tor_proc <- p
  }
  hostfile <- file.path(hs, "hostname")
  deadline <- Sys.time() + timeout
  repeat {
    lg <- tryCatch(readLines(logf, warn = FALSE), error = function(e) character(0))
    if (any(grepl("Bootstrapped 100%", lg)) && file.exists(hostfile)) break
    if (Sys.time() > deadline) stop("Tor did not publish the hidden service in time.", call. = FALSE)
    Sys.sleep(2)
  }
  trimws(readLines(hostfile, warn = FALSE)[1])
}

#' Connect the federation over Tor (zero account, zero key)
#'
#' Publishes the local SuperLink as a Tor hidden service and brings each node
#' onto Tor, so SuperNodes reach the SuperLink with no public host, no overlay
#' account and no auth key. Slower than Tailscale (~MB/s) but zero setup.
#'
#' @param conns DSI connections object.
#' @param symbol Character; handle symbol (default "flower").
#' @return Invisible the .onion address.
#' @export
ds.flower.tor.up <- function(conns, symbol = "flower") {
  sl <- ds.flower.superlink.status()
  if (!isTRUE(sl$running)) {
    ds.flower.superlink.start()
    sl <- ds.flower.superlink.status()
  }
  fleet_port <- sl$ports$fleet %||% 9092L

  message("Publishing SuperLink as a Tor hidden service (bootstrapping Tor)...")
  onion <- .client_tor_hidden_service(fleet_port)
  message("  SuperLink onion: ", onion, ":", fleet_port)

  message("Bringing nodes onto Tor...")
  for (srv in names(conns)) {
    tryCatch(DSI::datashield.assign.expr(
      conns[srv], symbol = paste0(symbol, "_tor"),
      expr = call("flowerTorUpDS")), error = function(e)
        warning(srv, ": flowerTorUpDS failed: ", conditionMessage(e), call. = FALSE))
  }
  .dsflower_client_env$.overlay_ip <- onion  # ensure -> onion:fleet_port
  message("Tor transport ready. Run ds.flower.fit() / nodes.ensure() as usual.")
  invisible(onion)
}

#' Tear down the Tor transport
#' @param conns DSI connections object.
#' @param symbol Character; handle symbol.
#' @return Invisible TRUE.
#' @export
ds.flower.tor.down <- function(conns, symbol = "flower") {
  for (srv in names(conns)) {
    tryCatch(DSI::datashield.assign.expr(
      conns[srv], symbol = paste0(symbol, "_tor"),
      expr = call("flowerTorDownDS")), error = function(e) NULL)
  }
  p <- .dsflower_client_env$.tor_proc
  if (!is.null(p) && inherits(p, "process")) {
    tryCatch({ p$signal(15L); p$wait(timeout = 3000); if (p$is_alive()) p$kill() },
             error = function(e) NULL)
  }
  .dsflower_client_env$.tor_proc <- NULL
  .dsflower_client_env$.overlay_ip <- NULL
  invisible(TRUE)
}
