# Module: Tor transport (zero-account, zero-key NAT traversal)
# Publishes the local SuperLink as a Tor hidden service (.onion) and brings the
# data nodes onto Tor so their SuperNodes reach it -- no account, no auth key,
# no public host. The most "comfortable" transport: zero researcher setup.

#' @keywords internal
.client_tor_bin <- function() {
  pth <- Sys.which("tor")
  if (nzchar(pth)) return(pth)
  stop("tor not found. Install Tor (brew install tor / apt install tor) or it ",
       "will be auto-provisioned in a future release.", call. = FALSE)
}

#' Send one Tor control-port command, return its response lines
#' @keywords internal
.tor_ctl_cmd <- function(con, cmd) {
  writeLines(cmd, con)
  out <- character(0)
  repeat {
    line <- tryCatch(readLines(con, n = 1), error = function(e) character(0))
    if (length(line) == 0) break
    line <- sub("\r$", "", line)
    out <- c(out, line)
    if (grepl("^[0-9]{3} ", line)) break   # final line: "NNN <text>"
  }
  out
}

#' Find a free local TCP port
#' @keywords internal
.client_free_port <- function(start = 9100L) {
  p <- start
  while (.port_is_listening(p)) p <- p + 1L
  p
}

#' Publish a local port as an EPHEMERAL, in-memory Tor v3 single onion service.
#'
#' Uses the control port + ADD_ONION with Flags=DiscardPK so the service key
#' NEVER touches disk and the .onion vanishes the moment the control connection
#' closes (teardown). Everything lives in a per-session temp dir that is wiped
#' on \code{ds.flower.tor.down()}. Single onion (non-anonymous) keeps it fast;
#' confidentiality is unchanged (onion encryption + Flower TLS on top).
#'
#' @keywords internal
.client_tor_hidden_service <- function(target_port, timeout = 180) {
  d <- tempfile("dsflower_tor_"); dir.create(d, recursive = TRUE, showWarnings = FALSE)
  .dsflower_client_env$.tor_dir <- d
  logf <- file.path(d, "tor.log"); torrc <- file.path(d, "torrc")
  cookief <- file.path(d, "control.cookie")
  ctrl_port <- .client_free_port()

  writeLines(c(
    "SocksPort 0",
    # Single onion service: no anonymity needed for the SuperLink, so 3 hops
    # become 1 (~halves RTT). Confidentiality is unchanged.
    "HiddenServiceNonAnonymousMode 1",
    "HiddenServiceSingleHopMode 1",
    paste0("ControlPort 127.0.0.1:", ctrl_port),
    "CookieAuthentication 1",
    paste0("CookieAuthFile ", cookief),
    "ConnectionPadding 0", "ReducedConnectionPadding 1",
    paste0("DataDirectory ", file.path(d, "data")),
    paste0("Log notice file ", logf)
  ), torrc)

  p <- processx::process$new(.client_tor_bin(), c("-f", torrc),
    stdout = file.path(d, "tor.out"), stderr = "2>&1",
    cleanup = FALSE, cleanup_tree = FALSE)
  .dsflower_client_env$.tor_proc <- p

  deadline <- Sys.time() + timeout
  repeat {
    lg <- tryCatch(readLines(logf, warn = FALSE), error = function(e) character(0))
    if (any(grepl("Bootstrapped 100%", lg))) break
    if (Sys.time() > deadline || !p$is_alive())
      stop("Tor did not bootstrap in time.", call. = FALSE)
    Sys.sleep(2)
  }

  # The control cookie appears when tor opens the control port (before bootstrap).
  for (i in 1:30) { if (file.exists(cookief) && file.info(cookief)$size > 0) break; Sys.sleep(0.5) }
  con <- socketConnection("127.0.0.1", ctrl_port, open = "r+",
                          blocking = TRUE, timeout = 30)
  cf <- file(cookief, "rb"); cookie_raw <- readBin(cf, "raw", n = 64L); close(cf)
  cookie_hex <- paste(as.character(cookie_raw), collapse = "")  # raw -> hex
  if (!any(grepl("^250", .tor_ctl_cmd(con, paste0("AUTHENTICATE ", cookie_hex))))) {
    close(con); stop("Tor control authentication failed.", call. = FALSE)
  }
  resp <- .tor_ctl_cmd(con, paste0(
    "ADD_ONION NEW:ED25519-V3 Flags=DiscardPK,NonAnonymous Port=",
    target_port, ",127.0.0.1:", target_port))
  sid_line <- grep("ServiceID=", resp, value = TRUE)
  sid <- if (length(sid_line)) sub(".*ServiceID=([a-z2-7]+).*", "\\1", sid_line[1]) else NA
  if (is.na(sid) || !nzchar(sid)) {
    close(con); stop("ADD_ONION failed: ", paste(resp, collapse = " "), call. = FALSE)
  }
  # Keep the control connection open: the ephemeral onion lives only while it is
  # open, and disappears when ds.flower.tor.down() closes it.
  .dsflower_client_env$.tor_control <- con
  paste0(sid, ".onion")
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
  # Tell the nodes to stop Tor + wipe their session state.
  for (srv in names(conns)) {
    tryCatch(DSI::datashield.assign.expr(
      conns[srv], symbol = paste0(symbol, "_tor"),
      expr = call("flowerTorDownDS")), error = function(e) NULL)
  }
  # Close the control connection -> the ephemeral in-memory onion is destroyed.
  con <- .dsflower_client_env$.tor_control
  if (!is.null(con)) {
    tryCatch({ .tor_ctl_cmd(con, "QUIT"); close(con) }, error = function(e) NULL)
  }
  .dsflower_client_env$.tor_control <- NULL
  # Stop the local Tor process.
  p <- .dsflower_client_env$.tor_proc
  if (!is.null(p) && inherits(p, "process")) {
    tryCatch({ p$signal(15L); p$wait(timeout = 3000); if (p$is_alive()) p$kill() },
             error = function(e) NULL)
  }
  .dsflower_client_env$.tor_proc <- NULL
  # Wipe the per-session temp dir -> no residue left on disk.
  d <- .dsflower_client_env$.tor_dir
  if (!is.null(d) && dir.exists(d)) unlink(d, recursive = TRUE, force = TRUE)
  .dsflower_client_env$.tor_dir <- NULL
  .dsflower_client_env$.overlay_ip <- NULL
  invisible(TRUE)
}
