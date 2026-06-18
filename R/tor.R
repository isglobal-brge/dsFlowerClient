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
#' on \code{ds.flower.link.down()}. Single onion (non-anonymous) keeps it fast;
#' confidentiality is unchanged (onion encryption + Flower TLS on top).
#'
#' Clear any leftover local Tor state from a previous (re)run
#' @keywords internal
.client_tor_local_cleanup <- function() {
  e <- .dsflower_client_env
  if (!is.null(e$.tor_control)) {
    tryCatch(close(e$.tor_control), error = function(x) NULL); e$.tor_control <- NULL
  }
  if (!is.null(e$.tor_proc) && inherits(e$.tor_proc, "process")) {
    tryCatch(e$.tor_proc$kill(), error = function(x) NULL); e$.tor_proc <- NULL
  }
  if (!is.null(e$.tor_dir) && dir.exists(e$.tor_dir)) {
    unlink(e$.tor_dir, recursive = TRUE, force = TRUE)
  }
  e$.tor_dir <- NULL
  invisible(NULL)
}

#' @keywords internal
.client_tor_hidden_service <- function(target_port, timeout = 180) {
  # Idempotent: drop any leftover state so re-runs never accumulate or clash.
  .client_tor_local_cleanup()
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
    # tor creates tor.log a moment after it starts; reading it before then makes
    # file() emit a "cannot open file" warning (readLines' warn=FALSE does not
    # cover the connection-open warning). Only read once it exists.
    lg <- if (file.exists(logf))
            suppressWarnings(tryCatch(readLines(logf, warn = FALSE),
                                      error = function(e) character(0)))
          else character(0)
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
  # open, and disappears when ds.flower.link.down() closes it.
  .dsflower_client_env$.tor_control <- con
  paste0(sid, ".onion")
}

#' Open the federation link
#'
#' Starts a local Flower SuperLink and connects each node (SuperNode) to it, so
#' training can run with no public host, account or key required. The transport
#' is set up automatically; pass \code{verbose = TRUE} to see its details.
#'
#' @param conns DSI connections object.
#' @param symbol Character; handle symbol (default "flower").
#' @param verbose Logical; show the internal transport details (SuperLink
#'   PID/ports and the transient address it is published at). Defaults to
#'   \code{getOption("dsflower.verbose", FALSE)} -- off, because this is internal
#'   plumbing the user does not need to see. The address is still returned
#'   invisibly for programmatic use.
#' @return Invisible the published transport address.
#' @export
ds.flower.link.up <- function(conns, symbol = "flower",
                              verbose = getOption("dsflower.verbose", FALSE)) {
  verbose <- isTRUE(verbose)
  vmsg <- function(...) if (verbose) message(...)

  # Bring up an INSECURE local SuperLink (the DataSHIELD channel already provides
  # TLS) and start a node-side tunnel forwarder on each site. Each SuperNode then
  # dials its own loopback forwarder and its entire Fleet-API conversation with
  # the SuperLink is carried over DataSHIELD -- no public host, Tor or tailnet.
  options(dsflower.superlink_insecure = TRUE)
  sl <- ds.flower.superlink.status()
  if (!isTRUE(sl$running)) {
    if (verbose) ds.flower.superlink.start(insecure = TRUE)
    else suppressMessages(ds.flower.superlink.start(insecure = TRUE))
    sl <- ds.flower.superlink.status()
  }
  fleet_port <- sl$ports$fleet %||% 9092L

  if (!verbose)
    message("Connecting your machine to the federation (this can take ~1 min)...")

  cid <- paste0("dsf", paste(sample(c(letters, 0:9), 10, replace = TRUE), collapse = ""))
  fwd_port <- as.integer(getOption("dsflower.tunnel_port", 18080L))
  vmsg("Starting DSI tunnel forwarders on ", length(conns), " node(s)...")
  ok <- DSI::datashield.aggregate(conns, call("flowerTunnelUpDS", cid, fwd_port))
  for (srv in names(conns)) {
    if (!isTRUE(ok[[srv]]$ok))
      warning(srv, ": tunnel forwarder did not report ready.", call. = FALSE)
  }

  .dsflower_client_env$.tunnel <- list(
    active     = TRUE,
    conns      = conns,
    hosts      = names(conns),
    conn_id    = cid,
    fleet_port = fleet_port,
    fwd_port   = fwd_port,
    socks      = vector("list", length(conns))
  )

  if (verbose)
    message("DSI tunnel ready. Run ds.flower.fit() / nodes.ensure() as usual.")
  else
    message("Federation ready: ", length(conns), " site",
            if (length(conns) != 1) "s" else "", " connected.")
  invisible(cid)
}

#' Close the federation link
#'
#' Stops each node's transport, tears down the local SuperLink's published
#' address and stops the local transport process. Reverses
#' \code{ds.flower.link.up()}.
#'
#' @param conns DSI connections object.
#' @param symbol Character; handle symbol.
#' @return Invisible TRUE.
#' @export
ds.flower.link.down <- function(conns, symbol = "flower") {
  st <- .dsflower_client_env$.tunnel
  # Close the relay's local SuperLink sockets.
  if (!is.null(st) && !is.null(st$socks)) {
    for (s in st$socks) if (!is.null(s)) tryCatch(close(s), error = function(e) NULL)
  }
  # Tell each node to stop its forwarder and wipe the spool.
  cid <- if (!is.null(st)) st$conn_id else NULL
  if (!is.null(cid)) {
    tryCatch(DSI::datashield.aggregate(conns, call("flowerTunnelDownDS", cid)),
             error = function(e) NULL)
  }
  .dsflower_client_env$.tunnel <- NULL
  options(dsflower.superlink_insecure = NULL)
  # Stop the local insecure SuperLink we started for the tunnel.
  tryCatch(ds.flower.superlink.stop(), error = function(e) NULL)
  invisible(TRUE)
}
