# Module: federation link (DSI tunnel transport)
# Brings up a local insecure Flower SuperLink and a node-side tunnel forwarder on
# each site, so every SuperNode dials its own loopback forwarder and its whole
# Fleet-API conversation with the SuperLink is carried over the (TLS) DataSHIELD
# channel. No public host, account, key, Tor or tailnet -- zero researcher setup.
# Byte shuttling happens in .tunnel_pump(), driven by the run wait loop.

#' Generate an unguessable capability for one tunnel session
#' @keywords internal
.new_tunnel_conn_id <- function() {
  .new_capability_token("dsf")
}

#' Validate a TCP port used only by the node-local tunnel forwarder
#' @keywords internal
.client_tunnel_port <- function(port) {
  value <- suppressWarnings(as.numeric(port))
  if (length(value) != 1L || is.na(value) || !is.finite(value) ||
      value != floor(value) || value < 1 || value > 65535) {
    stop("Invalid dsflower.tunnel_port option.", call. = FALSE)
  }
  as.integer(value)
}

#' Validate a node-advertised tunnel chunk size
#' @keywords internal
.client_tunnel_chunk_bytes <- function(value) {
  value <- suppressWarnings(as.numeric(value))
  if (length(value) != 1L || is.na(value) || !is.finite(value) ||
      value != floor(value) || value < 16 * 1024 || value > 8 * 1024^2) {
    stop("Node advertised an invalid tunnel chunk size.", call. = FALSE)
  }
  as.integer(value)
}

#' Open the federation link
#'
#' Starts a local Flower SuperLink and connects each node (SuperNode) to it over
#' the DataSHIELD channel, so training can run with no public host, account or
#' key required. Startup is all-or-nothing: if any node does not publish a live,
#' ready forwarder, every attempted node is torn down and the call fails. Pass
#' \code{verbose = TRUE} to see the internal transport details.
#'
#' @param conns DSI connections object.
#' @param symbol Character; handle symbol (default "flower").
#' @param verbose Logical; show internal transport details (SuperLink PID/ports).
#'   Defaults to \code{getOption("dsflower.verbose", FALSE)} -- off, because this
#'   is internal plumbing the user does not need to see.
#' @return Invisibly, the tunnel connection id.
#' @export
ds.flower.link.up <- function(conns, symbol = "flower",
                              verbose = getOption("dsflower.verbose", FALSE)) {
  verbose <- isTRUE(verbose)
  vmsg <- function(...) if (verbose) message(...)
  fwd_port <- .client_tunnel_port(getOption("dsflower.tunnel_port", 18080L))
  hosts <- names(conns)
  if (length(hosts) == 0L || anyNA(hosts) || any(!nzchar(hosts)) ||
      anyDuplicated(hosts)) {
    stop("Tunnel connections must have non-empty, unique node names.",
         call. = FALSE)
  }
  previous_insecure <- getOption("dsflower.superlink_insecure", NULL)
  superlink_started_here <- FALSE
  attempted <- character()
  committed <- FALSE

  on.exit({
    if (!committed) {
      for (srv in attempted) {
        tryCatch(
          DSI::datashield.aggregate(
            conns[srv], call("flowerTunnelDownDS", cid)
          ),
          error = function(e) NULL
        )
      }
      .dsflower_client_env$.tunnel <- NULL
      options(dsflower.superlink_insecure = previous_insecure)
      if (superlink_started_here) {
        tryCatch(ds.flower.superlink.stop(), error = function(e) NULL)
      }
    }
  }, add = TRUE)

  # Insecure local SuperLink: the bytes already travel inside the TLS DataSHIELD
  # channel, so the inner Flower gRPC needs no TLS of its own.
  options(dsflower.superlink_insecure = TRUE)
  sl <- ds.flower.superlink.status()
  if (!isTRUE(sl$running)) {
    # Mark ownership before spawning so a start error also triggers rollback.
    superlink_started_here <- TRUE
    if (verbose) ds.flower.superlink.start(insecure = TRUE)
    else suppressMessages(ds.flower.superlink.start(insecure = TRUE))
    sl <- ds.flower.superlink.status()
  }
  if (!isTRUE(sl$running)) {
    stop("Local Flower SuperLink failed to start.", call. = FALSE)
  }
  fleet_port <- sl$ports$fleet %||% 9092L

  .dsf_msg("Connecting to the federation...")

  cid <- .new_tunnel_conn_id()
  chunk_bytes <- stats::setNames(integer(length(hosts)), hosts)
  vmsg("Starting DSI tunnel forwarders on ", length(hosts), " node(s)...")
  # Per-node so each forwarder also learns its node's federation name, which
  # flowerTunnelExchangeDS uses to pick its slice out of the fan-out payload.
  for (srv in hosts) {
    attempted <- c(attempted, srv)
    r <- tryCatch(
      DSI::datashield.aggregate(
        conns[srv], call("flowerTunnelUpDS", cid, fwd_port, srv)
      ),
      error = function(e) NULL
    )
    node_result <- if (!is.null(r)) r[[srv]] else NULL
    if (is.null(node_result) || !isTRUE(node_result$ok)) {
      stop(srv, ": tunnel forwarder did not report ready; federation aborted.",
           call. = FALSE)
    }
    chunk_bytes[[srv]] <- .client_tunnel_chunk_bytes(
      node_result$chunk_bytes %||% 1024^2
    )
  }

  .dsflower_client_env$.tunnel <- list(
    active     = TRUE,
    conns      = conns,
    hosts      = hosts,
    conn_id    = cid,
    fleet_port = fleet_port,
    fwd_port   = fwd_port,
    chunk_bytes = unname(chunk_bytes),
    socks      = vector("list", length(hosts))
  )
  committed <- TRUE

  .dsf_msg("Federation ready: ", length(hosts), " site",
           if (length(hosts) != 1) "s" else "", " connected.")
  invisible(cid)
}

#' Close the federation link
#'
#' Stops each node's tunnel forwarder, closes the relay sockets and stops the
#' local SuperLink. Reverses \code{ds.flower.link.up()}.
#'
#' @param conns DSI connections object.
#' @param symbol Character; handle symbol.
#' @return Invisibly, TRUE.
#' @export
ds.flower.link.down <- function(conns, symbol = "flower") {
  st <- .dsflower_client_env$.tunnel
  if (!is.null(st) && !is.null(st$socks)) {
    for (s in st$socks) if (!is.null(s)) tryCatch(close(s), error = function(e) NULL)
  }
  cid <- if (!is.null(st)) st$conn_id else NULL
  if (!is.null(cid)) {
    tryCatch(DSI::datashield.aggregate(conns, call("flowerTunnelDownDS", cid)),
             error = function(e) NULL)
  }
  .dsflower_client_env$.tunnel <- NULL
  options(dsflower.superlink_insecure = NULL)
  tryCatch(ds.flower.superlink.stop(), error = function(e) NULL)
  invisible(TRUE)
}
