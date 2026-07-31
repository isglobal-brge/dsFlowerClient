# Module: federation link (DSI tunnel transport)
# Brings up a local insecure Flower SuperLink and a node-side tunnel forwarder on
# each site, so every SuperNode dials its own loopback forwarder and its whole
# Fleet-API conversation with the SuperLink is carried over a DataSHIELD channel
# whose transport passes the connector-aware security preflight before startup.
# No public host, account, key, Tor or tailnet -- zero researcher setup.
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

#' Resolve one node-local forwarder port per DataSHIELD connection
#' @keywords internal
.client_tunnel_ports <- function(port, hosts) {
  n <- length(hosts)
  if (length(port) == 1L) {
    return(stats::setNames(rep(.client_tunnel_port(port), n), hosts))
  }
  if (length(port) != n) {
    stop(
      "dsflower.tunnel_port must be one port or one port per node.",
      call. = FALSE
    )
  }
  if (!is.null(names(port))) {
    port_names <- names(port)
    if (anyNA(port_names) || any(!nzchar(port_names)) ||
        anyDuplicated(port_names) || !setequal(port_names, hosts)) {
      stop(
        "Named dsflower.tunnel_port values must match the node names.",
        call. = FALSE
      )
    }
    port <- port[hosts]
  }
  stats::setNames(
    vapply(as.list(port), .client_tunnel_port, integer(1)), hosts
  )
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
#' \code{options(dsflower.tunnel_port = ...)} accepts one shared port or one
#' port per node (positional or named), which permits multiple node sessions in
#' the same Rock/container without listener collisions.
#'
#' Transport security is checked before any local service or node forwarder is
#' started. DSOpal requires HTTPS with retained certificate verification enabled.
#' A recognized DSMolgenisArmadillo connection is accepted automatically when it
#' exposes a valid HTTPS endpoint; its frontend or reverse proxy remains
#' responsible for certificate and hostname validation. DSLite is accepted as an
#' in-process transport. For an unknown or unidentifiable connector, an operator
#' who has independently verified transport confidentiality, integrity, and peer
#' authentication must list the exact names in
#' \code{options(dsflower.dsi_tls_attested = c("site1", "site2"))}.
#' Plaintext HTTP is rejected by default, but an exact site can be allowed
#' deliberately with \code{allow_insecure_http} or
#' \code{options(dsflower.dsi_allow_insecure_http = "site1")}; this requires an
#' independent trusted network layer for confidentiality and integrity. The
#' exception only relaxes this package's preflight; it cannot override a DSI
#' connector that rejects remote HTTP while creating the connection, and it
#' cannot protect credentials already exchanged during connection creation.
#' Plaintext or unverifiable loopback connections are also available for explicit
#' local development with
#' \code{options(dsflower.dsi_allow_insecure_loopback = TRUE)}.
#' This client-side gate prevents accidental downgrade; production DataSHIELD
#' frontends must also reject plaintext independently.
#'
#' @param conns DSI connections object.
#' @param symbol Character; handle symbol (default "flower").
#' @param verbose Logical; show internal transport details (SuperLink PID/ports).
#'   Defaults to \code{getOption("dsflower.verbose", FALSE)} -- off, because this
#'   is internal plumbing the user does not need to see.
#' @param allow_insecure_http Character vector of exact connection names allowed
#'   to use plaintext HTTP. Empty by default. This exception does not provide
#'   transport security; use it only behind an independently trusted network.
#' @return Invisibly, the tunnel connection id.
#' @export
ds.flower.link.up <- function(conns, symbol = "flower",
                              verbose = getOption("dsflower.verbose", FALSE),
                              allow_insecure_http = getOption(
                                "dsflower.dsi_allow_insecure_http", character())) {
  verbose <- isTRUE(verbose)
  vmsg <- function(...) if (verbose) message(...)
  hosts <- names(conns)
  if (length(hosts) == 0L || anyNA(hosts) || any(!nzchar(hosts)) ||
      anyDuplicated(hosts)) {
    stop("Tunnel connections must have non-empty, unique node names.",
         call. = FALSE)
  }
  fwd_ports <- .client_tunnel_ports(
    getOption("dsflower.tunnel_port", 18080L), hosts
  )
  .validate_dsi_transport_security(
    conns, allow_insecure_http = allow_insecure_http)
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

  # Insecure local SuperLink: the outer DataSHIELD transport passed the security
  # gate above, so the loopback-only inner Flower gRPC needs no TLS of its own.
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
    expected_listen <- paste0("127.0.0.1:", fwd_ports[[srv]])
    expr <- call(
      "flowerTunnelUpDS", cid, fwd_ports[[srv]], srv,
      protocol_abi = 2L
    )
    r <- tryCatch(
      .dsi_retry_exact_aggregate(
        conns[srv], expr,
        validate = function(value, node) {
          if (!is.list(value) ||
              !identical(names(value),
                         c("ok", "listen", "chunk_bytes", "protocol_abi"))) {
            return(FALSE)
          }
          ack_abi <- suppressWarnings(as.numeric(value$protocol_abi))
          isTRUE(value$ok) &&
            identical(as.character(value$listen), expected_listen) &&
            length(ack_abi) == 1L && is.finite(ack_abi) &&
            ack_abi == floor(ack_abi) && ack_abi == 2 &&
            !inherits(
              try(.client_tunnel_chunk_bytes(value$chunk_bytes), silent = TRUE),
              "try-error")
        },
        operation = "Tunnel forwarder startup"),
      error = function(e) {
        stop(srv, ": tunnel forwarder did not report ready; federation aborted.",
             call. = FALSE)
      })
    chunk_bytes[[srv]] <- .client_tunnel_chunk_bytes(r[[srv]]$chunk_bytes)
  }

  .dsflower_client_env$.tunnel <- list(
    active     = TRUE,
    conns      = conns,
    hosts      = hosts,
    conn_id    = cid,
    fleet_port = fleet_port,
    fwd_ports  = unname(fwd_ports),
    fwd_port   = if (length(unique(fwd_ports)) == 1L) fwd_ports[[1L]] else NULL,
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
