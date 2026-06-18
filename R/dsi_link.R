# Module: federation link (DSI tunnel transport)
# Brings up a local insecure Flower SuperLink and a node-side tunnel forwarder on
# each site, so every SuperNode dials its own loopback forwarder and its whole
# Fleet-API conversation with the SuperLink is carried over the (TLS) DataSHIELD
# channel. No public host, account, key, Tor or tailnet -- zero researcher setup.
# Byte shuttling happens in .tunnel_pump(), driven by the run wait loop.

#' Open the federation link
#'
#' Starts a local Flower SuperLink and connects each node (SuperNode) to it over
#' the DataSHIELD channel, so training can run with no public host, account or
#' key required. Pass \code{verbose = TRUE} to see the internal transport details.
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

  # Insecure local SuperLink: the bytes already travel inside the TLS DataSHIELD
  # channel, so the inner Flower gRPC needs no TLS of its own.
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
  # Per-node so each forwarder also learns its node's federation name, which
  # flowerTunnelExchangeDS uses to pick its slice out of the fan-out payload.
  for (srv in names(conns)) {
    r <- tryCatch(DSI::datashield.aggregate(
           conns[srv], call("flowerTunnelUpDS", cid, fwd_port, srv)),
           error = function(e) NULL)
    if (is.null(r) || !isTRUE(r[[srv]]$ok))
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
