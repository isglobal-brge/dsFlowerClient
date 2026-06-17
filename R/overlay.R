# Module: Overlay transport (Tailscale)
# Joins the data nodes to the researcher's tailnet so SuperNodes can reach the
# local SuperLink across NAT, with no public host and no slow relay. The
# researcher's machine runs Tailscale normally; the nodes join in userspace
# mode (driven by dsFlower). The auth key is pushed to the nodes over the
# existing DataSHIELD channel.

#' Detect this machine's tailnet (Tailscale) IPv4 address
#' @keywords internal
.client_tailnet_ip <- function() {
  ts <- Sys.which("tailscale")
  if (nzchar(ts)) {
    ip <- tryCatch(trimws(system2(ts, c("ip", "-4"), stdout = TRUE, stderr = FALSE)[1]),
                   error = function(e) "")
    if (length(ip) && nzchar(ip) && grepl("^100\\.", ip)) return(ip)
  }
  # Fallback: scan interfaces for a 100.64/10 (CGNAT/tailnet) address
  ips <- tryCatch(.detect_all_ips(), error = function(e) character(0))
  cg <- ips[grepl("^100\\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\\.", ips)]
  if (length(cg)) return(cg[1])
  ""
}

#' Join the data nodes to the tailnet (overlay transport)
#'
#' Brings each node onto your Tailscale tailnet (userspace mode, no root) so its
#' SuperNode can reach the SuperLink on this machine across NAT. Requires that
#' THIS machine is already on the tailnet (install Tailscale + `tailscale up`).
#'
#' @param conns DSI connections object.
#' @param authkey Character; a reusable Tailscale auth key (tskey-...). Pushed to
#'   the nodes over DataSHIELD; prefer an ephemeral, tagged key.
#' @param symbol Character; handle symbol name (default "flower").
#' @param hostname_prefix Character; tailnet hostname prefix per node.
#' @return A \code{dsflower_result} with per-node join status.
#' @export
ds.flower.overlay.up <- function(conns, authkey, symbol = "flower",
                                 hostname_prefix = "dsflower-node") {
  if (missing(authkey) || is.null(authkey) || !nzchar(authkey)) {
    stop("'authkey' is required (a Tailscale auth key, tskey-...).", call. = FALSE)
  }
  my_ip <- .client_tailnet_ip()
  if (!nzchar(my_ip)) {
    stop("This machine is not on a tailnet. Install Tailscale and run ",
         "`tailscale up`, then retry (the SuperLink must be reachable at your ",
         "tailnet IP).", call. = FALSE)
  }
  message("This machine's tailnet IP: ", my_ip)

  results <- list()
  for (srv in names(conns)) {
    res <- tryCatch({
      DSI::datashield.assign.expr(
        conns[srv], symbol = paste0(symbol, "_overlay"),
        expr = call("flowerOverlayUpDS",
                    .ds_encode(authkey),
                    paste0(hostname_prefix, "-", srv)))
      DSI::datashield.aggregate(
        conns[srv],
        expr = call("flowerStatusDS", symbol))[[srv]]
    }, error = function(e) list(overlay_error = conditionMessage(e)))
    results[[srv]] <- res
  }

  .dsflower_client_env$.overlay_ip <- my_ip
  message("Nodes joined the tailnet. The SuperLink at ", my_ip,
          " is now reachable from the nodes. Run ds.flower.fit() / nodes.ensure() as usual.")
  dsflower_result(per_site = results,
                  meta = list(call_code = paste0("ds.flower.overlay.up(authkey=***)"),
                              scope = "per_site"))
}

#' Tear down the overlay on all nodes
#' @param conns DSI connections object.
#' @param symbol Character; handle symbol name.
#' @return Invisible TRUE.
#' @export
ds.flower.overlay.down <- function(conns, symbol = "flower") {
  for (srv in names(conns)) {
    tryCatch(DSI::datashield.assign.expr(
      conns[srv], symbol = paste0(symbol, "_overlay"),
      expr = call("flowerOverlayDownDS")), error = function(e) NULL)
  }
  .dsflower_client_env$.overlay_ip <- NULL
  invisible(TRUE)
}
