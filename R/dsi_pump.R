# DSI tunnel relay pump (researcher side). Runs in the main R process, driven by
# the run wait loop: each call carries one batch of Flower Fleet-API bytes between
# the local SuperLink and the nodes' tunnel forwarders, entirely over DataSHIELD.
# A separate process is unnecessary because the SuperNodes connect to their
# forwarders only after nodes.ensure returns and retry until the pump starts.

#' @keywords internal
.tunnel_enc_client <- function(raw) {
  if (length(raw) == 0) return("")
  b <- gsub("[\r\n]", "", jsonlite::base64_enc(raw))
  b <- gsub("\\+", "-", b); b <- gsub("/", "_", b); b <- gsub("=+$", "", b)
  paste0("B64:", b)
}

#' @keywords internal
.tunnel_dec_client <- function(s) {
  if (!is.character(s) || length(s) != 1 || !nzchar(s) || !startsWith(s, "B64:")) {
    return(raw(0))
  }
  b <- substring(s, 5); b <- gsub("-", "+", b); b <- gsub("_", "/", b)
  pad <- (4 - nchar(b) %% 4) %% 4; if (pad > 0) b <- paste0(b, strrep("=", pad))
  jsonlite::base64_dec(b)
}

#' Carry one batch of tunnel bytes (returns TRUE if the tunnel is active)
#' @keywords internal
.tunnel_pump <- function() {
  st <- .dsflower_client_env$.tunnel
  if (is.null(st) || !isTRUE(st$active)) return(FALSE)
  conns <- st$conns; hosts <- st$hosts; cid <- st$conn_id; fp <- st$fleet_port
  changed <- FALSE

  # up: poll every node in one fan-out; write each node's bytes to its (lazily
  # opened) SuperLink socket. The socket is opened only once a node actually has
  # bytes, i.e. its SuperNode has connected -- avoids idle-timing out the Fleet API.
  ups <- tryCatch(DSI::datashield.aggregate(conns, call("flowerTunnelPollDS", cid)),
                  error = function(e) NULL)
  if (!is.null(ups)) {
    for (i in seq_along(hosts)) {
      u <- ups[[hosts[i]]]
      if (is.character(u) && nzchar(u)) {
        if (is.null(st$socks[[i]])) {
          st$socks[[i]] <- tryCatch(
            socketConnection("127.0.0.1", fp, open = "r+b", blocking = FALSE,
                             timeout = 10),
            error = function(e) NULL)
          changed <- TRUE
        }
        s <- st$socks[[i]]
        if (!is.null(s)) tryCatch({ writeBin(.tunnel_dec_client(u), s); flush(s) },
                                  error = function(e) NULL)
      }
    }
  }

  # down: read from each open SuperLink socket; push to its node.
  for (i in seq_along(hosts)) {
    s <- st$socks[[i]]; if (is.null(s)) next
    down <- tryCatch(readBin(s, "raw", 1e6), error = function(e) raw(0))
    if (length(down) > 0) {
      tryCatch(DSI::datashield.aggregate(conns[hosts[i]],
        call("flowerTunnelPushDS", cid, .tunnel_enc_client(down))),
        error = function(e) NULL)
    }
  }

  if (changed) .dsflower_client_env$.tunnel <- st
  TRUE
}
