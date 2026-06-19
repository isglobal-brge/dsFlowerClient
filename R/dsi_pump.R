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
#'
#' One fan-out flowerTunnelExchangeDS per cycle. The relay OWNS the byte offsets
#' (up_off = acked node->SuperLink bytes, down_sent = acked SuperLink->node
#' bytes), so a transient DSI failure loses nothing: the next cycle re-requests
#' the same ranges and the node applies them idempotently. SuperLink replies are
#' buffered per node and only dropped once the node acks them, and the up offset
#' only advances after the bytes are confirmed written to the SuperLink socket.
#' A single fan-out call (not per-node pushes) keeps every node in lock-step so
#' none is skewed past a SecAgg stage window.
#' @keywords internal
.tunnel_pump <- function() {
  st <- .dsflower_client_env$.tunnel
  if (is.null(st) || !isTRUE(st$active)) return(FALSE)
  conns <- st$conns; hosts <- st$hosts; cid <- st$conn_id; fp <- st$fleet_port
  n <- length(hosts)
  if (is.null(st$up_off))    st$up_off    <- rep(0, n)            # acked node->SL
  if (is.null(st$down_sent)) st$down_sent <- rep(0, n)           # acked SL->node
  if (is.null(st$down_buf))  st$down_buf  <- vector("list", n)   # SL bytes pending node ack
  if (is.null(st$gen))       st$gen       <- rep(0, n)           # forwarder connection gen

  # 1) drain available SuperLink->node bytes from each open socket into its buffer
  for (i in seq_len(n)) {
    s <- st$socks[[i]]; if (is.null(s)) next
    d <- tryCatch(readBin(s, "raw", 1e6), error = function(e) raw(0))
    if (length(d) > 0) st$down_buf[[i]] <- c(st$down_buf[[i]] %||% raw(0), d)
  }

  # 2) one fan-out exchange: push pending down bytes (idempotent at down_sent) and
  #    poll up bytes from up_off, for every node at once
  req <- list()
  for (i in seq_len(n)) {
    buf <- st$down_buf[[i]] %||% raw(0)
    req[[hosts[i]]] <- list(pa = st$down_sent[i],
                            pd = if (length(buf)) .tunnel_enc_client(buf) else "",
                            pf = st$up_off[i])
  }
  res <- tryCatch(DSI::datashield.aggregate(conns, call("flowerTunnelExchangeDS", cid, .ds_encode(req))),
                  error = function(e) NULL)
  if (is.null(res)) return(TRUE)  # transient: nothing lost, retry next cycle
  if (isTRUE(getOption("dsflower.pump_debug", FALSE)))
    cat(sprintf("[%s] up_off=%s down_sent=%s gen=%s resgen=%s udlen=%s\n",
        format(Sys.time(), "%H:%M:%OS1"),
        paste(st$up_off, collapse=","), paste(st$down_sent, collapse=","), paste(st$gen, collapse=","),
        paste(vapply(seq_len(n), function(i) as.numeric(res[[hosts[i]]]$g %||% NA), 0), collapse=","),
        paste(vapply(seq_len(n), function(i){u<-res[[hosts[i]]]$ud; if(is.character(u)&&nzchar(u)) nchar(u) else 0L}, 0L), collapse=",")),
        file="/tmp/pump_debug.log", append=TRUE)

  # 3) apply per-node results
  for (i in seq_len(n)) {
    r <- res[[hosts[i]]]; if (is.null(r)) next
    # reconnect: the forwarder bumped the generation (the SuperNode dropped and
    # redialed). Reset this node's SuperLink socket + offsets so the new stream
    # maps to a fresh SuperLink connection; the truncated spool means stale data
    # is harmless.
    g <- suppressWarnings(as.numeric(r$g)); if (is.na(g)) g <- st$gen[i]
    if (g != st$gen[i]) {
      if (!is.null(st$socks[[i]])) tryCatch(close(st$socks[[i]]), error = function(e) NULL)
      st$socks[i] <- list(NULL)   # [i]<-list(NULL) keeps the slot; [[i]]<-NULL would drop it
      st$up_off[i] <- 0; st$down_sent[i] <- 0; st$down_buf[[i]] <- raw(0)
      st$gen[i] <- g
      next
    }
    # down ack: node down.bin advanced -> drop the acked prefix of the buffer
    sz <- suppressWarnings(as.numeric(r$sz)); if (is.na(sz)) sz <- st$down_sent[i]
    acked <- sz - st$down_sent[i]
    if (acked > 0) {
      buf <- st$down_buf[[i]] %||% raw(0)
      st$down_buf[[i]] <- if (acked >= length(buf)) raw(0) else buf[(acked + 1):length(buf)]
      st$down_sent[i] <- sz
    }
    # up: write node->SuperLink bytes to the (lazily opened) SuperLink socket
    ud <- r$ud
    if (is.character(ud) && nzchar(ud)) {
      if (is.null(st$socks[[i]])) {
        newsock <- tryCatch(
          socketConnection("127.0.0.1", fp, open = "r+b", blocking = FALSE, timeout = 10),
          error = function(e) NULL)
        st$socks[i] <- list(newsock)   # keep the slot even if the open failed (NULL)
      }
      s <- st$socks[[i]]
      if (!is.null(s)) {
        ok <- tryCatch({ writeBin(.tunnel_dec_client(ud), s); flush(s); TRUE },
                       error = function(e) FALSE)
        if (ok) { ue <- suppressWarnings(as.numeric(r$ue)); if (!is.na(ue)) st$up_off[i] <- ue }
      }
    }
  }
  .dsflower_client_env$.tunnel <- st
  TRUE
}
