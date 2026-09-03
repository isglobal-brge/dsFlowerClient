# DSI tunnel relay pump (researcher side). Runs in the main R process, driven by
# the run wait loop: each call carries one batch of Flower Fleet-API bytes between
# the local SuperLink and the nodes' tunnel forwarders, entirely over DataSHIELD.
# A separate process is unnecessary because the SuperNodes connect to their
# forwarders only after nodes.ensure returns and retry until the pump starts.

#' @useDynLib dsFlowerClient, .registration = TRUE, .fixes = "C_"
NULL

#' @keywords internal
.tunnel_enc_client <- function(raw) {
  if (length(raw) == 0) return("")
  b <- gsub("[\r\n]", "", jsonlite::base64_enc(raw))
  b <- gsub("\\+", "-", b); b <- gsub("/", "_", b); b <- gsub("=+$", "", b)
  paste0("B64:", b)
}

#' @keywords internal
.tunnel_dec_client <- function(s, max_bytes = 8 * 1024^2) {
  if (!is.character(s) || length(s) != 1 || is.na(s) ||
      !nzchar(s) || !startsWith(s, "B64:")) {
    return(raw(0))
  }
  b <- substring(s, first = 5L, last = nchar(s, type = "chars"))
  if (nchar(b, type = "bytes") > 4 * ceiling(max_bytes / 3) + 4) {
    stop("Tunnel response exceeds the negotiated chunk size.", call. = FALSE)
  }
  b <- gsub("-", "+", b); b <- gsub("_", "/", b)
  pad <- (4 - nchar(b) %% 4) %% 4; if (pad > 0) b <- paste0(b, strrep("=", pad))
  value <- tryCatch(jsonlite::base64_dec(b), error = function(e) NULL)
  if (is.null(value) || length(value) > max_bytes) {
    stop("Invalid or oversized tunnel response.", call. = FALSE)
  }
  value
}

#' Write once to a non-blocking socket and return the bytes accepted by R
#'
#' Base R's writeBin() intentionally does not report whether a non-blocking
#' output write succeeded. The small native shim exposes R_WriteConnection's
#' byte count so the relay never acknowledges bytes that remain unwritten. It
#' is compile-time pinned to R's connection ABI version 1 and fails installation
#' explicitly if a future R release changes that private ABI.
#' @keywords internal
.tunnel_socket_write_some <- function(socket, payload) {
  if (!is.raw(payload)) {
    stop("Tunnel socket payload must be raw bytes.", call. = FALSE)
  }
  if (length(payload) == 0L) return(0)
  written <- .Call(C_dsf_socket_write, socket, payload)
  written <- suppressWarnings(as.numeric(written))
  if (length(written) != 1L || is.na(written) || !is.finite(written) ||
      written < 0 || written > length(payload) || written != floor(written)) {
    stop("Invalid tunnel socket write count.", call. = FALSE)
  }
  written
}

#' Read currently available bytes without hiding a closed/broken socket
#' @keywords internal
.tunnel_socket_read_available <- function(socket, max_bytes) {
  ready <- socketSelect(list(socket), write = FALSE, timeout = 0)
  if (length(ready) != 1L || !isTRUE(ready[[1]])) return(raw(0))
  value <- readBin(socket, "raw", max_bytes)
  if (length(value) == 0L && !isTRUE(isIncomplete(socket))) {
    stop("Local SuperLink tunnel socket closed.", call. = FALSE)
  }
  value
}

#' Carry one batch of tunnel bytes (returns TRUE if the tunnel is active)
#'
#' One fan-out flowerTunnelExchangeDS per cycle. The relay OWNS the byte offsets
#' (up_off = acked node->SuperLink bytes, down_sent = acked SuperLink->node
#' bytes), so a transient DSI failure loses nothing: the next cycle re-requests
#' the same ranges and the node applies them idempotently. SuperLink replies are
#' buffered per node and only dropped once the node acks them, and the up offset
#' only advances after the bytes are confirmed written to the SuperLink socket.
#' Negotiated chunks and bounded per-node buffers apply TCP backpressure instead
#' of accumulating unbounded payloads in the R process.
#' A single asynchronous fan-out call (not sequential per-node pushes) lets DSI
#' service the sites concurrently. Its named expression list carries only the
#' node-local payload to each DataSHIELD connection.
#' @keywords internal
.tunnel_pump <- function() {
  st <- .dsflower_client_env$.tunnel
  if (is.null(st) || !isTRUE(st$active)) return(FALSE)
  conns <- st$conns; hosts <- st$hosts; cid <- st$conn_id; fp <- st$fleet_port
  n <- length(hosts)
  if (is.null(st$chunk_bytes)) {
    st$chunk_bytes <- rep(.dsi_max_raw_chunk_bytes, n)
  }
  if (length(st$chunk_bytes) != n) {
    stop("Invalid tunnel chunk negotiation state.", call. = FALSE)
  }
  chunks <- vapply(st$chunk_bytes, .client_tunnel_chunk_bytes, integer(1))
  buffer_caps <- 8 * as.numeric(chunks)
  if (is.null(st$up_off))    st$up_off    <- rep(0, n)            # acked node->SL
  if (is.null(st$down_sent)) st$down_sent <- rep(0, n)           # acked SL->node
  if (is.null(st$down_buf))  st$down_buf  <- vector("list", n)   # SL bytes pending node ack
  if (is.null(st$down_inflight_len)) {
    st$down_inflight_len <- rep(0, n)                             # frozen message prefix
  }
  if (is.null(st$gen))       st$gen       <- rep(0, n)           # forwarder connection gen
  if (length(st$down_inflight_len) != n) {
    stop("Invalid tunnel in-flight message state.", call. = FALSE)
  }
  # Socket reads/writes are irreversible. Preserve every buffer and offset
  # change even if a later node in this fan-out returns a malformed ACK or a
  # local socket operation fails.
  on.exit(.dsflower_client_env$.tunnel <- st, add = TRUE)

  # 1) drain available SuperLink->node bytes from each open socket into its buffer
  for (i in seq_len(n)) {
    s <- st$socks[[i]]; if (is.null(s)) next
    buffered <- length(st$down_buf[[i]] %||% raw(0))
    capacity <- buffer_caps[i] - buffered
    if (capacity <= 0) next
    d <- .tunnel_socket_read_available(s, min(chunks[i], capacity))
    if (length(d) > 0) st$down_buf[[i]] <- c(st$down_buf[[i]] %||% raw(0), d)
  }

  # 2) one fan-out exchange: push pending down bytes (idempotent at down_sent) and
  #    poll up bytes from up_off, for every node at once
  exprs <- stats::setNames(vector("list", n), hosts)
  sent_lengths <- integer(n)
  for (i in seq_len(n)) {
    buf <- st$down_buf[[i]] %||% raw(0)
    inflight <- suppressWarnings(as.numeric(st$down_inflight_len[i]))
    if (length(inflight) != 1L || is.na(inflight) || !is.finite(inflight) ||
        inflight < 0 || inflight > chunks[i] || inflight != floor(inflight) ||
        inflight > length(buf)) {
      stop("Invalid tunnel in-flight message state.", call. = FALSE)
    }
    if (inflight == 0 && length(buf) > 0L) {
      inflight <- min(length(buf), chunks[i])
      st$down_inflight_len[i] <- inflight
    }
    send <- if (inflight > 0) buf[seq_len(inflight)] else raw(0)
    sent_lengths[i] <- as.integer(inflight)
    exprs[[hosts[i]]] <- call(
      "flowerTunnelExchangeDS",
      conn_id = cid,
      pa = st$down_sent[i],
      # Opal's expression parser rejects an explicit empty string in this slot;
      # NULL is parser-safe and the server normalizes it to an empty payload.
      pd = if (length(send)) .tunnel_enc_client(send) else NULL,
      pf = st$up_off[i],
      g = st$gen[i]
    )
  }
  raw_res <- tryCatch(
    .dsi_private_aggregate(conns, exprs),
    error = function(e) NULL
  )
  res <- .dsi_exact_node_results(raw_res, conns)
  if (is.null(res)) {
    # Preserve bytes already drained from the SuperLink socket; the next cycle
    # retries the same unacknowledged prefix.
    return(TRUE)
  }
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
    if (!is.list(r) ||
        !identical(names(r), c("ok", "node", "sz", "ud", "ue", "g")) ||
        !isTRUE(r$ok) || !identical(as.character(r$node), hosts[i])) {
      stop("Node returned an invalid or misassociated tunnel ACK.",
           call. = FALSE)
    }
    # reconnect: the forwarder bumped the generation (the SuperNode dropped and
    # redialed). Reset this node's SuperLink socket + offsets so the new stream
    # maps to a fresh SuperLink connection; the truncated spool means stale data
    # is harmless.
    g <- suppressWarnings(as.numeric(r$g))
    if (length(g) != 1L || is.na(g) || !is.finite(g) ||
        g < 0 || g != floor(g)) {
      stop("Node returned an invalid tunnel generation.", call. = FALSE)
    }
    if (g != st$gen[i]) {
      if (!is.null(st$socks[[i]])) tryCatch(close(st$socks[[i]]), error = function(e) NULL)
      st$socks[i] <- list(NULL)   # [i]<-list(NULL) keeps the slot; [[i]]<-NULL would drop it
      st$up_off[i] <- 0; st$down_sent[i] <- 0; st$down_buf[[i]] <- raw(0)
      st$down_inflight_len[i] <- 0
      st$gen[i] <- g
      next
    }
    # down ack: node down.bin advanced -> drop the acked prefix of the buffer
    sz <- suppressWarnings(as.numeric(r$sz))
    if (length(sz) != 1L || is.na(sz) || !is.finite(sz)) {
      stop("Node returned an invalid tunnel acknowledgment.", call. = FALSE)
    }
    acked <- sz - st$down_sent[i]
    if (!is.finite(acked) || acked < 0 || acked != floor(acked) ||
        !acked %in% c(0, sent_lengths[i])) {
      stop("Node returned an invalid tunnel acknowledgment.", call. = FALSE)
    }
    if (acked > 0) {
      buf <- st$down_buf[[i]] %||% raw(0)
      st$down_buf[[i]] <- if (acked >= length(buf)) raw(0) else buf[(acked + 1):length(buf)]
      st$down_sent[i] <- sz
      st$down_inflight_len[i] <- 0
    }
    # up: write node->SuperLink bytes to the (lazily opened) SuperLink socket
    ud <- r$ud
    if (!is.character(ud) || length(ud) != 1L || is.na(ud)) {
      stop("Node returned an invalid tunnel payload.", call. = FALSE)
    }
    if (nzchar(ud)) {
      payload <- .tunnel_dec_client(ud, max_bytes = chunks[i])
      ue <- suppressWarnings(as.numeric(r$ue))
      expected_ue <- st$up_off[i] + length(payload)
      if (length(ue) != 1L || is.na(ue) || !is.finite(ue) ||
          ue != expected_ue) {
        stop("Node returned an invalid tunnel offset.", call. = FALSE)
      }
      if (is.null(st$socks[[i]])) {
        newsock <- tryCatch(
          socketConnection("127.0.0.1", fp, open = "r+b", blocking = FALSE, timeout = 10),
          error = function(e) NULL)
        st$socks[i] <- list(newsock)   # keep the slot even if the open failed (NULL)
      }
      s <- st$socks[[i]]
      if (!is.null(s)) {
        written <- .tunnel_socket_write_some(s, payload)
        # A short non-blocking write is normal under backpressure. Ask the node
        # for the unwritten suffix on the next exchange; never infer a full ACK.
        if (written > 0) st$up_off[i] <- st$up_off[i] + written
      }
    } else {
      ue <- suppressWarnings(as.numeric(r$ue))
      if (length(ue) != 1L || is.na(ue) || !is.finite(ue) ||
          ue != st$up_off[i]) {
        stop("Node returned an invalid tunnel offset.", call. = FALSE)
      }
    }
  }
  TRUE
}
