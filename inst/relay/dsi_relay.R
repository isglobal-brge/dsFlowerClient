# Background DSI relay: shuttle Flower Fleet-API bytes between a local SuperLink
# and N data nodes' tunnel forwarders, entirely over DataSHIELD. Runs as its own
# process (its own DSI login) so it can carry bytes while ds.flower.fit submits
# and polls the run in the main session. Stops when the stop-file appears.
#
# Invoked as: Rscript dsi_relay.R <config.json>
# config.json: { libpath, fleet_port, conn_id, stop_file, log,
#                nodes:[{server,url,user,password}], ssl_opts }

args <- commandArgs(trailingOnly = TRUE)
cfg <- jsonlite::fromJSON(args[1], simplifyVector = FALSE)
.libPaths(c(cfg$libpath, .libPaths()))
suppressMessages({library(DSI); library(DSOpal); library(jsonlite)})

logf <- function(...) cat(sprintf("[%s] ", format(Sys.time(), "%H:%M:%S")), ..., "\n",
                          file = cfg$log, append = TRUE)

enc <- function(raw) {
  if (length(raw) == 0) return("")
  b <- gsub("[\r\n]", "", base64_enc(raw))
  b <- gsub("\\+", "-", b); b <- gsub("/", "_", b); b <- gsub("=+$", "", b)
  paste0("B64:", b)
}
dec <- function(s) {
  if (!is.character(s) || !nzchar(s)) return(raw(0))
  b <- substring(s, 5); b <- gsub("-", "+", b); b <- gsub("_", "/", b)
  pad <- (4 - nchar(b) %% 4) %% 4; if (pad > 0) b <- paste0(b, strrep("=", pad))
  base64_dec(b)
}

builder <- DSI::newDSLoginBuilder()
hosts <- vapply(cfg$nodes, function(n) n$server, character(1))
for (n in cfg$nodes) {
  builder$append(server = n$server, url = n$url, user = n$user, password = n$password,
                 driver = "OpalDriver", options = cfg$ssl_opts)
}
conns <- DSI::datashield.login(builder$build())
logf("relay logged in to", length(hosts), "nodes; fleet port", cfg$fleet_port)

cid <- cfg$conn_id
fleet_port <- as.integer(cfg$fleet_port)

# one TCP connection to the local SuperLink Fleet API per node (each SuperNode is
# a distinct gRPC client to the SuperLink)
sk <- vector("list", length(hosts))
open_sock <- function(i) {
  tryCatch(socketConnection("127.0.0.1", fleet_port, open = "r+b",
                            blocking = FALSE, timeout = 10),
           error = function(e) NULL)
}

repeat {
  if (file.exists(cfg$stop_file)) break
  # down: SuperLink -> each node
  for (i in seq_along(hosts)) {
    if (is.null(sk[[i]])) { sk[[i]] <- open_sock(i); if (is.null(sk[[i]])) next }
    down <- tryCatch(readBin(sk[[i]], "raw", 1e6), error = function(e) raw(0))
    if (length(down) > 0) {
      tryCatch(DSI::datashield.aggregate(conns[hosts[i]],
        call("flowerTunnelPushDS", cid, enc(down))), error = function(e) NULL)
    }
  }
  # up: poll all nodes in one fan-out, write each to its SuperLink socket
  ups <- tryCatch(DSI::datashield.aggregate(conns, call("flowerTunnelPollDS", cid)),
                  error = function(e) NULL)
  if (!is.null(ups)) {
    for (i in seq_along(hosts)) {
      u <- ups[[hosts[i]]]
      if (is.character(u) && nzchar(u) && !is.null(sk[[i]])) {
        tryCatch({ writeBin(dec(u), sk[[i]]); flush(sk[[i]]) }, error = function(e) NULL)
      }
    }
  }
  Sys.sleep(0.03)
}

for (s in sk) if (!is.null(s)) tryCatch(close(s), error = function(e) NULL)
tryCatch(DSI::datashield.logout(conns), error = function(e) NULL)
logf("relay stopped")
