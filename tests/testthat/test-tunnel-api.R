test_that("the client uses only the production tunnel API", {
  code <- paste(
    deparse(body(dsFlowerClient::ds.flower.link.up)),
    deparse(body(dsFlowerClient::ds.flower.link.down)),
    deparse(body(dsFlowerClient:::.tunnel_pump)),
    collapse = "\n"
  )
  expect_match(code, "flowerTunnelUpDS")
  expect_match(code, "flowerTunnelExchangeDS")
  expect_match(code, "flowerTunnelDownDS")

  removed <- c(
    "flowerTunnelResetDS", "flowerTunnelPollDS", "flowerTunnelPushDS",
    "flowerTunnelInjectDS", "flowerTunnelDrainDS", "flowerTunnelReapDS",
    "flowerTunnelTestClientDS", "flowerTunnelTestResultDS",
    "flowerTunnelSupernodeDS", "flowerTunnelLogDS"
  )
  expect_false(any(vapply(removed, grepl, logical(1), x = code, fixed = TRUE)))
})

test_that("tunnel session ids use 128 bits from Python secrets", {
  first <- dsFlowerClient:::.new_tunnel_conn_id()
  second <- dsFlowerClient:::.new_tunnel_conn_id()
  expect_match(first, "^dsf_[0-9a-f]{32}$")
  expect_match(second, "^dsf_[0-9a-f]{32}$")
  expect_false(identical(first, second))
})

test_that("the client rejects invalid tunnel ports before link side effects", {
  expect_identical(dsFlowerClient:::.client_tunnel_port(1), 1L)
  expect_identical(dsFlowerClient:::.client_tunnel_port("65535"), 65535L)
  for (bad in list(0, 65536, 1.5, NA_real_, Inf, "bad", c(1, 2))) {
    expect_error(
      dsFlowerClient:::.client_tunnel_port(bad),
      "Invalid dsflower.tunnel_port"
    )
  }
  touched_superlink <- FALSE
  withr::local_options(list(dsflower.tunnel_port = 0))
  local_mocked_bindings(
    ds.flower.superlink.status = function() {
      touched_superlink <<- TRUE
      list(running = TRUE, ports = list(fleet = 9092L))
    },
    .package = "dsFlowerClient"
  )
  expect_error(
    dsFlowerClient::ds.flower.link.up(list(site1 = list())),
    "Invalid dsflower.tunnel_port"
  )
  expect_false(touched_superlink)
})

test_that("tunnel ports can be scalar, positional, or named per node", {
  hosts <- c("site1", "site2")
  expect_identical(
    dsFlowerClient:::.client_tunnel_ports(18080L, hosts),
    c(site1 = 18080L, site2 = 18080L)
  )
  expect_identical(
    dsFlowerClient:::.client_tunnel_ports(c(18081L, 18082L), hosts),
    c(site1 = 18081L, site2 = 18082L)
  )
  expect_identical(
    dsFlowerClient:::.client_tunnel_ports(
      c(site2 = 18082L, site1 = 18081L), hosts
    ),
    c(site1 = 18081L, site2 = 18082L)
  )
  expect_error(
    dsFlowerClient:::.client_tunnel_ports(c(18081L, 18082L, 18083L), hosts),
    "one port per node"
  )
  expect_error(
    dsFlowerClient:::.client_tunnel_ports(
      c(site1 = 18081L, wrong = 18082L), hosts
    ),
    "must match the node names"
  )
})

test_that("link-up aborts and rolls back every attempted node on mixed tunnel ABI", {
  conns <- list(site1 = structure(list(), class = "mock_connection"),
                site2 = structure(list(), class = "mock_connection"))
  withr::local_options(list(
    dsflower.dsi_tls_attested = c("site1", "site2"),
    dsflower.tunnel_port = c(site2 = 18082L, site1 = 18081L)
  ))
  events <- character()
  up_ports <- integer()
  running <- FALSE
  stopped <- FALSE
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_tunnel <- client_env$.tunnel
  old_insecure <- getOption("dsflower.superlink_insecure", NULL)
  withr::defer({
    client_env$.tunnel <- old_tunnel
    options(dsflower.superlink_insecure = old_insecure)
  })
  client_env$.tunnel <- NULL
  options(dsflower.superlink_insecure = "previous")

  local_mocked_bindings(
    ds.flower.superlink.status = function() {
      list(running = running, ports = list(fleet = 9092L))
    },
    ds.flower.superlink.start = function(insecure = FALSE) {
      running <<- TRUE
      TRUE
    },
    ds.flower.superlink.stop = function() {
      stopped <<- TRUE
      running <<- FALSE
      TRUE
    },
    .new_tunnel_conn_id = function() paste0("dsf_", strrep("a", 32)),
    .dsf_msg = function(...) invisible(NULL),
    .package = "dsFlowerClient"
  )
  local_mocked_bindings(
    datashield.aggregate = function(conns, call) {
      method <- as.character(call[[1]])
      server <- names(conns)
      events <<- c(events, paste(method, server, sep = ":"))
      if (identical(method, "flowerTunnelUpDS")) {
        expect_identical(call$protocol_abi, 4L)
        up_ports[[server]] <<- as.integer(call[[3]])
        return(setNames(list(list(
          ok = TRUE, listen = paste0("127.0.0.1:", call[[3]]),
          chunk_bytes = 512L * 1024L,
          protocol_abi = if (identical(server, "site1")) 4L else 3L
        )), server))
      }
      setNames(list(TRUE), server)
    },
    .package = "DSI"
  )

  expect_error(
    dsFlowerClient::ds.flower.link.up(conns),
    "site2: tunnel forwarder did not report ready"
  )
  expect_null(client_env$.tunnel)
  expect_identical(getOption("dsflower.superlink_insecure"), "previous")
  expect_true(stopped)
  expect_true(all(c(
    "flowerTunnelDownDS:site1", "flowerTunnelDownDS:site2"
  ) %in% events))
  expect_identical(up_ports[c("site1", "site2")], c(site1 = 18081L, site2 = 18082L))
})

.test_direct_tunnel_request <- function(expr) {
  fields <- c("pa", "pd", "pf", "g")
  stats::setNames(lapply(fields, function(field) expr[[field]]), fields)
}

test_that("the relay sends at most the node-negotiated chunk", {
  chunk <- 16 * 1024L
  captured <- NULL
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_tunnel <- client_env$.tunnel
  withr::defer(client_env$.tunnel <- old_tunnel)
  client_env$.tunnel <- list(
    active = TRUE,
    conns = list(site1 = structure(list(), class = "mock_connection")),
    hosts = "site1",
    conn_id = paste0("dsf_", strrep("b", 32)),
    fleet_port = 9092L,
    chunk_bytes = chunk,
    socks = list(NULL),
    up_off = 0,
    down_sent = 0,
    down_buf = list(as.raw(rep(1, chunk + 100L))),
    gen = 0
  )
  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      captured <<- expr
      list(site1 = list(
        ok = TRUE, node = "site1", sz = 0, ud = "", ue = 0, g = 0))
    },
    .package = "DSI"
  )

  expect_true(dsFlowerClient:::.tunnel_pump())
  expect_named(captured, "site1")
  expect_identical(as.character(captured$site1[[1]]), "flowerTunnelExchangeDS")
  request <- .test_direct_tunnel_request(captured$site1)
  payload <- dsFlowerClient:::.tunnel_dec_client(
    request$pd, max_bytes = chunk
  )
  expect_length(payload, chunk)
  expect_equal(request$g, 0)
  expect_length(
    client_env$.tunnel$down_buf[[1]],
    chunk + 100L
  )
})

test_that("a transient DSI exchange preserves initialized relay state", {
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_tunnel <- client_env$.tunnel
  withr::defer(client_env$.tunnel <- old_tunnel)
  client_env$.tunnel <- list(
    active = TRUE,
    conns = list(site1 = structure(list(), class = "mock_connection")),
    hosts = "site1",
    conn_id = paste0("dsf_", strrep("c", 32)),
    fleet_port = 9092L,
    chunk_bytes = 16 * 1024L,
    socks = list(NULL)
  )
  captured <- NULL
  local_mocked_bindings(
    datashield.aggregate = function(conns, call) {
      captured <<- call
      stop("transient")
    },
    .package = "DSI"
  )

  expect_true(dsFlowerClient:::.tunnel_pump())
  expect_identical(client_env$.tunnel$up_off, 0)
  expect_identical(client_env$.tunnel$down_sent, 0)
  expect_identical(client_env$.tunnel$down_buf, list(NULL))
  expect_identical(client_env$.tunnel$gen, 0)
  expect_null(captured$site1$pd)
})

test_that("a DSI-dropped node retries the identical tunnel chunk and offset", {
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_tunnel <- client_env$.tunnel
  withr::defer(client_env$.tunnel <- old_tunnel)
  payload <- charToRaw("lost-ack")
  later <- charToRaw("-later")
  fake_socket <- structure(list(), class = "test_tunnel_socket")
  client_env$.tunnel <- list(
    active = TRUE,
    conns = list(site1 = structure(list(), class = "mock_connection")),
    hosts = "site1",
    conn_id = paste0("dsf_", strrep("8", 32)),
    fleet_port = 9092L,
    chunk_bytes = 16 * 1024L,
    socks = list(fake_socket),
    up_off = 0,
    down_sent = 0,
    down_buf = list(payload),
    gen = 1
  )
  calls <- list()
  reads <- 0L
  local_mocked_bindings(
    .tunnel_socket_read_available = function(socket, max_bytes) {
      reads <<- reads + 1L
      if (reads == 2L) later else raw(0)
    },
    .package = "dsFlowerClient"
  )
  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      calls[[length(calls) + 1L]] <<- expr
      if (length(calls) == 1L) return(list())
      list(site1 = list(
        ok = TRUE, node = "site1",
        sz = if (length(calls) == 2L) length(payload) else
          length(payload) + length(later),
        ud = "", ue = 0, g = 1))
    },
    .package = "DSI"
  )

  expect_true(dsFlowerClient:::.tunnel_pump())
  expect_identical(client_env$.tunnel$down_sent, 0)
  expect_identical(client_env$.tunnel$down_buf[[1L]], payload)
  expect_true(dsFlowerClient:::.tunnel_pump())
  expect_identical(calls[[2L]], calls[[1L]])
  expect_identical(client_env$.tunnel$down_sent, as.numeric(length(payload)))
  expect_identical(client_env$.tunnel$down_buf[[1L]], later)
  expect_identical(client_env$.tunnel$down_inflight_len, 0)

  expect_true(dsFlowerClient:::.tunnel_pump())
  third <- .test_direct_tunnel_request(calls[[3L]]$site1)
  expect_equal(third$pa, length(payload))
  expect_identical(
    dsFlowerClient:::.tunnel_dec_client(third$pd, 16 * 1024L), later
  )
  expect_identical(
    client_env$.tunnel$down_sent,
    as.numeric(length(payload) + length(later))
  )
  expect_identical(client_env$.tunnel$down_buf[[1L]], raw(0))
})

test_that("late multi-node failures preserve every irreversible relay change", {
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_tunnel <- client_env$.tunnel
  withr::defer(client_env$.tunnel <- old_tunnel)
  conns <- list(
    site1 = structure(list(), class = "mock_connection"),
    site2 = structure(list(), class = "mock_connection")
  )
  sockets <- list(
    structure(list(site = "site1"), class = "test_tunnel_socket"),
    structure(list(site = "site2"), class = "test_tunnel_socket")
  )
  base_state <- list(
    active = TRUE, conns = conns, hosts = names(conns),
    conn_id = paste0("dsf_", strrep("7", 32)), fleet_port = 9092L,
    chunk_bytes = rep(16 * 1024L, 2L), socks = sockets,
    up_off = c(0, 0), down_sent = c(0, 0),
    down_buf = list(raw(0), raw(0)), gen = c(1, 1)
  )
  first <- charToRaw("a")
  second <- charToRaw("b")
  client_env$.tunnel <- base_state
  local_mocked_bindings(
    .tunnel_socket_read_available = function(socket, max_bytes) {
      if (identical(socket$site, "site1")) first else stop("site2 read failed")
    },
    .package = "dsFlowerClient"
  )
  expect_error(dsFlowerClient:::.tunnel_pump(), "site2 read failed")
  expect_identical(client_env$.tunnel$down_buf, list(first, raw(0)))

  client_env$.tunnel <- within(base_state, {
    socks <- list(NULL, NULL)
    down_buf <- list(first, second)
  })
  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) list(
      site1 = list(
        ok = TRUE, node = "site1", sz = 1, ud = "", ue = 0, g = 1),
      site2 = list(ok = TRUE)
    ),
    .package = "DSI"
  )
  expect_error(
    dsFlowerClient:::.tunnel_pump(),
    "invalid or misassociated tunnel ACK"
  )
  expect_identical(client_env$.tunnel$down_sent, c(1, 0))
  expect_identical(client_env$.tunnel$down_buf, list(raw(0), second))
  expect_identical(client_env$.tunnel$down_inflight_len, c(0, 1))
})

.decode_test_tunnel_arg <- function(encoded) {
  b64 <- substring(
    encoded, first = 5L, last = nchar(encoded, type = "chars")
  )
  b64 <- gsub("-", "+", b64)
  b64 <- gsub("_", "/", b64)
  pad <- (4 - nchar(b64) %% 4) %% 4
  if (pad > 0) b64 <- paste0(b64, strrep("=", pad))
  jsonlite::fromJSON(
    rawToChar(jsonlite::base64_dec(b64)), simplifyVector = FALSE
  )
}

test_that("large tunnel and outer request payloads are not truncated", {
  payload <- as.raw((seq_len(2 * 1024^2 + 37L) - 1L) %% 256L)
  encoded <- dsFlowerClient:::.tunnel_enc_client(payload)
  expect_gt(nchar(encoded, type = "bytes"), 1000000L)
  expect_identical(
    dsFlowerClient:::.tunnel_dec_client(encoded, max_bytes = 3 * 1024^2),
    payload
  )

  outer <- dsFlowerClient:::.ds_encode(list(
    pa = 0, pd = encoded, pf = 0, g = 3
  ))
  expect_gt(nchar(outer, type = "bytes"), 1000000L)
  expect_identical(.decode_test_tunnel_arg(outer)$pd, encoded)
})

test_that("one async fan-out sends only each node's local request", {
  captured <- NULL
  calls <- 0L
  chunk <- 16 * 1024L
  first <- as.raw(rep(0x11, 500L))
  second <- as.raw(rep(0xee, 700L))
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_tunnel <- client_env$.tunnel
  withr::defer(client_env$.tunnel <- old_tunnel)
  client_env$.tunnel <- list(
    active = TRUE,
    conns = list(
      site1 = structure(list(), class = "mock_connection"),
      site2 = structure(list(), class = "mock_connection")
    ),
    hosts = c("site1", "site2"),
    conn_id = paste0("dsf_", strrep("d", 32)),
    fleet_port = 9092L,
    chunk_bytes = c(chunk, chunk),
    socks = list(NULL, NULL),
    up_off = c(0, 0),
    down_sent = c(0, 0),
    down_buf = list(first, second),
    gen = c(4, 9)
  )
  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      calls <<- calls + 1L
      captured <<- expr
      list(
        site1 = list(
          ok = TRUE, node = "site1", sz = 0, ud = "", ue = 0, g = 4),
        site2 = list(
          ok = TRUE, node = "site2", sz = 0, ud = "", ue = 0, g = 9)
      )
    },
    .package = "DSI"
  )

  expect_true(dsFlowerClient:::.tunnel_pump())
  expect_identical(calls, 1L)
  expect_named(captured, c("site1", "site2"))
  requests <- lapply(captured, function(expr) {
    expect_identical(as.character(expr[[1]]), "flowerTunnelExchangeDS")
    .test_direct_tunnel_request(expr)
  })
  expect_named(requests$site1, c("pa", "pd", "pf", "g"))
  expect_named(requests$site2, c("pa", "pd", "pf", "g"))
  expect_identical(
    dsFlowerClient:::.tunnel_dec_client(requests$site1$pd, chunk), first
  )
  expect_identical(
    dsFlowerClient:::.tunnel_dec_client(requests$site2$pd, chunk), second
  )
  expect_equal(requests$site1$g, 4)
  expect_equal(requests$site2$g, 9)
})

test_that("a generation change discards every stale acknowledgment and payload", {
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_tunnel <- client_env$.tunnel
  withr::defer(client_env$.tunnel <- old_tunnel)
  fake_socket <- structure(list(), class = "test_tunnel_socket")
  client_env$.tunnel <- list(
    active = TRUE,
    conns = list(site1 = structure(list(), class = "mock_connection")),
    hosts = "site1",
    conn_id = paste0("dsf_", strrep("3", 32)),
    fleet_port = 9092L,
    chunk_bytes = 16 * 1024L,
    socks = list(fake_socket),
    up_off = 23,
    down_sent = 17,
    down_buf = list(charToRaw("pending")),
    gen = 1
  )
  sent_generation <- NULL
  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      sent_generation <<- .test_direct_tunnel_request(expr$site1)$g
      list(site1 = list(
        ok = TRUE,
        node = "site1",
        sz = 24,
        ud = dsFlowerClient:::.tunnel_enc_client(charToRaw("stale")),
        ue = 28,
        g = 2
      ))
    },
    .package = "DSI"
  )
  wrote <- FALSE
  local_mocked_bindings(
    .tunnel_socket_read_available = function(socket, max_bytes) raw(0),
    .tunnel_socket_write_some = function(socket, bytes) {
      wrote <<- TRUE
      length(bytes)
    },
    .package = "dsFlowerClient"
  )

  expect_true(dsFlowerClient:::.tunnel_pump())
  expect_equal(sent_generation, 1)
  expect_false(wrote)
  expect_null(client_env$.tunnel$socks[[1]])
  expect_identical(client_env$.tunnel$up_off, 0)
  expect_identical(client_env$.tunnel$down_sent, 0)
  expect_identical(client_env$.tunnel$down_buf, list(raw(0)))
  expect_identical(client_env$.tunnel$gen, 2)
})

test_that("partial non-blocking writes advance only the accepted prefix", {
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_tunnel <- client_env$.tunnel
  withr::defer(client_env$.tunnel <- old_tunnel)
  payload <- charToRaw("0123456789")
  fake_socket <- structure(list(), class = "test_tunnel_socket")
  client_env$.tunnel <- list(
    active = TRUE,
    conns = list(site1 = structure(list(), class = "mock_connection")),
    hosts = "site1",
    conn_id = paste0("dsf_", strrep("e", 32)),
    fleet_port = 9092L,
    chunk_bytes = 16 * 1024L,
    socks = list(fake_socket),
    up_off = 0,
    down_sent = 0,
    down_buf = list(raw(0)),
    gen = 1
  )
  polls <- numeric()
  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      request <- .test_direct_tunnel_request(expr$site1)
      polls <<- c(polls, request$pf)
      from <- as.integer(request$pf) + 1L
      remainder <- payload[from:length(payload)]
      list(site1 = list(
        ok = TRUE,
        node = "site1",
        sz = 0,
        ud = dsFlowerClient:::.tunnel_enc_client(remainder),
        ue = length(payload),
        g = 1
      ))
    },
    .package = "DSI"
  )
  writes <- 0L
  local_mocked_bindings(
    .tunnel_socket_read_available = function(socket, max_bytes) raw(0),
    .tunnel_socket_write_some = function(socket, bytes) {
      writes <<- writes + 1L
      if (writes == 1L) 3 else length(bytes)
    },
    .package = "dsFlowerClient"
  )

  expect_true(dsFlowerClient:::.tunnel_pump())
  expect_identical(client_env$.tunnel$up_off, 3)
  expect_true(dsFlowerClient:::.tunnel_pump())
  expect_identical(client_env$.tunnel$up_off, 10)
  expect_equal(polls, c(0, 3))
})

test_that("socket read and write failures are never treated as backpressure", {
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_tunnel <- client_env$.tunnel
  withr::defer(client_env$.tunnel <- old_tunnel)
  fake_socket <- structure(list(), class = "test_tunnel_socket")
  client_env$.tunnel <- list(
    active = TRUE,
    conns = list(site1 = structure(list(), class = "mock_connection")),
    hosts = "site1",
    conn_id = paste0("dsf_", strrep("2", 32)),
    fleet_port = 9092L,
    chunk_bytes = 16 * 1024L,
    socks = list(fake_socket),
    up_off = 0,
    down_sent = 0,
    down_buf = list(raw(0)),
    gen = 0
  )
  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      list(site1 = list(
        ok = TRUE,
        node = "site1",
        sz = 0,
        ud = dsFlowerClient:::.tunnel_enc_client(charToRaw("x")),
        ue = 1,
        g = 0
      ))
    },
    .package = "DSI"
  )
  fail_read <- TRUE
  local_mocked_bindings(
    .tunnel_socket_read_available = function(socket, max_bytes) {
      if (fail_read) stop("read failed")
      raw(0)
    },
    .tunnel_socket_write_some = function(socket, bytes) stop("write failed"),
    .package = "dsFlowerClient"
  )

  expect_error(dsFlowerClient:::.tunnel_pump(), "read failed")
  fail_read <- FALSE
  expect_error(dsFlowerClient:::.tunnel_pump(), "write failed")
  expect_identical(client_env$.tunnel$up_off, 0)
})

test_that("an indeterminate zero write closes the socket and aborts", {
  socket <- rawConnection(raw(0), open = "w+b")
  withr::defer(tryCatch(close(socket), error = function(e) NULL))
  local_mocked_bindings(
    .tunnel_socket_write_native = function(socket, payload) 0,
    .package = "dsFlowerClient"
  )

  expect_error(
    dsFlowerClient:::.tunnel_socket_write_some(socket, charToRaw("payload")),
    "progress is indeterminate; connection closed"
  )
  expect_error(isOpen(socket), "invalid connection")
})

test_that("native socket writes complete when the reader resumes before timeout", {
  python <- Sys.which("python3")
  if (!nzchar(python)) python <- Sys.which("python")
  skip_if(!nzchar(python), "Python is required for the socket integration test")
  server_code <- paste(
    "import hashlib,socket,time",
    "s=socket.socket()",
    "s.bind(('127.0.0.1',0))",
    "s.listen(1)",
    "print(s.getsockname()[1],flush=True)",
    "c,_=s.accept()",
    "time.sleep(0.3)",
    "h=hashlib.sha256()",
    "n=0",
    "while True:",
    " d=c.recv(65536)",
    " if not d: break",
    " h.update(d)",
    " n+=len(d)",
    "print(str(n)+' '+h.hexdigest(),flush=True)",
    "c.close()",
    "s.close()",
    sep = "\n"
  )
  proc <- processx::process$new(
    python, c("-u", "-c", server_code), stdout = "|", stderr = "|",
    cleanup = TRUE, cleanup_tree = TRUE
  )
  withr::defer(if (proc$is_alive()) proc$kill())
  deadline <- Sys.time() + 5
  lines <- character()
  while (length(lines) == 0L && Sys.time() < deadline) {
    lines <- c(lines, proc$read_output_lines())
    if (length(lines) == 0L) Sys.sleep(0.02)
  }
  expect_length(lines, 1L)
  port <- as.integer(lines[[1]])
  socket <- socketConnection(
    "127.0.0.1", port, open = "r+b", blocking = FALSE, timeout = 2
  )
  withr::defer(tryCatch(close(socket), error = function(e) NULL))

  payload <- as.raw((seq_len(8 * 1024^2) - 1L) %% 256L)
  offset <- 0L
  deadline <- Sys.time() + 10
  while (offset < length(payload) && Sys.time() < deadline) {
    remaining <- payload[(offset + 1L):length(payload)]
    written <- dsFlowerClient:::.tunnel_socket_write_some(socket, remaining)
    offset <- offset + as.integer(written)
  }
  expect_identical(offset, length(payload))
  # R's socket writer may wait until the peer drains its receive buffer. This
  # covers the successful path where it finishes before the socket timeout.
  close(socket)
  proc$wait(timeout = 5000)
  lines <- c(lines, proc$read_all_output_lines())
  expect_length(lines, 2L)
  received <- strsplit(lines[[2]], " ", fixed = TRUE)[[1]]
  expect_identical(as.integer(received[[1]]), length(payload))
  expect_identical(
    received[[2]], digest::digest(payload, algo = "sha256", serialize = FALSE)
  )
})

test_that("a stalled native socket write closes on indeterminate timeout", {
  python <- Sys.which("python3")
  if (!nzchar(python)) python <- Sys.which("python")
  skip_if(!nzchar(python), "Python is required for the socket integration test")
  server_code <- paste(
    "import socket,time",
    "s=socket.socket()",
    "s.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,4096)",
    "s.bind(('127.0.0.1',0))",
    "s.listen(1)",
    "print(s.getsockname()[1],flush=True)",
    "c,_=s.accept()",
    "time.sleep(4)",
    "n=0",
    "while True:",
    " d=c.recv(65536)",
    " if not d: break",
    " n+=len(d)",
    "print(n,flush=True)",
    "c.close()",
    "s.close()",
    sep = "\n"
  )
  proc <- processx::process$new(
    python, c("-u", "-c", server_code), stdout = "|", stderr = "|",
    cleanup = TRUE, cleanup_tree = TRUE
  )
  withr::defer(if (proc$is_alive()) proc$kill())
  deadline <- Sys.time() + 5
  lines <- character()
  while (length(lines) == 0L && Sys.time() < deadline) {
    lines <- c(lines, proc$read_output_lines())
    if (length(lines) == 0L) Sys.sleep(0.02)
  }
  expect_length(lines, 1L)
  socket <- socketConnection(
    "127.0.0.1", as.integer(lines[[1]]), open = "r+b",
    blocking = FALSE, timeout = 1
  )
  withr::defer(tryCatch(close(socket), error = function(e) NULL))

  payload <- as.raw((seq_len(512 * 1024L) - 1L) %% 256L)
  confirmed <- 0
  failure <- NULL
  for (attempt in seq_len(128L)) {
    result <- tryCatch(
      dsFlowerClient:::.tunnel_socket_write_some(socket, payload),
      error = identity
    )
    if (inherits(result, "error")) {
      failure <- result
      break
    }
    confirmed <- confirmed + result
  }
  expect_s3_class(failure, "error")
  expect_match(
    conditionMessage(failure),
    "progress is indeterminate; connection closed"
  )
  expect_error(isOpen(socket), "invalid connection")

  proc$wait(timeout = 10000)
  lines <- c(lines, proc$read_all_output_lines())
  expect_length(lines, 2L)
  received <- as.numeric(lines[[2]])
  # R may have delivered none or a prefix of the timed-out call. Either way,
  # closing the connection prevents that indeterminate payload from being retried.
  expect_true(received >= confirmed)
  expect_true(received <= confirmed + length(payload))
})
