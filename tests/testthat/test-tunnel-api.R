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

test_that("link-up aborts and rolls back every attempted node on partial startup", {
  conns <- list(site1 = structure(list(), class = "mock_connection"),
                site2 = structure(list(), class = "mock_connection"))
  events <- character()
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
        ok <- identical(server, "site1")
        return(setNames(list(list(ok = ok, chunk_bytes = 1024^2)), server))
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
})

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
    datashield.aggregate = function(conns, call) {
      captured <<- call
      list(site1 = list(sz = 0, ud = "", ue = 0, g = 0))
    },
    .package = "DSI"
  )

  expect_true(dsFlowerClient:::.tunnel_pump())
  encoded_request <- captured[[3]]
  b64 <- substring(encoded_request, 5)
  b64 <- gsub("-", "+", b64)
  b64 <- gsub("_", "/", b64)
  pad <- (4 - nchar(b64) %% 4) %% 4
  if (pad > 0) b64 <- paste0(b64, strrep("=", pad))
  request <- jsonlite::fromJSON(
    rawToChar(jsonlite::base64_dec(b64)), simplifyVector = FALSE
  )
  payload <- dsFlowerClient:::.tunnel_dec_client(
    request$site1$pd, max_bytes = chunk
  )
  expect_length(payload, chunk)
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
  local_mocked_bindings(
    datashield.aggregate = function(conns, call) stop("transient"),
    .package = "DSI"
  )

  expect_true(dsFlowerClient:::.tunnel_pump())
  expect_identical(client_env$.tunnel$up_off, 0)
  expect_identical(client_env$.tunnel$down_sent, 0)
  expect_identical(client_env$.tunnel$down_buf, list(NULL))
  expect_identical(client_env$.tunnel$gen, 0)
})
