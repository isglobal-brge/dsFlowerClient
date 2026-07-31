mock_dsi_connection <- function(class, url = NULL, tls_options = list()) {
  if (identical(class, "OpalConnection")) {
    return(structure(
      list(opal = list(url = url, config = list(options = tls_options))),
      class = class
    ))
  }
  if (identical(class, "ArmadilloConnection")) {
    return(structure(
      list(handle = structure(list(url = url), class = "handle")),
      class = class
    ))
  }
  if (identical(class, "DSLiteConnection")) {
    return(structure(list(), class = class))
  }
  structure(list(url = url), class = class)
}

test_that("recognized Opal and Armadillo HTTPS connections are plug-and-play", {
  withr::local_options(list(dsflower.dsi_tls_attested = character()))
  conns <- list(
    opal = mock_dsi_connection("OpalConnection", "https://opal.example.org")
  )

  expect_invisible(dsFlowerClient:::.validate_dsi_transport_security(conns))

  conns$opal <- mock_dsi_connection(
    "OpalConnection", "https://opal.example.org",
    list(ssl_verifypeer = TRUE, ssl_verifyhost = TRUE)
  )
  expect_invisible(dsFlowerClient:::.validate_dsi_transport_security(conns))

  armadillo <- list(
    armadillo = mock_dsi_connection(
      "ArmadilloConnection", "https://armadillo.example.org"
    )
  )
  expect_invisible(
    dsFlowerClient:::.validate_dsi_transport_security(armadillo)
  )

  armadillo$armadillo <- mock_dsi_connection(
    "ArmadilloConnection", "https://[::1]:8443"
  )
  expect_invisible(
    dsFlowerClient:::.validate_dsi_transport_security(armadillo)
  )

  armadillo$armadillo <- mock_dsi_connection("ArmadilloConnection")
  expect_error(
    dsFlowerClient:::.validate_dsi_transport_security(armadillo),
    "armadillo: DSI transport security cannot be inspected"
  )
  withr::local_options(list(dsflower.dsi_tls_attested = "armadillo"))
  expect_invisible(
    dsFlowerClient:::.validate_dsi_transport_security(armadillo)
  )

  incomplete_opal <- list(opal = structure(
    list(opal = list(url = "https://opal.example.org")),
    class = "OpalConnection"
  ))
  expect_error(
    dsFlowerClient:::.validate_dsi_transport_security(incomplete_opal),
    "opal: DSI transport security cannot be inspected"
  )
  withr::local_options(list(dsflower.dsi_tls_attested = "opal"))
  expect_invisible(
    dsFlowerClient:::.validate_dsi_transport_security(incomplete_opal)
  )

  conns$opal <- mock_dsi_connection(
    "OpalConnection", "https://opal.example.org",
    list(ssl_verifypeer = FALSE)
  )
  expect_error(
    dsFlowerClient:::.validate_dsi_transport_security(conns),
    "opal: TLS certificate verification is disabled"
  )

  conns$opal <- mock_dsi_connection(
    "OpalConnection", "https://opal.example.org",
    list(ssl_verifyhost = 0L)
  )
  expect_error(
    dsFlowerClient:::.validate_dsi_transport_security(conns),
    "opal: TLS certificate verification is disabled"
  )
})

test_that("DSOpal S4 connection metadata is inspected", {
  skip_if_not_installed("DSOpal")
  loadNamespace("DSOpal")
  withr::local_options(list(dsflower.dsi_tls_attested = character()))

  opal <- new.env(parent = emptyenv())
  opal$url <- "https://opal.example.org"
  opal$config <- structure(list(options = list()), class = "request")
  class(opal) <- "opal"
  conn <- methods::new("OpalConnection", name = "opal", opal = opal)

  expect_invisible(
    dsFlowerClient:::.validate_dsi_transport_security(list(opal = conn))
  )
  opal$config$options$ssl_verifypeer <- FALSE
  expect_error(
    dsFlowerClient:::.validate_dsi_transport_security(list(opal = conn)),
    "TLS certificate verification is disabled"
  )
})

test_that("DSMolgenisArmadillo S4 endpoint metadata is inspected", {
  skip_if_not_installed("DSMolgenisArmadillo")
  loadNamespace("DSMolgenisArmadillo")
  withr::local_options(list(dsflower.dsi_tls_attested = character()))

  handle <- structure(
    list(url = "https://armadillo.example.org"), class = "handle")
  conn <- methods::new(
    "ArmadilloConnection", name = "armadillo", handle = handle,
    user = "researcher", cookies = list(), token = "test-token")

  expect_invisible(
    dsFlowerClient:::.validate_dsi_transport_security(
      list(armadillo = conn))
  )
  conn@handle$url <- "http://armadillo.example.org"
  expect_error(
    dsFlowerClient:::.validate_dsi_transport_security(
      list(armadillo = conn)),
    "DSI endpoint uses plaintext HTTP"
  )
})

test_that("global httr TLS downgrades are detected with request precedence", {
  global_insecure <- structure(
    list(options = list(ssl_verifypeer = 0, ssl_verifyhost = 0)),
    class = "request"
  )
  withr::local_options(list(httr_config = global_insecure))

  opal <- list(opal = mock_dsi_connection(
    "OpalConnection", "https://opal.example.org"))
  expect_error(
    dsFlowerClient:::.validate_dsi_transport_security(opal),
    "TLS certificate verification is disabled"
  )

  opal$opal <- mock_dsi_connection(
    "OpalConnection", "https://opal.example.org",
    list(ssl_verifypeer = TRUE, ssl_verifyhost = TRUE)
  )
  expect_invisible(dsFlowerClient:::.validate_dsi_transport_security(opal))

  opal$opal <- mock_dsi_connection(
    "OpalConnection", "https://opal.example.org",
    list(ssl_verifypeer = TRUE)
  )
  expect_error(
    dsFlowerClient:::.validate_dsi_transport_security(opal),
    "TLS certificate verification is disabled"
  )

  opal$opal <- mock_dsi_connection(
    "OpalConnection", "https://opal.example.org",
    list(ssl_verifypeer = NULL, ssl_verifyhost = TRUE)
  )
  expect_error(
    dsFlowerClient:::.validate_dsi_transport_security(opal),
    "TLS certificate verification is disabled"
  )

  armadillo <- list(armadillo = mock_dsi_connection(
    "ArmadilloConnection", "https://armadillo.example.org"))
  expect_error(
    dsFlowerClient:::.validate_dsi_transport_security(armadillo),
    "TLS certificate verification is disabled"
  )
})

test_that("plaintext DSI needs an exact explicit site exception", {
  conns <- list(
    site1 = mock_dsi_connection(
      "ArmadilloConnection", "http://operator:secret@armadillo.example.org"
    )
  )
  withr::local_options(list(
    dsflower.dsi_tls_attested = "site1",
    dsflower.dsi_allow_insecure_http = character()
  ))

  error <- expect_error(
    dsFlowerClient:::.validate_dsi_transport_security(conns),
    "site1: DSI endpoint uses plaintext HTTP"
  )
  expect_false(grepl("operator|secret", conditionMessage(error)))

  withr::local_options(list(dsflower.dsi_allow_insecure_http = "site1"))
  expect_warning(
    expect_invisible(
      dsFlowerClient:::.validate_dsi_transport_security(conns)
    ),
    "allowing explicitly configured plaintext HTTP"
  )

  conns$site2 <- mock_dsi_connection(
    "ArmadilloConnection", "http://armadillo2.example.org"
  )
  expect_error(
    suppressWarnings(
      dsFlowerClient:::.validate_dsi_transport_security(conns)
    ),
    "site2: DSI endpoint uses plaintext HTTP"
  )
  conns$site2 <- NULL

  conns$site1 <- mock_dsi_connection("PrivateDSIConnection", "http://")
  expect_error(
    dsFlowerClient:::.validate_dsi_transport_security(conns),
    "site1: DSI endpoint uses plaintext HTTP"
  )

  conns$site1 <- mock_dsi_connection(
    "ArmadilloConnection", "ftp://armadillo.example.org"
  )
  expect_error(
    dsFlowerClient:::.validate_dsi_transport_security(conns),
    "site1: unsupported DSI endpoint scheme"
  )
})

test_that("unknown DSI connectors require an exact per-site attestation", {
  withr::local_options(list(dsflower.dsi_tls_attested = character()))
  conns <- list(
    site1 = mock_dsi_connection("PrivateDSIConnection"),
    site2 = mock_dsi_connection("PrivateDSIConnection")
  )

  expect_error(
    dsFlowerClient:::.validate_dsi_transport_security(conns),
    "site1: DSI transport security cannot be inspected"
  )

  withr::local_options(list(dsflower.dsi_tls_attested = "site1"))
  expect_error(
    dsFlowerClient:::.validate_dsi_transport_security(conns),
    "site2: DSI transport security cannot be inspected"
  )

  withr::local_options(list(
    dsflower.dsi_tls_attested = c("site1", "site2")
  ))
  expect_invisible(dsFlowerClient:::.validate_dsi_transport_security(conns))
})

test_that("DSLite is in-process and loopback exceptions are explicit", {
  withr::local_options(list(
    dsflower.dsi_tls_attested = character(),
    dsflower.dsi_allow_insecure_loopback = FALSE
  ))
  dslite <- list(local = mock_dsi_connection("DSLiteConnection"))
  expect_invisible(dsFlowerClient:::.validate_dsi_transport_security(dslite))

  loopback <- list(
    dev = mock_dsi_connection("OpalConnection", "http://127.0.0.1:8080")
  )
  expect_error(
    dsFlowerClient:::.validate_dsi_transport_security(loopback),
    "explicit development-only loopback exception"
  )

  withr::local_options(list(dsflower.dsi_allow_insecure_loopback = TRUE))
  expect_invisible(dsFlowerClient:::.validate_dsi_transport_security(loopback))

  unknown_loopback <- list(
    custom = mock_dsi_connection(
      "PrivateDSIConnection", "https://[::1]:8443"
    )
  )
  expect_invisible(
    dsFlowerClient:::.validate_dsi_transport_security(unknown_loopback)
  )

  remote <- list(
    prod = mock_dsi_connection("OpalConnection", "http://10.0.0.8:8080")
  )
  expect_error(
    dsFlowerClient:::.validate_dsi_transport_security(remote),
    "prod: DSI endpoint uses plaintext HTTP"
  )
})

test_that("DSI security options are validated fail closed", {
  conns <- list(local = mock_dsi_connection("DSLiteConnection"))
  for (bad in list(TRUE, NA_character_, "", c("site1", "site1"))) {
    withr::local_options(list(dsflower.dsi_tls_attested = bad))
    expect_error(
      dsFlowerClient:::.validate_dsi_transport_security(conns),
      "Invalid dsflower.dsi_tls_attested option"
    )
  }

  withr::local_options(list(dsflower.dsi_tls_attested = character()))
  for (bad in list(TRUE, NA_character_, "", c("site1", "site1"))) {
    expect_error(
      dsFlowerClient:::.validate_dsi_transport_security(
        conns, allow_insecure_http = bad),
      "Invalid dsflower.dsi_allow_insecure_http option"
    )
  }

  for (bad in list(1, NA, c(TRUE, FALSE), "true")) {
    withr::local_options(list(dsflower.dsi_allow_insecure_loopback = bad))
    expect_error(
      dsFlowerClient:::.validate_dsi_transport_security(conns),
      "Invalid dsflower.dsi_allow_insecure_loopback option"
    )
  }
})

test_that("link-up checks DSI security before starting local services", {
  withr::local_options(list(dsflower.dsi_tls_attested = character()))
  touched_superlink <- FALSE
  conns <- list(site1 = mock_dsi_connection("PrivateDSIConnection"))
  local_mocked_bindings(
    ds.flower.superlink.status = function() {
      touched_superlink <<- TRUE
      list(running = TRUE, ports = list(fleet = 9092L))
    },
    .package = "dsFlowerClient"
  )

  expect_error(
    dsFlowerClient::ds.flower.link.up(conns),
    "site1: DSI transport security cannot be inspected"
  )
  expect_false(touched_superlink)
})

test_that("high-level runs reject plaintext before their first DSI side effect", {
  conns <- list(site1 = mock_dsi_connection(
    "ArmadilloConnection", "http://armadillo.example.org"))
  connected <- FALSE
  local_mocked_bindings(
    .require_flwr_cli = function() TRUE,
    ds.flower.connect = function(...) {
      connected <<- TRUE
      stop("connect reached")
    },
    .package = "dsFlowerClient"
  )

  expect_error(
    ds.flower.submit(
      conns, model = "pytorch_logreg", target = "y", features = "x"),
    "site1: DSI endpoint uses plaintext HTTP"
  )
  expect_false(connected)

  expect_error(
    ds.flower.hook.run(
      conns, user_app_dir = ".", target = "y", features = "x"),
    "site1: DSI endpoint uses plaintext HTTP"
  )
  expect_false(connected)

  expect_error(
    ds.flower.submit(
      conns, model = "pytorch_logreg", target = "y", features = "x",
      allow_insecure_http = "site1"),
    "connect reached"
  )
  expect_true(connected)
})
