# Focused tests for public preprocessing and Hook/Tier-2 client hardening.

test_that("harness dependencies include the secure RNG backend", {
  expect_true("cryptography>=42.0.0" %in%
                dsFlowerClient:::.harness_dependencies())
  expect_true("flwr[app]>=1.31.0,<1.32.0" %in%
                dsFlowerClient:::.harness_dependencies())
  expect_true("torch>=2.0.0,<3.0.0" %in%
                dsFlowerClient:::.harness_dependencies())
  expect_true("opacus>=1.4.0,<2.0.0" %in%
                dsFlowerClient:::.harness_dependencies())
})

test_that("client venv marker invalidates an incompatible Flower runtime", {
  venv <- withr::local_tempdir()
  bindir <- dsFlowerClient:::.venv_bindir(venv)
  dir.create(bindir, recursive = TRUE)
  for (binary in c("python", "flwr", "flower-superlink")) {
    file.create(file.path(bindir, dsFlowerClient:::.venv_exe(binary)))
  }
  marker <- file.path(venv, ".dsflower_client_ready")
  writeLines(dsFlowerClient:::.client_venv_marker(), marker)
  local_mocked_bindings(
    .client_venv_path = function() venv,
    .package = "dsFlowerClient"
  )

  expect_true(dsFlowerClient:::.client_venv_is_healthy())
  writeLines("flwr[app]>=1.31.0", marker)
  expect_false(dsFlowerClient:::.client_venv_is_healthy())
})

test_that("local runner hash matches the server canonical algorithm", {
  runner <- withr::local_tempdir()
  dir.create(file.path(runner, "nested"))
  dir.create(file.path(runner, "__pycache__"))
  writeBin(charToRaw("a"), file.path(runner, "a.py"))
  writeBin(charToRaw("z"), file.path(runner, "nested", "z.txt"))
  writeBin(charToRaw("ignored"), file.path(runner, "__pycache__", "a.pyc"))
  blob <- c(
    charToRaw("a.py\n"), charToRaw("a"), as.raw(0x00),
    charToRaw("nested/z.txt\n"), charToRaw("z"), as.raw(0x00)
  )

  expect_identical(
    dsFlowerClient:::.compute_local_runner_hash(runner),
    digest::digest(blob, algo = "sha256", serialize = FALSE)
  )
})

test_that("runner preflight requires ABI 2 and the exact local hash", {
  expected <- paste(rep("a", 64L), collapse = "")
  captured <- NULL
  local_mocked_bindings(
    .compute_local_runner_hash = function(...) expected,
    .package = "dsFlowerClient"
  )
  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      captured <<- expr
      list(site1 = list(runner_abi = 2L, runner_sha256 = expected),
           site2 = list(runner_abi = 2, runner_sha256 = toupper(expected)))
    },
    .package = "DSI"
  )

  expect_silent(
    dsFlowerClient:::.assert_runner_compatibility(
      list(site1 = list(), site2 = list()), "flower_handle")
  )
  expect_identical(as.character(captured[[1L]]), "flowerGetCapabilitiesDS")
  expect_identical(captured[[2L]], "flower_handle")
})

test_that("runner preflight reports incompatible nodes", {
  expected <- paste(rep("a", 64L), collapse = "")
  local_mocked_bindings(
    .compute_local_runner_hash = function(...) expected,
    .package = "dsFlowerClient"
  )
  local_mocked_bindings(
    datashield.aggregate = function(...) {
      list(old = list(runner_abi = 1L, runner_sha256 = expected),
           drift = list(runner_abi = 2L,
                        runner_sha256 = paste(rep("b", 64L), collapse = "")))
    },
    .package = "DSI"
  )

  expect_error(
    dsFlowerClient:::.assert_runner_compatibility(
      list(old = list(), drift = list()), "flower_handle"),
    "old .*runner_abi=1.*drift"
  )
})

test_that("runner preflight rejects DSI 1.8 NULL and misassociated results", {
  expected <- paste(rep("a", 64L), collapse = "")
  conns <- list(site1 = list(), site2 = list())
  local_mocked_bindings(
    .compute_local_runner_hash = function(...) expected,
    .package = "dsFlowerClient"
  )
  result <- list(site1 = list(runner_abi = 2L, runner_sha256 = expected),
                 site2 = NULL)
  local_mocked_bindings(
    datashield.aggregate = function(...) result,
    .package = "DSI"
  )
  expect_error(
    dsFlowerClient:::.assert_runner_compatibility(conns, "flower_handle"),
    "no node capabilities returned on: site2"
  )

  result <- list(
    site1 = list(runner_abi = 2L, runner_sha256 = expected),
    wrong = list(runner_abi = 2L, runner_sha256 = expected)
  )
  expect_error(
    dsFlowerClient:::.assert_runner_compatibility(conns, "flower_handle"),
    "no node capabilities returned"
  )
})

test_that("public feature bounds are validated in feature order", {
  bounds <- dsFlowerClient:::.validate_public_feature_bounds(
    list(lower = c(0, 10), upper = c(2, 14)), c("a", "b"))

  expect_equal(bounds$lower, c(0, 10))
  expect_equal(bounds$upper, c(2, 14))
  expect_equal(bounds$features, c("a", "b"))
  expect_error(
    dsFlowerClient:::.validate_public_feature_bounds(
      list(lower = 0, upper = 1), c("a", "b")),
    "feature count"
  )
  expect_error(
    dsFlowerClient:::.validate_public_feature_bounds(
      list(lower = c(0, 2), upper = c(0, 1)), c("a", "b")),
    "lower < upper"
  )
  expect_error(
    dsFlowerClient:::.validate_public_feature_bounds(
      list(lower = -1e7, upper = 1e7), "a"),
    "1e6"
  )
})

test_that("saved public bounds are read back for prediction", {
  model_dir <- withr::local_tempdir()
  jsonlite::write_json(
    list(feature_lower = c(0, 10), feature_upper = c(2, 14)),
    file.path(model_dir, "metadata.json"),
    auto_unbox = TRUE
  )

  expect_equal(
    dsFlowerClient:::.read_meta_bounds(model_dir),
    list(lower = c(0, 10), upper = c(2, 14))
  )

  jsonlite::write_json(
    list(feature_lower = c(0, 10), feature_upper = c(0, 9)),
    file.path(model_dir, "metadata.json"),
    auto_unbox = TRUE
  )
  expect_null(dsFlowerClient:::.read_meta_bounds(model_dir))
})

test_that("fit forwards public feature bounds without shifting legacy arguments", {
  seen <- NULL
  fake_model <- structure(
    list(name = "pytorch_logreg", template = "pytorch_logreg",
         framework = "pytorch", params = list()),
    class = "dsflower_model"
  )
  local_mocked_bindings(
    ds.flower.model = function(...) fake_model,
    ds.flower.submit = function(...) {
      seen <<- list(...)
      structure(list(), class = "dsflower_run")
    },
    .package = "dsFlowerClient"
  )

  bounds <- list(lower = 0, upper = 1)
  levels <- c("no", "yes")
  ds.flower.fit(
    conns = list(site = TRUE), target = "y", features = "x",
    feature_bounds = bounds, target_levels = levels
  )
  expect_equal(seen$feature_bounds, bounds)
  expect_equal(seen$target_levels, levels)
})

test_that("submit appends public target semantics for positional compatibility", {
  expect_identical(
    tail(names(formals(ds.flower.submit)), 4L),
    c("feature_bounds", "target_levels", "target_bounds",
      "allow_insecure_http"))
})

test_that("public target semantics are strict and data-independent", {
  validate <- dsFlowerClient:::.validate_public_target_spec
  expect_equal(
    validate(c("no", "yes"), NULL, "classification", n_classes = 2L)$levels,
    c("no", "yes"))
  expect_error(
    validate(c("a", "b"), NULL, "classification", n_classes = 3L),
    "class count")
  expect_error(
    validate(NULL, NULL, "regression"), "required")
  expect_equal(
    validate(NULL, list(lower = 0, upper = 10), "regression")$bounds,
    list(lower = 0, upper = 10))
  expect_error(
    validate(NULL, list(lower = -1, upper = 10), "count"), "lower >= 0")
  expect_error(
    validate(NULL, list(lower = 0, upper = 10), "regression",
             loss_name = "gamma_nll"), "lower > 0")
})

test_that("submission target shape matches the node-owned loss", {
  scalar <- dsFlowerClient:::.emit_submission(ds.flower.model("pytorch_logreg"))
  multi <- dsFlowerClient:::.emit_submission(
    ds.flower.model("pytorch_multilabel", num_labels = 3L))

  expect_silent(dsFlowerClient:::.validate_submission_target(scalar, "outcome"))
  expect_error(
    dsFlowerClient:::.validate_submission_target(scalar, c("a", "b")),
    "exactly one"
  )
  expect_silent(
    dsFlowerClient:::.validate_submission_target(multi, c("a", "b", "c"))
  )
  expect_error(
    dsFlowerClient:::.validate_submission_target(multi, c("a", "b")),
    "num_labels=3"
  )
})

test_that("adaptive strategy is fully serialized before any side effect", {
  reached_cli <- FALSE
  local_mocked_bindings(
    .require_flwr_cli = function() {
      reached_cli <<- TRUE
      stop("validated-before-side-effects")
    },
    .package = "dsFlowerClient"
  )
  local_mocked_bindings(
    datashield.aggregate = function(...) stop("unexpected DSI side effect"),
    .package = "DSI"
  )

  expect_error(
    ds.flower.submit(
      conns = list(site = TRUE), model = "pytorch_logreg",
      target = "y", features = "x",
      strategy = ds.flower.strategy.fedadam(server_learning_rate = 0.2)),
    "validated-before-side-effects"
  )
  expect_true(reached_cli)
})

test_that("round horizons fail before any Flower side effect", {
  expect_error(
    ds.flower.submit(
      conns = list(), model = "pytorch_logreg", target = "y",
      num_rounds = 0),
    "positive integer"
  )
  expect_error(
    ds.flower.hook.run(
      conns = list(), user_app_dir = ".", target = "y", features = "x",
      num_rounds = 1.5),
    "positive integer"
  )
})

test_that("submit checks runner compatibility immediately after connect", {
  events <- character()
  local_conn <- structure(list(), class = "DSLiteConnection")
  flower <- structure(
    list(conns = list(site = local_conn), symbol = "flower"),
    class = "dsflower_connection"
  )
  model <- structure(list(name = "model"), class = "dsflower_model")
  local_mocked_bindings(
    .require_flwr_cli = function() TRUE,
    .emit_submission = function(...) list(track = "neural"),
    ds.flower.connect = function(...) {
      events <<- c(events, "connect")
      flower
    },
    .assert_runner_compatibility = function(...) {
      events <<- c(events, "compat")
      stop("runner mismatch")
    },
    ds.flower.link.down = function(...) {
      events <<- c(events, "link")
      invisible(TRUE)
    },
    ds.flower.nodes.cleanup = function(...) {
      events <<- c(events, "nodes")
      invisible(TRUE)
    },
    ds.flower.disconnect = function(...) {
      events <<- c(events, "disconnect")
      invisible(TRUE)
    },
    .package = "dsFlowerClient"
  )

  expect_error(
    ds.flower.submit(
      conns = list(site = local_conn), model = model,
      target = "y", features = "x"),
    "runner mismatch"
  )
  expect_identical(
    events,
    c("connect", "compat", "link", "nodes", "disconnect")
  )
})

test_that("hook upload rejects unsafe chunk sizes and module names", {
  expect_error(
    dsFlowerClient:::.upload_user_module(
      conns = list(), user_pkg_dir = ".", chunk_bytes = 0),
    "positive integer"
  )

  parent <- withr::local_tempdir()
  bad <- file.path(parent, "bad-name")
  dir.create(bad)
  expect_error(
    dsFlowerClient:::.upload_user_module(list(), bad),
    "valid Python module name"
  )

  missing_init <- file.path(parent, "valid_name")
  dir.create(missing_init)
  expect_error(
    dsFlowerClient:::.upload_user_module(list(), missing_init),
    "__init__.py"
  )
})

test_that("public app upload rejects unsafe chunk sizes before building", {
  expect_error(
    ds.flower.app.upload(list(), ".", chunk_bytes = 0),
    "positive integer"
  )
})

test_that("app archives stream in bounded chunks with exact acknowledgements", {
  archive <- withr::local_tempfile(fileext = ".fab")
  payload <- as.raw(rep(0:255, length.out = 1024 * 1024 + 123L))
  writeBin(payload, archive)
  received <- raw(0)
  chunk_lengths <- integer()
  conns <- list(site1 = TRUE, site2 = TRUE)

  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      encoded <- as.character(expr[[3L]])
      b64 <- substr(encoded, 5L, nchar(encoded, type = "chars"))
      b64 <- gsub("-", "+", b64, fixed = TRUE)
      b64 <- gsub("_", "/", b64, fixed = TRUE)
      pad <- (4L - nchar(b64) %% 4L) %% 4L
      if (pad) b64 <- paste0(b64, strrep("=", pad))
      chunk <- jsonlite::base64_dec(b64)
      received <<- c(received, chunk)
      chunk_lengths <<- c(chunk_lengths, length(chunk))
      offset <- as.numeric(expr[[4L]])
      expected <- offset + length(chunk)
      hash <- digest::digest(chunk, algo = "sha256", serialize = FALSE)
      stats::setNames(lapply(conns, function(...) list(
        ok = TRUE, offset = offset, size = expected,
        bytes = length(chunk), sha256 = hash)), names(conns))
    },
    .package = "DSI"
  )

  expect_invisible(dsFlowerClient:::.push_app_archive(
    conns, archive, "app_0123456789abcdef0123456789abcdef", 128 * 1024L))
  expect_identical(received, payload)
  expect_lte(max(chunk_lengths), 128 * 1024L)
  expect_gt(length(chunk_lengths), 1L)
})

test_that("app archive streaming rejects a node offset mismatch", {
  archive <- withr::local_tempfile(fileext = ".fab")
  writeBin(as.raw(1:8), archive)
  conns <- list(site1 = TRUE, site2 = TRUE)
  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      chunk <- jsonlite::base64_dec(paste0(
        chartr("-_", "+/", substring(expr[[3L]], 5L)),
        strrep("=", (4L - nchar(substring(expr[[3L]], 5L)) %% 4L) %% 4L)))
      offset <- as.numeric(expr[[4L]])
      expected <- as.numeric(expr[[4L]]) + 8
      hash <- digest::digest(chunk, algo = "sha256", serialize = FALSE)
      list(
        site1 = list(ok = TRUE, offset = offset, size = expected,
                     bytes = 8L, sha256 = hash),
        site2 = list(ok = TRUE, offset = offset, size = expected - 1,
                     bytes = 8L, sha256 = hash))
    },
    .package = "DSI"
  )

  expect_error(
    dsFlowerClient:::.push_app_archive(
      conns, archive, "app_0123456789abcdef0123456789abcdef", 1024L),
    "invalid ACK on: site2"
  )
})

test_that("a NULL node result retries the exact same app chunk", {
  archive <- withr::local_tempfile(fileext = ".fab")
  payload <- as.raw(1:16)
  writeBin(payload, archive)
  conns <- list(site1 = TRUE, site2 = TRUE)
  calls <- list()

  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      calls[[length(calls) + 1L]] <<- expr
      encoded <- substring(expr[[3L]], 5L)
      b64 <- chartr("-_", "+/", encoded)
      b64 <- paste0(b64, strrep("=", (4L - nchar(b64) %% 4L) %% 4L))
      chunk <- jsonlite::base64_dec(b64)
      offset <- as.numeric(expr[[4L]])
      ack <- list(
        ok = TRUE, offset = offset, size = offset + length(chunk),
        bytes = length(chunk),
        sha256 = digest::digest(chunk, algo = "sha256", serialize = FALSE))
      if (length(calls) == 1L) {
        return(list(site1 = NULL, site2 = ack))
      }
      list(site1 = ack, site2 = ack)
    },
    .package = "DSI"
  )

  expect_invisible(dsFlowerClient:::.push_app_archive(
    conns, archive, "app_0123456789abcdef0123456789abcdef", 1024L))
  expect_length(calls, 2L)
  expect_identical(calls[[2L]], calls[[1L]])
})

test_that("failed hook upload removes its partial node spool", {
  parent <- withr::local_tempdir()
  package_dir <- file.path(parent, "valid_hook")
  dir.create(package_dir)
  writeLines("", file.path(package_dir, "__init__.py"))
  calls <- character()
  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      fun <- as.character(expr[[1L]])
      calls <<- c(calls, fun)
      if (identical(fun, "flowerAppPushDS")) stop("push failed")
      list()
    },
    .package = "DSI"
  )

  expect_error(
    dsFlowerClient:::.upload_user_module(list(site = TRUE), package_dir),
    "returned no ACK"
  )
  expect_identical(tail(calls, 1L), "flowerAppDeleteDS")
})

test_that("failed public app upload removes its partial node spool", {
  fab <- withr::local_tempfile(fileext = ".fab")
  writeBin(as.raw(1:4), fab)
  calls <- character()
  local_mocked_bindings(
    .flwr_build_fab = function(...) fab,
    .package = "dsFlowerClient"
  )
  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      fun <- as.character(expr[[1L]])
      calls <<- c(calls, fun)
      if (identical(fun, "flowerAppPushDS")) stop("push failed")
      list()
    },
    .package = "DSI"
  )

  expect_error(
    ds.flower.app.upload(list(site = TRUE), ".", verbose = FALSE),
    "returned no ACK"
  )
  expect_identical(tail(calls, 1L), "flowerAppDeleteDS")
})

test_that("hook run cleans client state when upload fails", {
  events <- character()
  local_conn <- structure(list(), class = "DSLiteConnection")
  flower <- structure(
    list(conns = list(site = local_conn), symbol = "flower"),
    class = "dsflower_connection"
  )
  local_mocked_bindings(
    .require_flwr_cli = function() TRUE,
    ds.flower.connect = function(...) flower,
    .assert_runner_compatibility = function(...) {
      events <<- c(events, "compat")
      TRUE
    },
    .upload_user_module = function(...) {
      events <<- c(events, "upload")
      stop("upload failed")
    },
    ds.flower.link.down = function(...) {
      events <<- c(events, "link")
      invisible(TRUE)
    },
    ds.flower.nodes.cleanup = function(...) {
      events <<- c(events, "nodes")
      invisible(TRUE)
    },
    ds.flower.disconnect = function(...) {
      events <<- c(events, "disconnect")
      invisible(TRUE)
    },
    .package = "dsFlowerClient"
  )

  expect_error(
    ds.flower.hook.run(
      conns = list(site = local_conn), user_app_dir = ".",
      target = "y", features = "x"),
    "upload failed"
  )
  expect_identical(events[1:2], c("compat", "upload"))
  expect_setequal(events[-(1:2)], c("link", "nodes", "disconnect"))
})

test_that("legacy tier2 entry point delegates to classification hook", {
  seen <- NULL
  local_mocked_bindings(
    ds.flower.hook.run = function(...) {
      seen <<- list(...)
      "ok"
    },
    .package = "dsFlowerClient"
  )

  expect_warning(
    out <- ds.flower.tier2.run(
      conns = list(), user_app_dir = "pkg", target = "y",
      features = "x", num_rounds = 2L),
    "deprecated"
  )
  expect_identical(out, "ok")
  expect_identical(seen$task, "classification")
  expect_identical(seen$num_rounds, 2L)
})
