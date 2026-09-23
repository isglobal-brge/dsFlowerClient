public_init_summary <- function() {
  list(provenance = list(manifest_sha256 = strrep("a", 64L),
    manifest = list(model_id = "pytorch_resnet18_segmentation", decoder = "narrow")),
    checkpoint_sha256 = strrep("b", 64L), encoder_sha256 = strrep("c", 64L),
    tensor_schema = list(list(name = "0", shape = list(1L), dtype = "float32",
                              sha256 = strrep("d", 64L))),
    identity_version = "dsflower-public-init-v1")
}

test_that("checkpoint upload bounds DSI expressions and requires exact acknowledgements", {
  archive <- withr::local_tempfile(fileext = ".zip")
  writeBin(as.raw(rep(1:255, length.out = 600000L)), archive)
  summary <- public_init_summary()
  calls <- list()
  token <- paste0("cku_", strrep("a", 32L))
  local_mocked_bindings(.dsi_private_aggregate = function(conns, expr) {
    calls[[length(calls) + 1L]] <<- expr
    action <- expr[[2L]]
    value <- if (action == "begin") {
      list(upload_id = token, next_index = 1L)
    } else if (action == "chunk") {
      list(upload_id = token, next_index = expr$index + 1L)
    } else if (action == "finish") {
      list(upload_id = token, public_initialisation = summary)
    } else list(aborted = TRUE)
    setNames(list(value), names(conns))
  }, .package = "dsFlowerClient")
  admitted <- dsFlowerClient:::.segmentation_upload_bundle(list(site = TRUE), archive, summary)
  expect_identical(admitted, list(site = token))
  chunks <- Filter(function(x) identical(x[[2L]], "chunk"), calls)
  expect_length(chunks, 2L)
  restored <- do.call(c, lapply(chunks, function(x) {
    wire <- chartr("-_", "+/", substring(x$chunk_b64, 5L))
    wire <- paste0(wire, strrep("=", (4L - nchar(wire) %% 4L) %% 4L))
    jsonlite::base64_dec(wire)
  }))
  expect_identical(restored, readBin(archive, "raw", file.size(archive)))
  expect_equal(vapply(chunks, function(x) x$index, integer(1)), 1:2)
  expect_true(all(vapply(chunks, function(x) nchar(paste(deparse(x), collapse = "")) < 1000000L,
                         logical(1))))
  expect_identical(calls[[1L]]$manifest_sha256, summary$provenance$manifest_sha256)
  expect_equal(calls[[1L]]$total_bytes, 600000)
  expect_false(any(vapply(calls, function(x) any(grepl(archive, as.character(x), fixed = TRUE)),
                         logical(1))))
})

test_that("checkpoint upload denial stops before transmitting bytes", {
  archive <- withr::local_tempfile()
  writeBin(as.raw(1:8), archive)
  calls <- character()
  local_mocked_bindings(.dsi_private_aggregate = function(conns, expr) {
    calls <<- c(calls, expr[[2L]])
    setNames(list(NULL), names(conns))
  }, .package = "dsFlowerClient")
  expect_error(dsFlowerClient:::.segmentation_upload_bundle(
    list(site = TRUE), archive, public_init_summary()), "refused")
  expect_identical(calls, "begin")
})

test_that("checkpoint upload disagreement aborts every created admission", {
  archive <- withr::local_tempfile()
  writeBin(as.raw(1:8), archive)
  summary <- public_init_summary()
  token <- paste0("cku_", strrep("a", 32L))
  aborted <- character()
  local_mocked_bindings(.dsi_private_aggregate = function(conns, expr) {
    value <- switch(expr[[2L]],
      begin = list(upload_id = token, next_index = 1L),
      chunk = list(upload_id = token, next_index = expr$index + 1L),
      finish = {
        changed <- summary
        changed$checkpoint_sha256 <- strrep("f", 64L)
        list(upload_id = token, public_initialisation = changed)
      },
      abort = { aborted <<- c(aborted, expr$upload_id); list(aborted = TRUE) })
    setNames(list(value), names(conns))
  }, .package = "dsFlowerClient")
  expect_error(dsFlowerClient:::.segmentation_upload_bundle(
    list(site = TRUE), archive, summary), "invalid ACK")
  expect_identical(aborted, token)
})

test_that("resource initialization requires an independent local copy before private staging", {
  called <- FALSE
  local_mocked_bindings(.dsi_private_aggregate = function(...) {
    called <<- TRUE
    stop("must not query")
  }, .ensure_client_framework = function(...) TRUE, .package = "dsFlowerClient")
  expect_error(dsFlowerClient:::.segmentation_client_initialization(list(site = TRUE),
    list(decoder_init = "resource:CKPT")), "public_checkpoint_file")
  expect_false(called)
  expect_error(dsFlowerClient:::.segmentation_client_initialization(list(site = TRUE),
    list(decoder_init = "random"), "weights.npz"), "requires resource")
})

test_that("resource coordinator weights use only admitted public summaries", {
  checkpoint <- withr::local_tempfile(fileext = ".npz")
  writeBin(as.raw(1:8), checkpoint)
  summary <- public_init_summary()
  seen <- NULL
  local_mocked_bindings(
    .ensure_client_framework = function(...) TRUE,
    .dsi_private_aggregate = function(conns, expr) {
      expect_identical(expr, quote(flowerCheckpointStatusDS("CKPT")))
      setNames(rep(list(summary), length(conns)), names(conns))
    },
    .segmentation_checkpoint_python = function(operation, path, spec, summary, ...) {
      seen <<- list(operation = operation, path = path, summary = summary)
      c(summary, list(local_arrays_b64 = "AQIDBA=="))
    }, .package = "dsFlowerClient")
  result <- dsFlowerClient:::.segmentation_client_initialization(list(a = TRUE, b = TRUE),
    list(decoder_init = "resource:CKPT", decoder = "narrow"), checkpoint)
  expect_identical(seen$operation, "coordinator")
  expect_identical(seen$summary, summary)
  expect_null(result$uploads)
  expect_identical(result$origin, paste0("resource:", strrep("a", 64L)))
})

test_that("analyst bundle validates locally then uploads the complete archive", {
  archive <- withr::local_tempfile(fileext = ".zip")
  writeBin(as.raw(1:8), archive)
  summary <- public_init_summary()
  calls <- character()
  local_mocked_bindings(
    .ensure_client_framework = function(...) TRUE,
    .segmentation_checkpoint_python = function(operation, ...) {
      calls <<- c(calls, operation)
      c(summary, list(local_arrays_b64 = "AQIDBA=="))
    },
    .segmentation_upload_bundle = function(conns, archive, summary) {
      calls <<- c(calls, "upload")
      list(site = paste0("cku_", strrep("e", 32L)))
    }, .package = "dsFlowerClient")
  result <- dsFlowerClient:::.segmentation_client_initialization(list(site = TRUE),
    list(decoder_init = paste0("client:", archive), decoder = "narrow"))
  expect_identical(calls, c("local", "upload"))
  expect_identical(result$origin, "analyst-declared")
})

test_that("per-node analyst admissions remain distinct and failure rolls back the federation", {
  prepared <- list()
  cleanup <- FALSE
  local_mocked_bindings(ds.flower.nodes.prepare = function(conns, symbol, ...,
                                                          run_config) {
    prepared[[names(conns)]] <<- run_config[["segmentation-decoder-init"]]
    if (identical(names(conns), "b")) stop("node b refused")
    list(per_site = setNames(list(list()), names(conns)))
  }, .dsi_cleanup_run_exact = function(conns, symbol) {
    cleanup <<- identical(names(conns), c("a", "b"))
    list(a = list(cleanup_ok = TRUE), b = list(cleanup_ok = TRUE))
  }, .package = "dsFlowerClient")
  expect_error(dsFlowerClient:::.segmentation_prepare_nodes(list(a = TRUE, b = TRUE),
    "flower", "mask", NULL, list(), list(uploads = list(a = "token_a", b = "token_b"))),
    "node b refused")
  expect_identical(prepared, list(a = "client:token_a", b = "client:token_b"))
  expect_true(cleanup)
})
