# Focused tests for public preprocessing and Hook/Tier-2 client hardening.

test_that("harness dependencies include the secure RNG backend", {
  expect_identical(
    grep("^flwr", dsFlowerClient:::.DSFLOWER_CLIENT_PYTHON_DEPS,
         value = TRUE),
    "flwr==1.31.0"
  )
  expect_true("cryptography>=42.0.0" %in%
                dsFlowerClient:::.harness_dependencies())
  expect_true("flwr==1.31.0" %in%
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
  writeLines("flwr==1.30.0", marker)
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

test_that("runner preflight requires ABI 3 and the exact local hash", {
  expected <- paste(rep("a", 64L), collapse = "")
  captured <- NULL
  local_mocked_bindings(
    .compute_local_runner_hash = function(...) expected,
    .package = "dsFlowerClient"
  )
  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      captured <<- expr
      list(site1 = list(runner_abi = 3L, runner_sha256 = expected),
           site2 = list(runner_abi = 3, runner_sha256 = toupper(expected)))
    },
    .package = "DSI"
  )

  expect_silent(
    dsFlowerClient:::.assert_runner_compatibility(
      list(site1 = list(), site2 = list()))
  )
  expect_identical(as.character(captured[[1L]]), "flowerGetCapabilitiesDS")
  expect_identical(as.character(captured[[2L]]), "none")
  expect_length(captured, 2L)
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
           drift = list(runner_abi = 3L,
                        runner_sha256 = paste(rep("b", 64L), collapse = "")))
    },
    .package = "DSI"
  )

  expect_error(
    dsFlowerClient:::.assert_runner_compatibility(
      list(old = list(), drift = list())),
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
  result <- list(site1 = list(runner_abi = 3L, runner_sha256 = expected),
                 site2 = NULL)
  local_mocked_bindings(
    datashield.aggregate = function(...) result,
    .package = "DSI"
  )
  expect_error(
    dsFlowerClient:::.assert_runner_compatibility(conns),
    "no node capabilities returned on: site2"
  )

  result <- list(
    site1 = list(runner_abi = 3L, runner_sha256 = expected),
    wrong = list(runner_abi = 3L, runner_sha256 = expected)
  )
  expect_error(
    dsFlowerClient:::.assert_runner_compatibility(conns),
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
  expect_error(
    dsFlowerClient:::.validate_public_feature_bounds(
      list(lower = "0", upper = "1"), "a"),
    "numeric vectors"
  )
  expect_error(
    dsFlowerClient:::.validate_public_feature_bounds(
      list(lower = FALSE, upper = TRUE), "a"),
    "numeric vectors"
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

test_that("saved declarative model contract is read without guessing architecture", {
  model_dir <- withr::local_tempdir()
  spec <- list(kind = "sequential", layers = list(
    list(op = "linear", out = 8L), list(op = "tanh"),
    list(op = "linear", out = "@out")))
  jsonlite::write_json(
    list(model_spec = spec, loss_name = "cross_entropy",
         model_params = list(n_classes = 4L, num_labels = 3L),
         data_kind = "tabular"),
    file.path(model_dir, "metadata.json"),
    auto_unbox = TRUE
  )

  contract <- dsFlowerClient:::.read_meta_model_contract(model_dir)
  expect_identical(contract$model_spec$kind, "sequential")
  expect_identical(contract$model_spec$layers[[2L]]$op, "tanh")
  expect_identical(contract$loss_name, "cross_entropy")
  expect_identical(contract$num_classes, 4L)
  expect_identical(contract$num_labels, 3L)
  expect_identical(contract$data_kind, "tabular")
})

test_that("malformed metadata cannot expand prediction dimensions", {
  model_dir <- withr::local_tempdir()
  jsonlite::write_json(
    list(model_spec = "not-an-object", loss_name = "",
         model_params = list(n_classes = 1e9, num_labels = -1),
         data_kind = "unsupported"),
    file.path(model_dir, "metadata.json"), auto_unbox = TRUE)

  contract <- dsFlowerClient:::.read_meta_model_contract(model_dir)
  expect_null(contract$model_spec)
  expect_null(contract$loss_name)
  expect_identical(contract$num_classes, 2L)
  expect_identical(contract$num_labels, 2L)
  expect_null(contract$data_kind)
})

test_that("saved data kind prevents tabular prediction of vision artifacts", {
  model_dir <- withr::local_tempdir()
  jsonlite::write_json(
    list(data_kind = "image", model_spec = list(kind = "linear")),
    file.path(model_dir, "metadata.json"), auto_unbox = TRUE)
  writeBin(as.raw(0), file.path(model_dir, "model.pt"))

  contract <- dsFlowerClient:::.read_meta_model_contract(model_dir)
  expect_identical(contract$data_kind, "image")
  expect_error(
    ds.flower.predict(model_dir, data.frame(x = 1)),
    "accepts tabular artifacts only")

  jsonlite::write_json(
    list(model = "pytorch_resnet18", framework = "pytorch_vision"),
    file.path(model_dir, "metadata.json"), auto_unbox = TRUE)
  expect_null(dsFlowerClient:::.read_meta_model_contract(model_dir)$data_kind)
  expect_error(
    ds.flower.predict(model_dir, data.frame(x = 1)),
    "metadata must declare data_kind"
  )
})

test_that("local prediction rejects Tier-2 and other no-spec artifacts", {
  model_dir <- withr::local_tempdir()
  jsonlite::write_json(
    list(model = "tier2", data_kind = "tabular"),
    file.path(model_dir, "metadata.json"), auto_unbox = TRUE)
  writeBin(as.raw(0), file.path(model_dir, "model.pt"))
  local_mocked_bindings(
    .ensure_client_framework = function(...) {
      stop("framework setup must not run")
    },
    .package = "dsFlowerClient"
  )

  expect_error(
    ds.flower.predict(model_dir, data.frame(x = 1)),
    "model_spec.*loss_name.*Tier-2"
  )
})

test_that("fit forwards public feature bounds through its stable argument contract", {
  seen <- NULL
  fake_model <- structure(
    list(name = "pytorch_logreg", framework = "pytorch", params = list()),
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

test_that("submit exposes public target semantics in canonical order", {
  expect_identical(
    tail(names(formals(ds.flower.submit)), 7L),
    c("feature_bounds", "feature_cuts", "target_levels", "target_bounds",
      "holdout", "cross_validation", "allow_insecure_http"))
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

test_that("low-level submit validates model parameters before any side effect", {
  reached_cli <- FALSE
  local_mocked_bindings(
    .require_flwr_cli = function() {
      reached_cli <<- TRUE
      stop("CLI must not be reached")
    },
    .package = "dsFlowerClient"
  )

  expect_error(
    ds.flower.submit(
      conns = list(site = TRUE), model = "pytorch_logreg",
      target = "y", features = "x",
      model_params = list(epsilon = 999)),
    "Unknown parameter.*epsilon"
  )
  expect_false(reached_cli)
})

test_that("low-level submit enforces model input kind before any side effect", {
  expect_error(
    ds.flower.submit(
      conns = list(), model = "pytorch_resnet18", target = "y",
      data_kind = "tabular"),
    "does not support data_kind"
  )
  expect_error(
    ds.flower.submit(
      conns = list(), model = "pytorch_logreg", target = "y",
      data_kind = "image"),
    "does not support data_kind"
  )
  expect_error(
    ds.flower.submit(
      conns = list(), model = "pytorch_logreg", target = "y",
      data_kind = "IMAGE"),
    "exactly 'tabular' or 'image'"
  )
})

test_that("architecture geometry fails before any Flower side effect", {
  reached_cli <- FALSE
  local_mocked_bindings(
    .require_flwr_cli = function() {
      reached_cli <<- TRUE
      stop("CLI must not be reached")
    },
    .package = "dsFlowerClient"
  )

  expect_error(
    ds.flower.submit(
      conns = list(site = TRUE),
      model = ds.flower.model(
        "pytorch_cnn", input_shape = c(1L, 8L, 8L)),
      target = "y", features = paste0("x", seq_len(63L))),
    "expects 64 public features"
  )
  expect_false(reached_cli)
})

test_that("declarative resource caps fail before DSI or staging", {
  reached_dsi <- FALSE
  reached_preflight <- FALSE
  local_mocked_bindings(
    .require_flwr_cli = function() TRUE,
    .ensure_client_framework = function(framework) {
      expect_identical(framework, "pytorch")
      TRUE
    },
    .package = "dsFlowerClient"
  )
  local_mocked_bindings(
    run = function(...) {
      reached_preflight <<- TRUE
      list(
        status = 2L,
        stderr = "invalid declarative model: parameter budget exceeded",
        stdout = ""
      )
    },
    .package = "processx"
  )
  local_mocked_bindings(
    datashield.aggregate = function(...) {
      reached_dsi <<- TRUE
      stop("DSI must not be reached")
    },
    .package = "DSI"
  )
  expect_error(
    ds.flower.submit(
      conns = list(site = TRUE),
      model = ds.flower.model(
        "pytorch_mlp", hidden_layers = 8192L),
      target = "y", features = paste0("x", seq_len(1024L))),
    "Declarative model preflight failed"
  )
  expect_true(reached_preflight)
  expect_false(reached_dsi)
})

test_that("learning-rate schedules must affect the pinned training horizon", {
  common <- list(conns = list(site = TRUE), target = "y", features = "x")
  expect_error(do.call(ds.flower.submit, c(common, list(
    model = ds.flower.model(
      "pytorch_logreg", scheduler = "exponential"),
    num_rounds = 1L))), "at least two")
  expect_error(do.call(ds.flower.submit, c(common, list(
    model = ds.flower.model(
      "pytorch_logreg", scheduler = "step", scheduler_step_size = 2L),
    num_rounds = 2L))), "smaller than")
  expect_error(do.call(ds.flower.submit, c(common, list(
    model = ds.flower.model(
      "pytorch_logreg", scheduler = "exponential", scheduler_gamma = 10),
    num_rounds = 4L))), "above 10")
})

test_that("bce_logits cannot silently request a multiclass head", {
  expect_error(
    dsFlowerClient:::.validate_submission_target(
      list(loss = "bce_logits", params = list(n_classes = 3L)), "y"),
    "binary only"
  )
})

test_that("unknown model names fail before any side effect", {
  reached_cli <- FALSE
  local_mocked_bindings(
    .require_flwr_cli = function() {
      reached_cli <<- TRUE
      stop("CLI must not be reached")
    },
    .package = "dsFlowerClient"
  )

  expect_error(
    ds.flower.submit(
      conns = list(site = TRUE), model = "unknown_model",
      target = "y", features = "x"),
    "Unknown model"
  )
  expect_false(reached_cli)
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

test_that("classified strategy objects are revalidated exactly", {
  forged <- structure(
    list(name = "FedAvg", params = list(eta = 0.1)),
    class = "dsflower_strategy")
  expect_error(
    ds.flower.submit(
      conns = list(), model = "pytorch_logreg", target = "y",
      features = "x", strategy = forged),
    "inapplicable parameters"
  )
  forged <- structure(
    list(name = "FedAdam", params = list(beta_1 = 2)),
    class = "dsflower_strategy")
  expect_error(
    ds.flower.submit(
      conns = list(), model = "pytorch_logreg", target = "y",
      features = "x", strategy = forged),
    "beta_1"
  )
})

test_that("round horizons fail before any Flower side effect", {
  expect_error(
    ds.flower.submit(
      conns = list(), model = "pytorch_logreg", target = "y",
      num_rounds = 0),
    "positive integer"
  )
  expect_error(
    ds.flower.submit(
      conns = list(), model = "pytorch_logreg", target = "y",
      num_rounds = "2"),
    "positive integer"
  )
  expect_error(
    ds.flower.submit(
      conns = list(), model = "pytorch_logreg", target = "y",
      num_rounds = TRUE),
    "positive integer"
  )
  expect_error(
    ds.flower.hook.run(
      conns = list(), user_app_dir = ".", target = "y", features = "x",
      num_rounds = 1.5),
    "positive integer"
  )
  expect_error(
    ds.flower.hook.run(
      conns = list(), user_app_dir = ".", target = "y", features = "x",
      num_rounds = "2"),
    "positive integer"
  )
  expect_error(
    ds.flower.hook.run(
      conns = list(), user_app_dir = ".", target = "y", features = "x",
      num_rounds = TRUE),
    "positive integer"
  )
})

test_that("torch backend choices fail before any Flower side effect", {
  expect_identical(dsFlowerClient:::.validate_torch_backend("CPU"), "cpu")
  expect_identical(dsFlowerClient:::.validate_torch_backend("cu126"), "cu126")
  for (value in list("rocm", "custom", "", TRUE, c("cpu", "gpu"))) {
    expect_error(dsFlowerClient:::.validate_torch_backend(value),
                 "torch_backend")
  }
  expect_error(
    ds.flower.submit(
      conns = list(), model = "pytorch_logreg", target = "y",
      torch_backend = "rocm"),
    "torch_backend"
  )
})

test_that("Hook feature geometry fails before any Flower side effect", {
  for (features in list(1:2, c("x", "x"), c("x", NA_character_), "")) {
    expect_error(
      ds.flower.hook.run(
        conns = list(), user_app_dir = ".", target = "y",
        features = features),
      "feature columns"
    )
  }
})

test_that("framework parameters never use R partial matching", {
  config <- dsFlowerClient:::.neural_training_config(
    list(learning_rate_decay = 0.8), "bce_logits")
  expect_identical(config[["learning-rate"]], 0.01)
  expect_silent(dsFlowerClient:::.dsflower_validate_parameter_limits(
    list(momentum_decay = 2)))
})

test_that("submit checks runner compatibility immediately after connect", {
  events <- character()
  local_conn <- structure(list(), class = "DSLiteConnection")
  flower <- structure(
    list(conns = list(site = local_conn), symbol = "flower"),
    class = "dsflower_connection"
  )
  model <- ds.flower.model("pytorch_logreg")
  local_mocked_bindings(
    .require_flwr_cli = function() TRUE,
    .emit_submission = function(...) list(track = "neural"),
    .validate_declarative_model_preflight = function(...) TRUE,
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
  expect_error(
    dsFlowerClient:::.upload_user_module(
      conns = list(), user_pkg_dir = ".", chunk_bytes = 512L * 1024L + 1L),
    "512 KiB"
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
  expect_error(
    ds.flower.app.upload(
      list(), ".", chunk_bytes = 512L * 1024L + 1L),
    "512 KiB"
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

test_that("a DSI-dropped node result retries the exact same app chunk", {
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
        return(list(site2 = ack))
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
      list(site = list(
        hook_execution_configured = TRUE, hook_enabled = TRUE,
        hook_sandbox_attested = TRUE,
        hook_resource_isolation_attested = TRUE,
        hook_time_envelope_configured = TRUE))
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

test_that("hook readiness fails before upload instead of silently no-oping", {
  conns <- list(site = TRUE)
  caps <- list(site = list(
    hook_execution_configured = FALSE, hook_enabled = TRUE,
    hook_sandbox_attested = FALSE,
    hook_resource_isolation_attested = TRUE,
    hook_time_envelope_configured = FALSE))
  expect_error(
    dsFlowerClient:::.assert_hook_execution_configured(caps, conns),
    "site.*sandbox,timing")
})

test_that("HookApp parameters are recursively canonical and hash pinned", {
  params <- list(
    zeta = list(3L, NULL, TRUE, list(beta = "", alpha = 0.25)),
    alpha = "adam",
    enabled = FALSE,
    integer_value = 1L,
    double_value = 1
  )
  pinned <- dsFlowerClient:::.canonical_hook_app_params(params)

  expect_identical(
    names(pinned$value),
    c("alpha", "double_value", "enabled", "integer_value", "zeta"))
  expect_identical(names(pinned$value$zeta[[4L]]), c("alpha", "beta"))
  expect_identical(
    jsonlite::base64_dec(pinned$b64), charToRaw(enc2utf8(pinned$json)))
  expect_identical(
    pinned$sha256,
    digest::digest(charToRaw(enc2utf8(pinned$json)),
                   algo = "sha256", serialize = FALSE))
  expect_identical(
    pinned$json,
    paste0(
      '{"alpha":"adam","double_value":1.0,"enabled":false,',
      '"integer_value":1,"zeta":[3,null,true,{"alpha":0.25,"beta":""}]}'
    )
  )

  reordered <- dsFlowerClient:::.canonical_hook_app_params(
    params[c("enabled", "alpha", "zeta", "double_value", "integer_value")])
  expect_identical(reordered$b64, pinned$b64)
  expect_identical(reordered$sha256, pinned$sha256)
})

test_that("HookApp parameter validation is bounded and fail closed", {
  bad <- list(
    1,
    list(epsilon = 1),
    list(training_epsilon = 1),
    list(noiseMultiplier = 1),
    list(apiToken = "public-value"),
    list(model_path = "weights"),
    list(requirements = "numpy"),
    list(round_index = 3L),
    list(label = "folder/model.bin"),
    list(value = Inf),
    list(value = NaN),
    list(value = NA_real_),
    list(value = 1 + 2i),
    list(value = factor("a")),
    list(value = structure(1, class = "custom_scalar")),
    structure(list(1L, 2L), names = c("named", ""))
  )
  for (value in bad) {
    expect_error(dsFlowerClient:::.canonical_hook_app_params(value))
  }

  nested <- list(leaf = 1L)
  for (i in seq_len(9L)) nested <- list(nested = nested)
  expect_error(
    dsFlowerClient:::.canonical_hook_app_params(nested), "nesting depth")
  expect_error(
    dsFlowerClient:::.canonical_hook_app_params(
      list(values = as.list(rep.int(0L, 2048L)))),
    "item count"
  )
  expect_error(
    dsFlowerClient:::.canonical_hook_app_params(
      stats::setNames(as.list(rep.int(strrep("x", 4000L), 20L)),
                      sprintf("field%02d", seq_len(20L)))),
    "65536 bytes"
  )
})

test_that("invalid HookApp parameters fail before any connection side effect", {
  touched <- FALSE
  local_mocked_bindings(
    .require_flwr_cli = function() {
      touched <<- TRUE
      stop("connection side effect")
    },
    .package = "dsFlowerClient"
  )

  expect_error(
    ds.flower.hook.run(
      conns = list(), user_app_dir = ".", target = "y", features = "x",
      app_params = list(dp_epsilon = 1000)),
    "reserved"
  )
  expect_false(touched)
})
