segmentation_model_fixture <- function() {
  path <- tempfile("segmentation_")
  dir.create(path)
  sub <- dsFlowerClient:::.emit_submission(
    ds.flower.model.pytorch_resnet18_segmentation())
  jsonlite::write_json(list(
    model = "pytorch_resnet18_segmentation", framework = "pytorch_vision",
    track = "neural", data_kind = "image", loss_name = sub$loss,
    model_spec = sub$spec, model_params = sub$params,
    available = TRUE, status = "success", privacy = "server-enforced-dp"),
    file.path(path, "metadata.json"), auto_unbox = TRUE)
  writeBin(as.raw(1:8), file.path(path, "model.pt"))
  path
}

segmentation_initialization_fixture <- function(id = "busi-v5-epochs60-seed20260919") {
  bytes <- as.raw(1:8)
  list(provenance = list(
    manifest_sha256 = strrep("a", 64L),
    manifest = list(checkpoint_id = id, model_id = "pytorch_resnet18_segmentation",
      decoder = "narrow", dataset = list(publisher_md5 = NULL,
        numeric_provenance = 0.12345678901234567),
      checkpoint = list(file = "decoder.npz",
        sha256 = digest::digest(bytes, algo = "sha256", serialize = FALSE),
        size_bytes = length(bytes)))),
    checkpoint_base64 = jsonlite::base64_enc(bytes))
}

test_that("segmentation named, generic and recipe routes agree", {
  named <- ds.flower.model.pytorch_resnet18_segmentation()
  generic <- ds.flower.model("pytorch_resnet18_segmentation")
  expect_identical(named, generic)
  expect_identical(named$loss, "segmentation_bce_dice")
  expect_identical(named$framework, "pytorch_vision")
  expect_identical(ds.flower.recipe(named)$task$type, "segmentation")
  expect_error(ds.flower.recipe(named, task = "classification"), "incompatible")
  expect_identical(ds.flower.task.segmentation(), ds.flower.task("segmentation"))
  catalogue <- ds.flower.list_models()
  expect_true("pytorch_resnet18_segmentation" %in% catalogue$name)
  expect_false(catalogue$vetted[catalogue$name == "pytorch_resnet18_segmentation"])
})

test_that("segmentation rejects unapproved loss, geometry and privacy controls", {
  for (alpha in list(0, .4, .9, 2, Inf, NA, TRUE)) {
    expect_error(ds.flower.model.pytorch_resnet18_segmentation(alpha = alpha))
  }
  expect_equal(ds.flower.model.pytorch_resnet18_segmentation(alpha = 1)$params$alpha, 1)
  expect_error(ds.flower.model.pytorch_resnet18_segmentation(mask_values = "auto"))
  for (key in c("epsilon", "delta", "noise_multiplier", "image_size", "backbone")) {
    expect_error(do.call(ds.flower.model, c(
      list("pytorch_resnet18_segmentation"), setNames(list(1), key))))
  }
})

test_that("segmentation codegen pins the decoder and spatial extractor", {
  sub <- dsFlowerClient:::.emit_submission(
    ds.flower.model.pytorch_resnet18_segmentation())
  expect_identical(sub$spec, dsFlowerClient:::.segmentation_decoder_spec())
  expect_identical(sub$params$vision_extractor_profile, "resnet18_layer2_128_v1")
  expect_identical(sub$params$backbone, "resnet18_layer2")
  expect_identical(sub$params$image_size, 128L)
  cfg <- dsFlowerClient:::.segmentation_public_config(sub$params)
  expect_identical(cfg[["segmentation-output-shape"]], "1,128,128")
  expect_identical(cfg[["segmentation-selection"]], "canonical-image-id-lexicographic-v1")
  expect_match(cfg[["segmentation-checkpoint-sha256"]], "^[0-9a-f]{64}$")
  expect_identical(sub$params$decoder_init, "random")
  expect_false("segmentation-decoder-init" %in% names(cfg))
  loss <- dsFlowerClient:::.neural_training_config(sub$params, sub$loss)
  expect_equal(loss[["segmentation-alpha"]], .5)
  expect_equal(loss[["segmentation-smooth"]], 1)
})

test_that("public decoder initialization has one validated constructor contract", {
  init <- "public:busi-v5-epochs60-seed20260919"
  named <- ds.flower.model.pytorch_resnet18_segmentation(
    decoder = "narrow", decoder_init = init)
  generic <- ds.flower.model("pytorch_resnet18_segmentation",
    decoder = "narrow", decoder_init = init)
  expect_identical(named, generic)
  expect_identical(ds.flower.model(named), named)
  expect_identical(ds.flower.recipe(named)$model$params$decoder_init, init)
  sub <- dsFlowerClient:::.emit_submission(named)
  expect_identical(sub$spec, dsFlowerClient:::.segmentation_decoder_spec("narrow"))
  cfg <- dsFlowerClient:::.segmentation_public_config(sub$params)
  expect_identical(cfg[["segmentation-decoder-init"]], init)
  expect_identical(cfg[["segmentation-checkpoint-sha256"]],
    dsFlowerClient:::.SEGMENTATION_CHECKPOINT_SHA256)
  expect_setequal(names(cfg), c("segmentation-checkpoint-sha256",
    "segmentation-selection", "segmentation-preprocessing",
    "segmentation-output-shape", "mask-vocabulary", "segmentation-decoder-init"))
  expect_identical(
    dsFlowerClient:::.segmentation_public_config(
      dsFlowerClient:::.emit_submission(
        ds.flower.model.pytorch_resnet18_segmentation(decoder_init = "random"))$params),
    dsFlowerClient:::.segmentation_public_config(
      dsFlowerClient:::.emit_submission(
        ds.flower.model.pytorch_resnet18_segmentation())$params))
})

test_that("public decoder ids reject paths, malformed values and node-owned pins", {
  invalid <- list("", "public:", "PUBLIC:valid", "public:Upper", "public:-id",
    "public:a/b", "public:../checkpoint", "public:https://example.org/a",
    "public:a b", "public:id\n", "public:caf\u00e9", " random",
    paste0("public:", strrep("a", 65L)), NA_character_, TRUE, 1,
    c("random", "public:id"), list("public:id"), NULL)
  for (value in invalid) {
    expect_error(ds.flower.model.pytorch_resnet18_segmentation(decoder_init = value),
                 "decoder_init")
  }
  for (value in c("public:a", "public:0._-", paste0("public:", strrep("a", 64L)))) {
    expect_identical(ds.flower.model.pytorch_resnet18_segmentation(
      decoder_init = value)$params$decoder_init, value)
  }
  for (key in c("segmentation_public_checkpoint", "segmentation_public_provenance",
                "segmentation_public_manifest_sha256", "segmentation_checkpoint_sha256")) {
    expect_error(do.call(ds.flower.model, c(list("pytorch_resnet18_segmentation"),
      setNames(list("analyst-supplied"), key))), "Unknown parameter")
  }
  altered <- ds.flower.model.pytorch_resnet18_segmentation()
  altered$params$decoder_init <- "public:../../untrusted"
  expect_error(ds.flower.model(altered), "decoder_init")
})

test_that("segmentation mask target and unsupported private scoring fail in public preflight", {
  sub <- dsFlowerClient:::.emit_submission(
    ds.flower.model.pytorch_resnet18_segmentation())
  expect_identical(dsFlowerClient:::.validate_submission_target(sub, "mask_path"), "mask_path")
  expect_error(dsFlowerClient:::.validate_submission_target(sub, c("a", "b")), "one target")
  expect_error(dsFlowerClient:::.validate_public_target_spec(c(0, 1), NULL, "segmentation"), "mask-path")
  expect_error(dsFlowerClient:::.assert_holdout_supported(sub, "image"), "unsupported")
  expect_error(dsFlowerClient:::.assert_cross_validation_supported(sub, "image"), "unsupported")
  local_mocked_bindings(
    .require_flwr_cli = function(...) stop("CLI must not run"),
    .validate_dsi_transport_security = function(...) stop("DSI must not run"),
    .package = "dsFlowerClient")
  expect_error(ds.flower.submit(list(), model = "pytorch_resnet18_segmentation",
    target = "mask_path", data_kind = "image", holdout = .2), "unsupported")
  expect_error(ds.flower.submit(list(), model = "pytorch_resnet18_segmentation",
    target = "mask_path", data_kind = "image", cross_validation = 3), "unsupported")
  expect_error(ds.flower.submit(list(), model = "pytorch_resnet18_segmentation",
    target = "mask_path", data_kind = "image", target_levels = c(0, 1)), "mask-path")
  path <- segmentation_model_fixture()
  on.exit(unlink(path, recursive = TRUE), add = TRUE)
  expect_error(ds.flower.validate(list(), path), "Private segmentation")
})

test_that("segmentation fit dispatch preserves image roles and task", {
  seen <- NULL
  local_mocked_bindings(ds.flower.submit = function(...) {
    seen <<- list(...)
    structure(list(), class = "dsflower_run")
  }, .package = "dsFlowerClient")
  ds.flower.fit(conns = list(site = TRUE), symbol = "D", target = "mask_path",
    model = "pytorch_resnet18_segmentation", task = "segmentation", data_kind = "image",
    model_params = list(sample_id_col = "image_key", mask_empty_col = "normal"))
  expect_identical(seen$data_kind, "image")
  expect_identical(seen$model$params$sample_id_col, "image_key")
  expect_identical(seen$model$params$mask_empty_col, "normal")
  expect_identical(seen$target, "mask_path")
  ds.flower.fit(conns = list(site = TRUE), symbol = "D", target = "mask_path",
    model = "pytorch_resnet18_segmentation", task = "segmentation", data_kind = "image",
    model_params = list(decoder = "narrow",
      decoder_init = "public:busi-v5-epochs60-seed20260919"))
  expect_identical(seen$model$params$decoder, "narrow")
  expect_identical(seen$model$params$decoder_init, "public:busi-v5-epochs60-seed20260919")
})

test_that("saved segmentation pins are required before local input reads", {
  path <- segmentation_model_fixture()
  on.exit(unlink(path, recursive = TRUE), add = TRUE)
  contract <- dsFlowerClient:::.resolve_segmentation_prediction_contract(path)
  expect_identical(contract$feature_dim, 32768L)
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$model_params$segmentation_checkpoint_sha256 <- paste(rep("0", 64), collapse = "")
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE)
  expect_error(ds.flower.predict(path, "secret/path.png"), "profile pin")
})

test_that("local segmentation prediction preserves canonical mask geometry", {
  values <- matrix(0, 2L, 16384L)
  values[1L, 129L] <- 1
  masks <- dsFlowerClient:::.format_segmentation_predictions(values, 2L, "response")
  expect_identical(dim(masks), c(2L, 1L, 128L, 128L))
  expect_identical(masks[1L, 1L, 2L, 1L], 1)
  expect_identical(sum(masks[2L, , , ]), 0)
  expect_error(dsFlowerClient:::.format_segmentation_predictions(values[, -1L], 2L, "prob"), "invalid masks")
  values[1L, 1L] <- .5
  expect_error(dsFlowerClient:::.format_segmentation_predictions(values, 2L, "response"), "invalid masks")
})

test_that("segmentation submit stages only the pinned public image and mask roles", {
  prepared <- NULL
  local_mocked_bindings(
    .require_flwr_cli = function(...) TRUE,
    .validate_dsi_transport_security = function(...) TRUE,
    .validate_declarative_model_preflight = function(...) TRUE,
    ds.flower.connect = function(conns, ...) list(conns = conns, symbol = "flower"),
    .assert_runner_compatibility = function(...) list(),
    ds.flower.nodes.prepare = function(conns, symbol, target_column,
                                        feature_columns, run_config) {
      prepared <<- list(target = target_column, features = feature_columns,
                         config = run_config)
      stop("captured prepare")
    },
    ds.flower.link.down = function(...) NULL,
    ds.flower.nodes.cleanup = function(...) NULL,
    .dsflower_disconnect_on_exit = function(...) NULL,
    .package = "dsFlowerClient")
  expect_error(ds.flower.submit(
    conns = list(site = TRUE), model = "pytorch_resnet18_segmentation",
    symbol = "D", target = "mask_path", data_kind = "image",
    model_params = list(mask_values = "0,1", mask_empty_col = "normal",
                        subject_id_col = "subject")), "captured prepare")
  expect_identical(prepared$target, "mask_path")
  expect_null(prepared$features)
  expect_identical(prepared$config[["task-type"]], "segmentation")
  expect_identical(prepared$config[["num-features"]], 32768L)
  expect_identical(prepared$config[["mask_path_col"]], "mask_path")
  expect_identical(prepared$config[["sample_id_col"]], "image_id")
  expect_identical(prepared$config[["mask_empty_col"]], "normal")
  expect_identical(prepared$config[["subject_id_col"]], "subject")
  expect_identical(prepared$config[["mask-vocabulary"]], "0,1")
  expect_false("num-labels" %in% names(prepared$config))
  expect_false(any(c("epsilon", "delta", "noise-multiplier", "max-grad-norm") %in%
                     names(prepared$config)))
  expect_false("segmentation-decoder-init" %in% names(prepared$config))
  expect_error(ds.flower.submit(
    conns = list(site = TRUE), model = "pytorch_resnet18_segmentation",
    symbol = "D", target = "mask_path", data_kind = "image",
    model_params = list(decoder = "narrow",
      decoder_init = "public:busi-v5-epochs60-seed20260919")), "captured prepare")
  expect_identical(prepared$config[["segmentation-decoder-init"]],
                   "public:busi-v5-epochs60-seed20260919")
})

test_that("local segmentation transport emits canonical arrays and masks paths in failures", {
  path <- segmentation_model_fixture()
  on.exit(unlink(path, recursive = TRUE), add = TRUE)
  seen <- NULL
  local_mocked_bindings(
    .ensure_client_framework = function(...) TRUE,
    .client_python_cmd = function() "python",
    .client_venv_env = function(...) character(),
    .package = "dsFlowerClient")
  local_mocked_bindings(run = function(command, args, ...) {
    seen <<- jsonlite::fromJSON(args[match("--config", args) + 1L])
    list(status = 0L, stderr = "", stdout = as.character(jsonlite::toJSON(
      matrix(.5, 1L, 16384L))))
  }, .package = "processx")
  probabilities <- ds.flower.predict(path, "secret/path.png", "prob")
  expect_identical(dim(probabilities), c(1L, 1L, 128L, 128L))
  expect_true(all(probabilities == .5))
  expect_identical(seen[["segmentation-output-shape"]], "1,128,128")
  expect_identical(seen[["loss-name"]], "segmentation_bce_dice")
  expect_identical(seen[["mask-vocabulary"]], "0,255")
  expect_error(ds.flower.predict(path, rep("secret/path.png", 123L)), "cell ceiling")
  local_mocked_bindings(run = function(...) {
    list(status = 1L, stderr = "secret/path.png", stdout = "")
  }, .package = "processx")
  message <- tryCatch(ds.flower.predict(path, "secret/path.png"), error = conditionMessage)
  expect_match(message, "saved artifact")
  expect_false(grepl("secret", message))
})

test_that("HPO objectives reject segmentation fit and submit before transport", {
  touched <- FALSE
  local_mocked_bindings(
    .require_flwr_cli = function(...) { touched <<- TRUE; stop("CLI reached") },
    .validate_dsi_transport_security = function(...) { touched <<- TRUE; stop("transport reached") },
    .package = "dsFlowerClient")
  run_objective <- dsFlowerClient:::.hpo_evaluate_objective
  for (method in list(ds.flower.fit, ds.flower.submit)) {
    expect_error(run_objective(function(params) {
      method(conns = list(site = TRUE), model = "pytorch_resnet18_segmentation",
             symbol = "D", target = "mask", data_kind = "image")
    }, list()), "Private segmentation HPO")
    expect_false(dsFlowerClient:::.DSFLOWER_HPO_CONTEXT$active)
    expect_false(touched)
  }
  expect_equal(run_objective(function(params) .75, list()), .75)
  expect_false(dsFlowerClient:::.DSFLOWER_HPO_CONTEXT$active)
  expect_error(run_objective(function(params) stop("objective failed"), list()), "objective failed")
  expect_false(dsFlowerClient:::.DSFLOWER_HPO_CONTEXT$active)
  run_objective(function(params) {
    run_objective(function(inner) .5, list())
    expect_true(dsFlowerClient:::.DSFLOWER_HPO_CONTEXT$active)
  }, list())
  expect_false(dsFlowerClient:::.DSFLOWER_HPO_CONTEXT$active)
})

test_that("the real local HPO callback cannot initiate private segmentation training", {
  python <- tryCatch(dsFlowerClient:::.local_hpo_python_cmd(), error = function(e) "")
  skip_if(!nzchar(python), "Optuna 4.8.0 is required")
  touched <- FALSE
  local_mocked_bindings(
    .local_hpo_python_cmd = function() python,
    .validate_dsi_transport_security = function(...) { touched <<- TRUE; stop("transport reached") },
    .require_flwr_cli = function(...) { touched <<- TRUE; stop("CLI reached") },
    .package = "dsFlowerClient")
  expect_error(ds.flower.hpo(function(params) {
    ds.flower.fit(conns = list(site = TRUE), model = "pytorch_resnet18_segmentation",
      symbol = "D", target = "mask", data_kind = "image")
  }, list(alpha = ds.flower.hpo.categorical(.5)), n_trials = 1L),
  "Private segmentation HPO")
  expect_false(touched)
  expect_false(dsFlowerClient:::.DSFLOWER_HPO_CONTEXT$active)
})

test_that("v4 public-development decoders are explicit choices with unchanged default", {
  original <- dsFlowerClient:::.segmentation_decoder_spec()
  expect_identical(original, dsFlowerClient:::.segmentation_decoder_spec("current"))
  narrow <- dsFlowerClient:::.segmentation_decoder_spec("narrow")
  expect_identical(narrow$layers[[2L]]$out_channels, 8L)
  expect_identical(narrow$layers[[5L]]$out_channels, 4L)
  pointwise <- dsFlowerClient:::.segmentation_decoder_spec("pointwise")
  expect_length(pointwise$layers, 3L)
  expect_identical(pointwise$layers[[3L]]$scale_factor, 8L)
  for (decoder in c("current", "narrow", "pointwise")) {
    expect_s3_class(ds.flower.model("pytorch_resnet18_segmentation", decoder = decoder), "dsflower_model")
  }
  expect_error(ds.flower.model("pytorch_resnet18_segmentation", decoder = "arbitrary"))
})

test_that("saved v4 decoders reconstruct locally and reject mismatched metadata", {
  path <- segmentation_model_fixture()
  on.exit(unlink(path, recursive = TRUE), add = TRUE)
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  for (decoder in c("current", "narrow", "pointwise")) {
    meta$model_params$decoder <- decoder
    meta$model_spec <- dsFlowerClient:::.segmentation_decoder_spec(decoder)
    jsonlite::write_json(meta, meta_path, auto_unbox = TRUE)
    contract <- dsFlowerClient:::.resolve_segmentation_prediction_contract(path)
    expect_identical(contract$feature_dim, 32768L)
    meta$model_params$decoder <- "unsupported"
    jsonlite::write_json(meta, meta_path, auto_unbox = TRUE)
    expect_error(dsFlowerClient:::.resolve_segmentation_prediction_contract(path), "decoder")
  }
})

test_that("saved decoder initialization accepts legacy defaults and validates public ids", {
  path <- segmentation_model_fixture()
  on.exit(unlink(path, recursive = TRUE), add = TRUE)
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$model_params$decoder_init <- NULL
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE)
  contract <- dsFlowerClient:::.resolve_segmentation_prediction_contract(path)
  expect_false("segmentation-decoder-init" %in% names(contract$segmentation_config))
  meta$model_params$decoder <- "narrow"
  meta$model_spec <- dsFlowerClient:::.segmentation_decoder_spec("narrow")
  meta$model_params$decoder_init <- "public:busi-v5-epochs60-seed20260919"
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE)
  contract <- dsFlowerClient:::.resolve_segmentation_prediction_contract(path)
  expect_identical(contract$segmentation_config[["segmentation-decoder-init"]],
                   meta$model_params$decoder_init)
  meta$model_params$decoder_init <- "public:../untrusted"
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE)
  expect_error(ds.flower.predict(path, "private/path.png"), "decoder_init")
})

test_that("public initialization relays only identical verified node checkpoint bytes", {
  payload <- segmentation_initialization_fixture()
  params <- list(decoder = "narrow", decoder_init = "public:busi-v5-epochs60-seed20260919")
  prepared <- list(per_site = list(a = list(segmentation_public_initialization = payload),
                                  b = list(segmentation_public_initialization = payload)))
  relay <- function(value = prepared) dsFlowerClient:::.segmentation_server_initialization(
    value, params, c("a", "b"))
  value <- relay()
  expect_identical(jsonlite::fromJSON(rawToChar(jsonlite::base64_dec(value$b64)),
    simplifyVector = FALSE), payload)
  expect_identical(value$provenance, payload$provenance)
  expect_null(dsFlowerClient:::.segmentation_server_initialization(NULL,
    list(decoder_init = "random"), "site"))
  missing <- prepared
  missing$per_site$b <- NULL
  expect_error(relay(missing), "every node")
  bad <- prepared
  bad$per_site$b$segmentation_public_initialization$checkpoint_base64 <- "AQIDBA=="
  expect_error(relay(bad), "digest mismatch")
  bad <- prepared
  bad$per_site$b$segmentation_public_initialization$provenance$manifest_sha256 <- strrep("b", 64L)
  expect_error(relay(bad), "disagree")
  bad <- prepared
  bad$per_site$b$segmentation_public_initialization$provenance$manifest$checkpoint_id <- "another"
  expect_error(relay(bad), "invalid")
  bad <- prepared
  bad$per_site$b$segmentation_public_initialization <- NULL
  expect_error(relay(bad), "invalid")
  bad <- prepared
  bad$per_site$b$segmentation_public_initialization$checkpoint_base64 <- "invalid%"
  expect_error(relay(bad), "invalid")
})

test_that("public submission initializes aggregation and persists node provenance", {
  payload <- segmentation_initialization_fixture()
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_superlink <- client_env$.superlink
  withr::defer(client_env$.superlink <- old_superlink)
  client_env$.superlink <- list(
    process = list(is_alive = function() TRUE), flwr_home = withr::local_tempdir())
  prepared <- app_config <- NULL
  app_dir <- withr::local_tempdir()
  local_mocked_bindings(
    .require_flwr_cli = function(...) TRUE,
    .validate_dsi_transport_security = function(...) TRUE,
    .validate_declarative_model_preflight = function(...) TRUE,
    ds.flower.connect = function(conns, ...) list(conns = conns, symbol = "flower"),
    .assert_runner_compatibility = function(...) list(),
    ds.flower.nodes.prepare = function(..., run_config) {
      prepared <<- run_config
      list(per_site = list(site = list(segmentation_public_initialization = payload)))
    },
    .build_submission_app = function(sub, config_lines, ...) {
      app_config <<- config_lines
      app_dir
    },
    .ensure_client_framework = function(...) TRUE,
    ds.flower.link.up = function(...) TRUE,
    ds.flower.nodes.ensure = function(...) TRUE,
    .client_flwr_cmd = function() "flwr",
    .client_venv_env = function(...) character(),
    .run_flwr_with_artifact_watchdog = function(...) list(
      status = 0L, stdout = "run_id=public-decoder", stderr = ""),
    .read_model_weights = function(...) list(coef = 1),
    .read_training_history = function(...) data.frame(round = 1L, n_failures = 0L),
    ds.flower.link.down = function(...) NULL,
    ds.flower.nodes.cleanup = function(...) NULL,
    .dsflower_disconnect_on_exit = function(...) NULL,
    .package = "dsFlowerClient")
  run <- ds.flower.submit(list(site = TRUE), symbol = "D", target = "mask_path",
    model = "pytorch_resnet18_segmentation", data_kind = "image", num_rounds = 1L,
    model_params = list(decoder = "narrow",
      decoder_init = "public:busi-v5-epochs60-seed20260919"), strategy = "fedadam",
    output_dir = withr::local_tempdir(), output_name = "public-decoder", silent = TRUE)
  expect_false("segmentation-public-initialization-b64" %in% names(prepared))
  expect_true(any(startsWith(app_config, "segmentation-public-initialization-b64 = ")))
  metadata <- jsonlite::fromJSON(file.path(run$output_dir, "metadata.json"),
                                simplifyVector = FALSE)
  expect_identical(metadata$segmentation_public_initialization, payload$provenance)
  saved <- readRDS(file.path(run$output_dir, "public-decoder.rds"))
  expect_identical(saved$segmentation_public_initialization, payload$provenance)
  expect_false("checkpoint_base64" %in% names(metadata$segmentation_public_initialization))
})
