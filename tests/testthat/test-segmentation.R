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
  loss <- dsFlowerClient:::.neural_training_config(sub$params, sub$loss)
  expect_equal(loss[["segmentation-alpha"]], .5)
  expect_equal(loss[["segmentation-smooth"]], 1)
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
  expect_false(any(c("epsilon", "delta", "noise-multiplier", "max-grad-norm") %in%
                     names(prepared$config)))
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
