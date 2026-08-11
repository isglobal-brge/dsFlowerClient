validation_model_fixture <- function(track = "neural", loss = "bce_logits") {
  path <- withr::local_tempdir(.local_envir = parent.frame())
  spec <- list(
    kind = "sequential", layers = list(list(op = "linear", out = "@out")))
  meta <- list(
    track = track, model_spec = spec, loss_name = loss,
    data_kind = "tabular",
    model_params = list(n_classes = 2L, num_labels = 2L),
    features = c("age", "marker"),
    feature_lower = c(0, -5), feature_upper = c(120, 5),
    target_levels = if (identical(loss, "bce_logits")) c("control", "case") else NULL,
    target_bounds = if (loss %in% c("mse", "quantile"))
      list(lower = 0, upper = 100) else NULL)
  jsonlite::write_json(meta, file.path(path, "metadata.json"),
                       auto_unbox = TRUE, null = "null")
  writeBin(charToRaw("public-model"), file.path(path, "model.pt"))
  path
}

vision_validation_model_fixture <- function(
    model = "pytorch_resnet18", n_classes = 2L, volumetric = FALSE,
    .local_envir = parent.frame()) {
  path <- withr::local_tempdir(.local_envir = .local_envir)
  backbone <- paste0(
    sub("^pytorch_", "", model), if (isTRUE(volumetric)) "_3d" else "")
  extractor_profile <- dsFlowerClient:::.vision_extractor_profile(backbone)
  image_size <- if (identical(backbone, "densenet121_3d")) 128L else 96L
  meta <- list(
    model_id = "vision-release", model = model,
    framework = "pytorch_vision", track = "neural",
    model_spec = list(
      kind = "sequential",
      layers = list(list(op = "linear", out = "@out"))),
    model_params = list(
      n_classes = as.integer(n_classes), volumetric = volumetric,
      image_size = image_size, backbone = backbone,
      vision_extractor_profile = extractor_profile),
    loss_name = "cross_entropy", data_kind = "image",
    available = TRUE, available_rounds = 1L,
    strategy = "fedavg", privacy = "server-enforced-dp",
    num_rounds = 1L, requested_num_rounds = 1L, n_clients = 2L,
    features = NULL, feature_lower = NULL, feature_upper = NULL,
    target_levels = paste0("class-", seq_len(n_classes)),
    target_bounds = NULL, created_at = "2026-08-11T12:00:00",
    holdout = NULL, holdout_contract = NULL, status = "success")
  jsonlite::write_json(
    meta, file.path(path, "metadata.json"), auto_unbox = TRUE,
    null = "list")
  writeBin(charToRaw("bounded-vision-state-dict"), file.path(path, "model.pt"))
  path
}

xgboost_validation_model_fixture <- function(task = "binary") {
  path <- withr::local_tempdir(.local_envir = parent.frame())
  artifact_name <- "model.xgboost-ensemble.json"
  model <- ds.flower.model.xgboost(task = task)
  request <- dsFlowerClient:::.build_xgboost_request(
    model$params, c("age", "marker"),
    list(lower = c(0, -5), upper = c(120, 5)),
    list(c(18, 40, 65), c(-1, 0, 1)),
    target_name = "outcome",
    target_levels = if (identical(task, "binary")) c("control", "case") else NULL,
    target_bounds = if (identical(task, "regression")) {
      list(lower = 0, upper = 100)
    } else NULL)
  public_schema_sha256 <- request$value$public_schema$sha256
  container <- list(
    contract = "dsflower-xgboost-ensemble-v1", version = 1L,
    engine = "xgboost", task = task, aggregation = "mean_prediction",
    public_schema_sha256 = public_schema_sha256,
    models = list(list(learner = list(attributes = list()),
                       version = list(3L, 4L, 0L))))
  artifact_path <- file.path(path, artifact_name)
  writeBin(dsFlowerClient:::.native_tree_json(container), artifact_path)
  bytes <- readBin(artifact_path, "raw", n = file.info(artifact_path)$size)
  artifact_hash <- digest::digest(bytes, algo = "sha256", serialize = FALSE)
  sanitization <- list(
    profile = "dsflower-xgboost-ensemble-v1",
    privacy_basis = "direct-dp-training-postprocessing",
    contains_raw_records = FALSE,
    contains_unnoised_statistics = FALSE,
    contains_feature_names = FALSE,
    contains_target_name = FALSE,
    contains_training_history = FALSE,
    contains_backend_logs = FALSE,
    contains_paths = FALSE,
    contains_executable_payload = FALSE)
  meta <- list(
    track = "native_tree", engine = "xgboost", task = task,
    data_kind = "tabular", features = c("age", "marker"),
    feature_lower = c(0, -5), feature_upper = c(120, 5),
    target_levels = if (identical(task, "binary")) c("control", "case") else NULL,
    target_bounds = if (identical(task, "regression"))
      list(lower = 0, upper = 100) else NULL,
    native_tree_request_b64 = request$b64,
    native_tree_request_sha256 = request$sha256,
    public_schema_sha256 = public_schema_sha256,
    artifact = list(
      file = artifact_name,
      format = "dsflower-xgboost-ensemble-json-v1",
      size_bytes = as.integer(length(bytes)), sha256 = artifact_hash),
    sanitization = sanitization)
  jsonlite::write_json(meta, file.path(path, "metadata.json"),
                       auto_unbox = TRUE, null = "null")
  profile <- list(
    artifact = list(
      format = "dsflower-xgboost-ensemble-json-v1",
      sha256 = artifact_hash, size_bytes = as.integer(length(bytes))),
    contract = "dsflower-xgboost-prediction-profile-v1",
    native_tree_request_b64 = request$b64,
    native_tree_request_sha256 = request$sha256,
    public_schema_sha256 = public_schema_sha256,
    task = task,
    version = 1L)
  writeBin(
    dsFlowerClient:::.native_tree_json(profile),
    file.path(path, "model.xgboost-ensemble.profile.json"))
  path
}

pure_native_tree_validation_model_fixture <- function(
    engine, task = "binary", .local_envir = parent.frame()) {
  path <- withr::local_tempdir(.local_envir = .local_envir)
  constructor <- switch(engine,
    extra_trees = ds.flower.model.extra_trees,
    random_forest = ds.flower.model.random_forest,
    lightgbm = ds.flower.model.lightgbm,
    catboost = ds.flower.model.catboost,
    stop("pure native-tree fixture engine is unsupported"))
  model <- constructor(task = task)
  request <- dsFlowerClient:::.build_native_tree_request(
    engine, model$params, c("age", "marker"),
    list(lower = c(0, -5), upper = c(120, 5)),
    list(c(18, 40, 65), c(-1, 0, 1)), "outcome",
    target_levels = if (identical(task, "binary"))
      c("control", "case") else NULL,
    target_bounds = if (identical(task, "regression"))
      list(lower = 0, upper = 100) else NULL)
  spec <- dsFlowerClient:::.native_tree_release_spec(engine)
  schema_hash <- request$value$public_schema$sha256
  container <- list(
    aggregation = "mean_prediction", contract = spec$artifact_contract,
    engine = engine, models = list(list()),
    public_schema_sha256 = schema_hash, task = task, version = 1L)
  artifact_path <- file.path(path, spec$artifact_file)
  writeBin(dsFlowerClient:::.native_tree_json(container), artifact_path)
  bytes <- readBin(artifact_path, "raw", n = file.info(artifact_path)$size)
  artifact_hash <- digest::digest(bytes, algo = "sha256", serialize = FALSE)
  artifact <- list(
    file = spec$artifact_file, format = spec$artifact_format,
    size_bytes = as.integer(length(bytes)), sha256 = artifact_hash)
  meta <- list(
    track = "native_tree", engine = engine, task = task,
    data_kind = "tabular", features = c("age", "marker"),
    feature_lower = c(0, -5), feature_upper = c(120, 5),
    target_levels = if (identical(task, "binary"))
      c("control", "case") else NULL,
    target_bounds = if (identical(task, "regression"))
      list(lower = 0, upper = 100) else NULL,
    native_tree_request_b64 = request$b64,
    native_tree_request_sha256 = request$sha256,
    public_schema_sha256 = schema_hash, artifact = artifact,
    sanitization = dsFlowerClient:::.native_tree_sanitization_attestation(
      engine))
  jsonlite::write_json(
    meta, file.path(path, "metadata.json"), auto_unbox = TRUE, null = "null")
  profile <- list(
    artifact = artifact[c("format", "sha256", "size_bytes")],
    contract = spec$profile_contract, engine = engine,
    native_tree_request_b64 = request$b64,
    native_tree_request_sha256 = request$sha256,
    public_schema_sha256 = schema_hash, task = task,
    version = spec$profile_version)
  writeBin(
    dsFlowerClient:::.native_tree_json(profile),
    file.path(path, spec$profile_file))
  path
}

test_that("validation contract resolves a declarative neural artifact", {
  path <- validation_model_fixture()
  contract <- dsFlowerClient:::.resolve_validation_contract(path, 24L)
  expect_identical(contract$track, "neural")
  expect_identical(contract$task, "binary")
  expect_identical(contract$bins, 24L)
  expect_identical(contract$features, c("age", "marker"))
  expect_equal(contract$feature_bounds$lower, c(0, -5))
  expect_identical(contract$target_levels, c("control", "case"))
})

test_that("validation treats quantile output as bounded regression", {
  path <- validation_model_fixture(loss = "quantile")
  contract <- dsFlowerClient:::.resolve_validation_contract(path, 24L)
  expect_identical(contract$task, "regression")
  expect_identical(contract$loss_name, "quantile")
  expect_equal(contract$target_bounds, list(lower = 0, upper = 100))
})

test_that("validation resolves only a sanitized XGBoost ensemble contract", {
  path <- xgboost_validation_model_fixture()
  contract <- dsFlowerClient:::.resolve_validation_contract(path, 24L)
  expect_identical(contract$track, "native_tree")
  expect_identical(contract$engine, "xgboost")
  expect_identical(contract$task, "binary")
  expect_identical(contract$data_kind, "tabular")
  expect_identical(
    contract$artifact_format, "dsflower-xgboost-ensemble-json-v1")
  expect_match(contract$artifact_sha256, "^[0-9a-f]{64}$")
  expect_match(contract$native_tree_request_sha256, "^[0-9a-f]{64}$")
  expect_match(contract$prediction_profile$sha256, "^[0-9a-f]{64}$")
  expect_identical(
    contract$prediction_profile$size_bytes,
    as.integer(file.info(contract$prediction_profile$path)$size))
  expect_identical(
    rawToChar(jsonlite::base64_dec(contract$native_tree_request_b64)),
    dsFlowerClient:::.validate_native_tree_request_wire(
      contract$native_tree_request_b64,
      contract$native_tree_request_sha256)$json)
  expect_match(contract$public_schema_sha256, "^[0-9a-f]{64}$")
  expect_identical(contract$loss_name, "bce_logits")

  regression <- dsFlowerClient:::.resolve_validation_contract(
    xgboost_validation_model_fixture("regression"), 32L)
  expect_identical(regression$task, "regression")
  expect_equal(regression$target_bounds, list(lower = 0, upper = 100))
  expect_identical(regression$loss_name, "mse")
})

test_that("pure native-tree bundles save and reopen with engine-specific assets", {
  for (engine in c("extra_trees", "random_forest", "lightgbm", "catboost")) {
    withr::local_tempdir()
    path <- pure_native_tree_validation_model_fixture(
      engine, .local_envir = environment())
    spec <- dsFlowerClient:::.native_tree_release_spec(engine)
    contract <- dsFlowerClient:::.resolve_validation_contract(path, 24L)
    expect_identical(contract$engine, engine)
    expect_identical(contract$artifact_format, spec$artifact_format)
    recipe <- list(
      model = list(engine = engine, task = "binary"),
      native_tree_request_b64 = contract$native_tree_request_b64,
      native_tree_request_sha256 = contract$native_tree_request_sha256,
      public_schema_sha256 = contract$public_schema_sha256)
    expect_no_error(dsFlowerClient:::.native_tree_release_metadata(recipe, path))

    run <- structure(list(
      model_id = engine, weights = NULL, available = TRUE,
      history = data.frame(round = 1L), model = engine,
      strategy = "mean_prediction", num_rounds = 1L,
      run_id = paste0("run-", engine), output_dir = path),
      class = "dsflower_run")
    destination <- file.path(withr::local_tempdir(), paste0(engine, ".rds"))
    expect_no_error(ds.flower.save_model(run, destination))
    assets <- paste0(destination, ".assets")
    expect_true(file.exists(file.path(assets, spec$artifact_file)))
    expect_true(file.exists(file.path(assets, spec$profile_file)))
    reopened <- dsFlowerClient:::.resolve_model_for_predict(assets)
    expect_identical(reopened$native_contract$engine, engine)
  }
})

test_that("prediction uses the exact XGBoost contract without provisioning torch", {
  path <- xgboost_validation_model_fixture()
  expect_true(file.exists(file.path(
    path, "model.xgboost-ensemble.profile.json")))
  resolved <- dsFlowerClient:::.resolve_model_for_predict(path)
  expect_identical(resolved$framework, "native_tree")
  expect_identical(basename(resolved$model_file),
                   "model.xgboost-ensemble.json")
  reached_framework <- FALSE
  local_mocked_bindings(
    .validate_validation_artifact_preflight = function(...) invisible(TRUE),
    .ensure_client_framework = function(...) {
      reached_framework <<- TRUE
      stop("must not provision torch")
    },
    .run_native_tree_local_predict = function(native_contract, frame, type) {
      expect_identical(native_contract$native_tree_request_sha256,
                       resolved$native_contract$native_tree_request_sha256)
      expect_identical(names(frame), c("age", "marker"))
      expect_identical(type, "response")
      c(0.49, 0.5)
    },
    .package = "dsFlowerClient")
  expect_identical(
    ds.flower.predict(
      path, data.frame(marker = c(0, 1), age = c(40, 65))),
    c("control", "case"))
  expect_false(reached_framework)
})

test_that("XGBoost local prediction validates columns, task and output", {
  expect_identical(
    names(dsFlowerClient:::.native_tree_prediction_frame(
      data.frame(marker = 0, age = 40), c("age", "marker"))),
    c("age", "marker"))
  expect_error(
    dsFlowerClient:::.native_tree_prediction_frame(
      data.frame(age = 40), c("age", "marker")),
    "missing: marker")
  expect_error(
    dsFlowerClient:::.native_tree_prediction_frame(
      data.frame(age = 40, marker = "x"), c("age", "marker")),
    "numeric or logical")

  path <- xgboost_validation_model_fixture("regression")
  expect_error(
    ds.flower.predict(
      path, data.frame(age = 40, marker = 0), type = "prob"),
    "unavailable for native-tree regression")

  output <- tempfile(fileext = ".json")
  on.exit(unlink(output), add = TRUE)
  writeBin(charToRaw(paste0(
    '{"contract":"dsflower-native-tree-local-prediction-v1",',
    '"engine":"xgboost","predictions":[0.25,0.75],"task":"binary",',
    '"type":"prob","version":1}')), output)
  expect_equal(
    dsFlowerClient:::.read_native_tree_prediction_output(
      output, 2L, "binary", "xgboost", "prob"),
    c(0.25, 0.75))
})

test_that("native release requires the exact bounded canonical profile sidecar", {
  expect_identical(
    dsFlowerClient:::.NATIVE_TREE_PREDICTION_PROFILE_MAX_BYTES, 128 * 1024L)
  path <- xgboost_validation_model_fixture()
  meta <- jsonlite::fromJSON(
    file.path(path, "metadata.json"), simplifyVector = TRUE)
  recipe <- list(
    model = list(engine = "xgboost", task = "binary"),
    native_tree_request_b64 = meta$native_tree_request_b64,
    native_tree_request_sha256 = meta$native_tree_request_sha256,
    public_schema_sha256 = meta$public_schema_sha256)
  expect_no_error(dsFlowerClient:::.native_tree_release_metadata(
    recipe, path))
  profile_path <- file.path(path, "model.xgboost-ensemble.profile.json")
  writeBin(charToRaw("opaque-public-profile"),
           profile_path)
  expect_error(
    dsFlowerClient:::.native_tree_release_metadata(recipe, path),
    "exact contract")
  writeBin(raw(128 * 1024L), profile_path)
  expect_error(
    dsFlowerClient:::.native_tree_release_metadata(recipe, path),
    "exact contract")
  writeBin(raw(128 * 1024L + 1L), profile_path)
  expect_error(
    dsFlowerClient:::.native_tree_release_metadata(recipe, path),
    "outside its byte bound")
  unlink(profile_path)
  expect_error(
    dsFlowerClient:::.native_tree_release_metadata(recipe, path),
    "missing or outside its byte bound")
})

test_that("portable save revalidates and copies only the exact XGBoost sidecar", {
  path <- xgboost_validation_model_fixture()
  writeBin(charToRaw("must-not-copy"), file.path(
    path, "model.xgboost-ensemble.profile.json.bak"))
  run <- structure(list(
    model_id = "native", weights = NULL, available = TRUE,
    history = data.frame(round = 1L), model = "xgboost",
    strategy = "mean_prediction", num_rounds = 1L, run_id = "run-native",
    output_dir = path), class = "dsflower_run")
  destination <- file.path(withr::local_tempdir(), "native.rds")
  expect_identical(
    ds.flower.save_model(run, destination),
    normalizePath(destination, winslash = "/", mustWork = TRUE))
  assets <- paste0(destination, ".assets")
  expect_true(file.exists(file.path(
    assets, "model.xgboost-ensemble.json")))
  expect_true(file.exists(file.path(
    assets, "model.xgboost-ensemble.profile.json")))
  expect_false(file.exists(file.path(
    assets, "model.xgboost-ensemble.profile.json.bak")))

  path <- xgboost_validation_model_fixture()
  profile_path <- file.path(path, "model.xgboost-ensemble.profile.json")
  bytes <- readBin(profile_path, "raw", n = file.info(profile_path)$size)
  writeBin(c(bytes, charToRaw("\n")), profile_path)
  run$output_dir <- path
  expect_error(
    ds.flower.save_model(
      run, file.path(withr::local_tempdir(), "tampered.rds")),
    "not canonically encoded")
})

test_that("XGBoost validation rejects public bundle tampering before private IO", {
  expect_identical(
    dsFlowerClient:::.NATIVE_TREE_ENSEMBLE_MAX_BYTES, 64 * 1024^2)
  path <- xgboost_validation_model_fixture()
  profile_path <- file.path(path, "model.xgboost-ensemble.profile.json")
  profile_bytes <- readBin(
    profile_path, what = "raw", n = file.info(profile_path)$size)
  writeBin(c(profile_bytes, charToRaw("\n")), profile_path)
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "not canonically encoded")
  expect_error(
    dsFlowerClient:::.resolve_model_for_predict(path),
    "not canonically encoded")

  path <- xgboost_validation_model_fixture()
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$artifact$sha256 <- strrep("0", 64L)
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "SHA-256 mismatch")

  path <- xgboost_validation_model_fixture()
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$artifact$size_bytes <- 64 * 1024^2 + 1
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "size is outside")

  path <- xgboost_validation_model_fixture()
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$sanitization$contains_backend_logs <- TRUE
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "sanitization attestation")

  path <- xgboost_validation_model_fixture()
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$public_schema_sha256 <- strrep("c", 64L)
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "request differs|public schema SHA-256 mismatch")

  path <- xgboost_validation_model_fixture()
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$target_bounds <- list(lower = 0, upper = 1)
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "two public target levels and no regression target bounds")

  path <- xgboost_validation_model_fixture("regression")
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$task <- "multiclass"
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "engine/task is not executable")
})

test_that("validation rejects unsupported artifacts and public config early", {
  path <- validation_model_fixture()
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 3L),
    "bins must be one integer")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, "32"),
    "bins must be one integer")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, TRUE),
    "bins must be one integer")
  expect_error(
    ds.flower.validate(
      conns = list(), model = path, target = "outcome",
      torch_backend = "rocm"),
    "torch_backend")
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$track <- "egress"
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "declarative neural artifacts")

  meta$track <- "trees"
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "declarative neural artifacts")

  path <- validation_model_fixture()
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$model_params$n_classes <- 3L
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "not a valid multiclass artifact")
})

test_that("validation accepts only pinned saved vision classifiers", {
  binary_path <- vision_validation_model_fixture()
  binary <- dsFlowerClient:::.resolve_validation_contract(binary_path, 32L)
  expect_identical(binary$data_kind, "image")
  expect_identical(binary$task, "binary")
  expect_identical(binary$backbone, "resnet18")
  expect_identical(
    binary$vision_extractor_profile,
    "dsflower-resnet18-imagenet1k-v1-extractor-v1")
  expect_identical(binary$feature_dim, 512L)
  expect_identical(binary$artifact_format, "pytorch-state-dict-v1")
  expect_match(binary$artifact_sha256, "^[0-9a-f]{64}$")
  expect_null(binary$features)
  expect_null(binary$feature_bounds)

  multiclass_path <- vision_validation_model_fixture(
    model = "pytorch_densenet121", n_classes = 4L, volumetric = TRUE)
  multiclass <- dsFlowerClient:::.resolve_validation_contract(
    multiclass_path, 64L)
  expect_identical(multiclass$task, "multiclass")
  expect_identical(multiclass$backbone, "densenet121_3d")
  expect_true(multiclass$volumetric)
  expect_identical(multiclass$feature_dim, 1024L)
  expect_identical(multiclass$image_size, 128L)
  expect_identical(length(multiclass$target_levels), 4L)

  dense_2d_path <- vision_validation_model_fixture(
    model = "pytorch_densenet121", n_classes = 3L)
  dense_2d_meta_path <- file.path(dense_2d_path, "metadata.json")
  dense_2d_meta <- jsonlite::fromJSON(
    dense_2d_meta_path, simplifyVector = FALSE)
  dense_2d_meta$model_params$image_size <- 28L
  jsonlite::write_json(
    dense_2d_meta, dense_2d_meta_path, auto_unbox = TRUE, null = "list")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(dense_2d_path, 32L),
    "canonical geometry")
  dense_2d_meta$model_params$image_size <- 29L
  jsonlite::write_json(
    dense_2d_meta, dense_2d_meta_path, auto_unbox = TRUE, null = "list")
  expect_identical(
    dsFlowerClient:::.resolve_validation_contract(
      dense_2d_path, 32L)$image_size,
    29L)

  multiclass_meta_path <- file.path(multiclass_path, "metadata.json")
  multiclass_geometry <- jsonlite::fromJSON(
    multiclass_meta_path, simplifyVector = FALSE)
  multiclass_geometry$model_params$image_size <- 127L
  jsonlite::write_json(
    multiclass_geometry, multiclass_meta_path,
    auto_unbox = TRUE, null = "list")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(multiclass_path, 32L),
    "canonical geometry")

  meta_path <- file.path(multiclass_path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$model_params$backbone <- "resnet50"
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(multiclass_path, 32L),
    "unsupported backbone contract")

  missing_profile_path <- vision_validation_model_fixture()
  missing_meta_path <- file.path(missing_profile_path, "metadata.json")
  missing_meta <- jsonlite::fromJSON(missing_meta_path, simplifyVector = FALSE)
  missing_meta$model_params$vision_extractor_profile <- NULL
  jsonlite::write_json(
    missing_meta, missing_meta_path, auto_unbox = TRUE, null = "list")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(missing_profile_path, 32L),
    "unsupported extractor profile")

  wrong_profile_path <- vision_validation_model_fixture()
  wrong_meta_path <- file.path(wrong_profile_path, "metadata.json")
  wrong_meta <- jsonlite::fromJSON(wrong_meta_path, simplifyVector = FALSE)
  wrong_meta$model_params$vision_extractor_profile <-
    "dsflower-resnet18-monai-seed0-extractor-v1"
  jsonlite::write_json(
    wrong_meta, wrong_meta_path, auto_unbox = TRUE, null = "list")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(wrong_profile_path, 32L),
    "unsupported extractor profile")
})

test_that("vision validation rejects unsupported, incomplete and tabular metadata", {
  path <- validation_model_fixture()
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$data_kind <- "image"
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")

  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "successful saved ResNet-18 or DenseNet-121")

  path <- vision_validation_model_fixture(n_classes = 3L)
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$target_levels <- list()
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "one public target level per class")

  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$target_levels <- list("a", "b", "c")
  meta$features <- list("private_pixel")
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "must not contain tabular columns or bounds")
})

test_that("vision target levels are homogeneous, non-empty and type-preserving", {
  with_levels <- function(levels) {
    path <- vision_validation_model_fixture(.local_envir = parent.frame())
    metadata_path <- file.path(path, "metadata.json")
    meta <- jsonlite::fromJSON(metadata_path, simplifyVector = FALSE)
    meta$target_levels <- levels
    jsonlite::write_json(
      meta, metadata_path, auto_unbox = TRUE, null = "list")
    path
  }

  expect_error(
    dsFlowerClient:::.resolve_validation_contract(
      with_levels(list("control", 1L)), 32L),
    "homogeneous public target levels")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(
      with_levels(list("", "case")), 32L),
    "homogeneous public target levels")
  expect_identical(
    dsFlowerClient:::.resolve_validation_contract(
      with_levels(list(FALSE, TRUE)), 32L)$target_levels,
    c(FALSE, TRUE))
  expect_identical(
    dsFlowerClient:::.resolve_validation_contract(
      with_levels(list(1L, 2.5)), 32L)$target_levels,
    c(1, 2.5))
})

test_that("vision artifact bytes are bound into the validation contract", {
  path <- vision_validation_model_fixture()
  first <- dsFlowerClient:::.resolve_validation_contract(path, 32L)
  config <- list(
    "dp-track" = "validation", "validation-model-track" = "neural",
    "validation-task" = first$task, "validation-bins" = first$bins,
    "task-type" = "classification", "loss-name" = first$loss_name,
    "model-spec-b64" = dsFlowerClient:::.spec_to_b64(first$model_spec),
    "num-server-rounds" = 1L, "num-features" = first$feature_dim,
    "num-classes" = first$n_classes, "num-labels" = first$n_labels,
    "target-levels" = first$target_levels, "data_type" = "image",
    "backbone" = first$backbone, "image-size" = first$image_size,
    "vision-extractor-profile" = first$vision_extractor_profile,
    "validation-artifact-format" = first$artifact_format,
    "validation-artifact-sha256" = first$artifact_sha256,
    "validation-artifact-size-bytes" = first$artifact_size_bytes)
  first_contract <- dsFlowerClient:::.validation_contract_sha256(
    config, NULL, "outcome", "row")

  writeBin(charToRaw("tampered"), file.path(path, "model.pt"))
  second <- dsFlowerClient:::.resolve_validation_contract(path, 32L)
  config[["validation-artifact-sha256"]] <- second$artifact_sha256
  config[["validation-artifact-size-bytes"]] <- second$artifact_size_bytes
  second_contract <- dsFlowerClient:::.validation_contract_sha256(
    config, NULL, "outcome", "row")
  expect_false(identical(first$artifact_sha256, second$artifact_sha256))
  expect_false(identical(first_contract, second_contract))
})

test_that("validation requires an explicit data kind contract", {
  path <- validation_model_fixture()
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$data_kind <- NULL
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")

  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "invalid data_kind contract")
})

test_that("validation rejects corrupt public artifacts before DSI or staging", {
  path <- validation_model_fixture()
  reached_cli <- FALSE
  local_mocked_bindings(
    .ensure_client_framework = function(...) TRUE,
    .require_flwr_cli = function() {
      reached_cli <<- TRUE
      stop("CLI must not be reached")
    },
    .package = "dsFlowerClient")
  expect_error(
    ds.flower.validate(
      conns = list(site = TRUE), model = path, target = "outcome",
      symbol = "D"),
    "Saved model artifact preflight failed")
  expect_false(reached_cli)
})

test_that("vision preflight has a bounded cold-start window", {
  timeouts <- numeric()
  local_mocked_bindings(
    .ensure_client_framework = function(...) TRUE,
    .client_python_cmd = function() "python",
    .client_venv_env = function(...) character(),
    .package = "dsFlowerClient")
  local_mocked_bindings(
    run = function(..., timeout) {
      timeouts <<- c(timeouts, timeout)
      list(status = 0L, stderr = "", stdout = "")
    },
    .package = "processx")

  image <- dsFlowerClient:::.resolve_validation_contract(
    vision_validation_model_fixture(), 32L)
  tabular <- dsFlowerClient:::.resolve_validation_contract(
    validation_model_fixture(), 32L)
  expect_no_error(
    dsFlowerClient:::.validate_validation_artifact_preflight(image))
  expect_no_error(
    dsFlowerClient:::.validate_validation_artifact_preflight(tabular))
  expect_identical(timeouts, c(300, 60))
})

test_that("local vision prediction sends only the strict public contract and paths", {
  path <- vision_validation_model_fixture(n_classes = 2L, volumetric = TRUE)
  ensured <- character()
  seen <- NULL
  local_mocked_bindings(
    .ensure_client_framework = function(framework) {
      ensured <<- c(ensured, framework)
      TRUE
    },
    .client_python_cmd = function() "python",
    .client_venv_env = function(...) character(),
    .package = "dsFlowerClient")
  local_mocked_bindings(
    run = function(command, args, ...) {
      data_path <- args[match("--data", args) + 1L]
      config_path <- args[match("--config", args) + 1L]
      seen <<- list(
        paths = jsonlite::fromJSON(data_path),
        config = jsonlite::fromJSON(config_path, simplifyVector = FALSE),
        directory_mode = as.character(file.info(dirname(data_path))$mode),
        modes = as.character(file.info(c(data_path, config_path))$mode),
        args = args)
      list(status = 0L, stderr = "", stdout = "[[0.8,0.2],[0.1,0.9]]")
    },
    .package = "processx")

  got <- ds.flower.predict(
    path, c("first-volume.nii.gz", "second-volume.nii.gz"), type = "prob")

  expect_equal(unname(got), matrix(c(0.8, 0.2, 0.1, 0.9), nrow = 2L,
                                   byrow = TRUE))
  expect_identical(colnames(got), c("class-1", "class-2"))
  expect_identical(ensured, "pytorch_vision")
  expect_identical(seen$paths, c("first-volume.nii.gz", "second-volume.nii.gz"))
  expect_identical(seen$config[["data-kind"]], "image")
  expect_identical(seen$config[["backbone"]], "resnet18_3d")
  expect_identical(
    seen$config[["vision-extractor-profile"]],
    dsFlowerClient:::.vision_extractor_profile("resnet18_3d"))
  expect_true(all(c("--framework", "pytorch_vision") %in% seen$args))
  if (.Platform$OS.type != "windows") {
    expect_identical(seen$directory_mode, "700")
    expect_identical(seen$modes, c("600", "600"))
  }
})

test_that("local vision prediction never reflects private paths in errors", {
  path <- vision_validation_model_fixture()
  private_path <- "/private/participant-42/secret-scan.png"
  local_mocked_bindings(
    .ensure_client_framework = function(...) TRUE,
    .client_python_cmd = function() "python",
    .client_venv_env = function(...) character(),
    .package = "dsFlowerClient")
  local_mocked_bindings(
    run = function(...) list(
      status = 1L, stdout = "",
      stderr = paste("decoder rejected", private_path)),
    .package = "processx")

  error <- expect_error(ds.flower.predict(path, private_path))
  expect_false(grepl(private_path, conditionMessage(error), fixed = TRUE))
})

test_that("oversized vision path JSON is rejected before serialization", {
  path <- vision_validation_model_fixture()
  reached <- character()
  local_mocked_bindings(
    .VISION_LOCAL_MAX_PATH_LIST_BYTES = 16L,
    .ensure_client_framework = function(...) reached <<- c(reached, "setup"),
    .package = "dsFlowerClient")
  local_mocked_bindings(
    write_json = function(...) reached <<- c(reached, "write"),
    .package = "jsonlite")
  local_mocked_bindings(
    run = function(...) reached <<- c(reached, "process"),
    .package = "processx")

  expect_error(
    ds.flower.predict(path, c("long-private-path", "second")),
    "path transport byte ceiling")
  expect_length(reached, 0L)
})

test_that("vision transport write failure cannot reach the predictor", {
  path <- vision_validation_model_fixture()
  reached_process <- FALSE
  local_mocked_bindings(
    .ensure_client_framework = function(...) TRUE,
    .package = "dsFlowerClient")
  local_mocked_bindings(
    write_json = function(...) stop("fixture write failure"),
    .package = "jsonlite")
  local_mocked_bindings(
    run = function(...) reached_process <<- TRUE,
    .package = "processx")

  expect_error(
    ds.flower.predict(path, "private-image"),
    "Could not write private vision prediction inputs")
  expect_false(reached_process)
})

test_that("validation metadata supports up to 1024 classes and labels", {
  path <- validation_model_fixture()
  meta_path <- file.path(path, "metadata.json")
  meta <- jsonlite::fromJSON(meta_path, simplifyVector = FALSE)
  meta$loss_name <- "cross_entropy"
  meta$model_params$n_classes <- 1024L
  meta$model_params$num_labels <- 1024L
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  contract <- dsFlowerClient:::.resolve_validation_contract(path, 32L)
  expect_identical(contract$n_classes, 1024L)
  expect_identical(contract$n_labels, 1024L)

  meta$model_params$n_classes <- 1025L
  jsonlite::write_json(meta, meta_path, auto_unbox = TRUE, null = "null")
  expect_error(
    dsFlowerClient:::.resolve_validation_contract(path, 32L),
    "2, 1024")
})

test_that("validation refuses mixed row/patient estimands", {
  expect_error(
    dsFlowerClient:::.validation_common_privacy_unit(list(
      row_site = list(privacy_unit = "row"),
      patient_site = list(privacy_unit = "patient"))),
    "same row/patient privacy unit")
})

test_that("validation contract SHA-256 has a cross-package canonical wire", {
  config <- list(
    "dp-track" = "validation", "validation-model-track" = "neural",
    "validation-task" = "multiclass", "validation-bins" = 24L,
    "task-type" = "classification", "loss-name" = "cross_entropy",
    "model-spec-b64" = "e30=", "num-server-rounds" = 1L,
    "num-features" = 2L, "num-classes" = 3L, "num-labels" = 2L,
    "feature-bounds" = list(lower = c(0, -5), upper = c(100, 5)),
    "target-levels" = c("a", "b", "c"))
  expect_identical(
    dsFlowerClient:::.validation_contract_sha256(
      config, c("age", "marker"), "outcome", "patient"),
    "ecf31e300ff0087e26799f6a4fbe1894e8f8357e0fd35667936653318b040e3f")
  changed <- config
  changed[["feature-bounds"]]$upper[[1L]] <- 101
  expect_false(identical(
    dsFlowerClient:::.validation_contract_sha256(
      changed, c("age", "marker"), "outcome", "patient"),
    dsFlowerClient:::.validation_contract_sha256(
      config, c("age", "marker"), "outcome", "patient")))
})

test_that("vision validation contract has a cross-package schema-2 wire", {
  config <- list(
    "dp-track" = "validation", "validation-model-track" = "neural",
    "validation-task" = "multiclass", "validation-bins" = 24L,
    "task-type" = "classification", "loss-name" = "cross_entropy",
    "model-spec-b64" = "e30=", "num-server-rounds" = 1L,
    "num-features" = 1024L, "num-classes" = 3L, "num-labels" = 2L,
    "target-levels" = c("class-1", "class-2", "class-3"),
    "data_type" = "image", "backbone" = "densenet121_3d",
    "image-size" = 128L,
    "vision-extractor-profile" =
      "dsflower-densenet121-monai-seed0-extractor-v1",
    "validation-artifact-format" = "pytorch-state-dict-v1",
    "validation-artifact-sha256" = strrep("a", 64L),
    "validation-artifact-size-bytes" = 4096L)
  expect_identical(
    dsFlowerClient:::.validation_contract_sha256(
      config, NULL, "diagnosis", "patient"),
    "95b536d6b8e0902691170a463177bfa64fd3b8dde4c5d002e2e098da94a37959")
  changed_profile <- config
  changed_profile[["vision-extractor-profile"]] <-
    "dsflower-densenet121-imagenet1k-v1-extractor-v1"
  expect_false(identical(
    dsFlowerClient:::.validation_contract_sha256(
      config, NULL, "diagnosis", "patient"),
    dsFlowerClient:::.validation_contract_sha256(
      changed_profile, NULL, "diagnosis", "patient")))
})

test_that("ds.flower.validate stages one validation release and returns pooled metrics", {
  path <- validation_model_fixture()
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_superlink <- client_env$.superlink
  withr::defer(client_env$.superlink <- old_superlink)
  client_env$.superlink <- list(flwr_home = withr::local_tempdir())
  prepared <- NULL
  app_config <- NULL
  release_available <- TRUE

  local_mocked_bindings(
    .validate_validation_artifact_preflight = function(...) TRUE,
    .require_flwr_cli = function() TRUE,
    .validate_dsi_transport_security = function(...) TRUE,
    ds.flower.connect = function(conns, ...) structure(
      list(conns = conns, symbol = "validation_handle"),
      class = "dsflower_connection"),
    .assert_runner_compatibility = function(...) list(
      site_a = list(privacy_unit = "row"),
      site_b = list(privacy_unit = "row")),
    ds.flower.nodes.prepare = function(..., run_config) {
      prepared <<- run_config
      invisible(NULL)
    },
    .build_submission_app = function(sub, config_lines, ...) {
      app_config <<- config_lines
      withr::local_tempdir()
    },
    .ensure_client_framework = function(...) TRUE,
    ds.flower.link.up = function(...) TRUE,
    ds.flower.nodes.ensure = function(...) TRUE,
    .client_flwr_cmd = function() "flwr",
    .client_venv_env = function(...) character(),
    .run_flwr_with_artifact_watchdog = function(..., results_dir) {
      payload <- list(
        pooled_only = TRUE,
        privacy = "node-dp-pooled-postprocessing",
        task = "binary", n_nodes = 2L, available = release_available)
      if (release_available) {
        payload$metrics <- .test_private_metrics("binary")
        payload$metrics$accuracy <- 0.81
        payload$metrics$roc_auc <- 0.87
      }
      jsonlite::write_json(payload,
        file.path(results_dir, "validation.json"), auto_unbox = TRUE)
      list(status = 0L, stdout = "run_id=validation", stderr = "")
    },
    ds.flower.link.down = function(...) TRUE,
    ds.flower.nodes.cleanup = function(...) TRUE,
    ds.flower.disconnect = function(...) TRUE,
    .package = "dsFlowerClient")

  result <- ds.flower.validate(
    list(site_a = TRUE, site_b = TRUE), model = path,
    target = "outcome", symbol = "D", bins = 16L, silent = TRUE)
  expect_s3_class(result, "dsflower_validation")
  expect_true(result$pooled_only)
  expect_true(result$available)
  expect_equal(result$metrics$accuracy, 0.81)
  expect_identical(prepared[["dp-track"]], "validation")
  expect_identical(prepared[["num-server-rounds"]], 1L)
  expect_identical(prepared[["validation-bins"]], 16L)
  expect_match(prepared[["validation-contract-sha256"]], "^[0-9a-f]{64}$")
  expect_identical(prepared[["feature-bounds"]]$upper, c(120, 5))
  expect_true(any(grepl('dp-track = "validation"', app_config, fixed = TRUE)))
  expect_true(any(grepl('loss-name = "bce_logits"', app_config, fixed = TRUE)))
  expect_true(any(grepl(
    paste0('validation-contract-sha256 = "',
           prepared[["validation-contract-sha256"]], '"'),
    app_config, fixed = TRUE)))
  expect_true(any(grepl("num-server-rounds = 1", app_config, fixed = TRUE)))
  expect_false("per_node" %in% names(result))

  release_available <- FALSE
  unavailable <- ds.flower.validate(
    list(site_a = TRUE, site_b = TRUE), model = path,
    target = "outcome", symbol = "D", bins = 16L, silent = TRUE)
  expect_s3_class(unavailable, "dsflower_validation")
  expect_false(unavailable$available)
  expect_null(unavailable$metrics)
  expect_output(print(unavailable), "no complete private metric release")
})

test_that("vision validation carries only canonical image and artifact pins", {
  path <- vision_validation_model_fixture(
    model = "pytorch_densenet121", n_classes = 3L, volumetric = TRUE)
  contract <- dsFlowerClient:::.resolve_validation_contract(path, 24L)
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_superlink <- client_env$.superlink
  withr::defer(client_env$.superlink <- old_superlink)
  client_env$.superlink <- list(flwr_home = withr::local_tempdir())
  prepared <- NULL
  prepared_features <- "not-called"
  app_config <- NULL
  app_vision <- FALSE
  ensured_framework <- NULL

  local_mocked_bindings(
    .validate_validation_artifact_preflight = function(...) TRUE,
    .require_flwr_cli = function() TRUE,
    .validate_dsi_transport_security = function(...) TRUE,
    ds.flower.connect = function(conns, ...) structure(
      list(conns = conns, symbol = "validation_handle"),
      class = "dsflower_connection"),
    .assert_runner_compatibility = function(...) list(
      site_a = list(privacy_unit = "row")),
    ds.flower.nodes.prepare = function(..., feature_columns, run_config) {
      prepared <<- run_config
      prepared_features <<- feature_columns
      invisible(NULL)
    },
    .build_submission_app = function(sub, config_lines, vision, ...) {
      app_config <<- config_lines
      app_vision <<- vision
      withr::local_tempdir()
    },
    .ensure_client_framework = function(framework) {
      ensured_framework <<- framework
      TRUE
    },
    ds.flower.link.up = function(...) TRUE,
    ds.flower.nodes.ensure = function(...) TRUE,
    .client_flwr_cmd = function() "flwr",
    .client_venv_env = function(...) character(),
    .run_flwr_with_artifact_watchdog = function(..., results_dir) {
      jsonlite::write_json(list(
        pooled_only = TRUE,
        privacy = "node-dp-pooled-postprocessing",
        task = "multiclass", n_nodes = 1L, available = TRUE,
        metrics = .test_private_metrics("multiclass")),
        file.path(results_dir, "validation.json"), auto_unbox = TRUE,
        null = "null")
      list(status = 0L, stdout = "", stderr = "")
    },
    ds.flower.link.down = function(...) TRUE,
    ds.flower.nodes.cleanup = function(...) TRUE,
    ds.flower.disconnect = function(...) TRUE,
    .package = "dsFlowerClient")

  result <- ds.flower.validate(
    list(site_a = TRUE), model = path, target = "diagnosis",
    symbol = "images", bins = 24L, silent = TRUE)

  expect_true(result$available)
  expect_identical(prepared_features, NULL)
  expect_identical(prepared[["data_type"]], "image")
  expect_identical(prepared[["backbone"]], "densenet121_3d")
  expect_identical(prepared[["image-size"]], 128L)
  expect_identical(
    prepared[["vision-extractor-profile"]],
    "dsflower-densenet121-monai-seed0-extractor-v1")
  expect_identical(prepared[["num-features"]], 1024L)
  expect_identical(
    prepared[["validation-artifact-sha256"]], contract$artifact_sha256)
  expect_null(prepared[["feature-bounds"]])
  expect_true(app_vision)
  expect_identical(ensured_framework, "pytorch_vision")
  expect_true(any(grepl('data-kind = "image"', app_config, fixed = TRUE)))
  expect_true(any(grepl('backbone = "densenet121_3d"',
                        app_config, fixed = TRUE)))
  expect_true(any(grepl(
    'vision-extractor-profile = "dsflower-densenet121-monai-seed0-extractor-v1"',
    app_config, fixed = TRUE)))
  expect_true(any(grepl(
    paste0('validation-artifact-sha256 = "',
           contract$artifact_sha256, '"'), app_config, fixed = TRUE)))
})

test_that("XGBoost validation uses the native app with exact ephemeral pins", {
  path <- xgboost_validation_model_fixture()
  contract <- dsFlowerClient:::.resolve_validation_contract(path, 16L)
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_superlink <- client_env$.superlink
  withr::defer(client_env$.superlink <- old_superlink)
  client_env$.superlink <- list(flwr_home = withr::local_tempdir())
  prepared <- NULL
  submission <- NULL
  app_config <- NULL
  framework_called <- FALSE
  node_backend <- "not-called"

  local_mocked_bindings(
    .validate_validation_artifact_preflight = function(...) TRUE,
    .require_flwr_cli = function() TRUE,
    .validate_dsi_transport_security = function(...) TRUE,
    ds.flower.connect = function(conns, ...) structure(
      list(conns = conns, symbol = "validation_handle"),
      class = "dsflower_connection"),
    .assert_runner_compatibility = function(...) list(
      site_a = list(privacy_unit = "row"),
      site_b = list(privacy_unit = "row")),
    ds.flower.nodes.prepare = function(..., run_config) {
      prepared <<- run_config
      invisible(NULL)
    },
    .build_submission_app = function(sub, config_lines, ...) {
      submission <<- sub
      app_config <<- config_lines
      withr::local_tempdir()
    },
    .ensure_client_framework = function(...) {
      framework_called <<- TRUE
      stop("native validation must not provision torch")
    },
    ds.flower.link.up = function(...) TRUE,
    ds.flower.nodes.ensure = function(..., torch_backend) {
      node_backend <<- torch_backend
      TRUE
    },
    .client_flwr_cmd = function() "flwr",
    .client_venv_env = function(...) character(),
    .run_flwr_with_artifact_watchdog = function(..., results_dir) {
      metrics <- .test_private_metrics("binary")
      metrics$roc_auc <- 0.8
      jsonlite::write_json(list(
        pooled_only = TRUE,
        privacy = "node-dp-pooled-postprocessing",
        task = "binary", n_nodes = 2L, available = TRUE,
        metrics = metrics),
        file.path(results_dir, "validation.json"), auto_unbox = TRUE)
      list(status = 0L, stdout = "run_id=native-validation", stderr = "")
    },
    ds.flower.link.down = function(...) TRUE,
    ds.flower.nodes.cleanup = function(...) TRUE,
    ds.flower.disconnect = function(...) TRUE,
    .package = "dsFlowerClient")

  result <- ds.flower.validate(
    list(site_a = TRUE, site_b = TRUE), model = path,
    target = "outcome", symbol = "D", bins = 16L, silent = TRUE)

  expect_s3_class(result, "dsflower_validation")
  expect_true(result$available)
  expect_equal(result$metrics$roc_auc, 0.8)
  expect_identical(
    result$model_training_privacy,
    "direct-dp-training-postprocessing")
  expect_identical(submission$track, "native_tree_validation")
  expect_false(framework_called)
  expect_null(node_backend)
  expect_identical(prepared[["validation-model-track"]], "native_tree")
  expect_identical(prepared[["validation-artifact-format"]],
                   contract$artifact_format)
  expect_identical(prepared[["validation-artifact-sha256"]],
                   contract$artifact_sha256)
  expect_identical(prepared[["validation-profile-sha256"]],
                   contract$prediction_profile$sha256)
  expect_identical(prepared[["validation-public-schema-sha256"]],
                   contract$public_schema_sha256)
  expect_match(prepared[["validation-contract-sha256"]], "^[0-9a-f]{64}$")
  expect_false(any(c(
    "model-spec-b64", "validation-model-path-b64",
    "validation-profile-path-b64") %in% names(prepared)))
  expect_true(any(grepl(
    paste0('validation-artifact-sha256 = "', contract$artifact_sha256, '"'),
    app_config, fixed = TRUE)))
  expect_true(any(grepl(
    paste0('validation-profile-sha256 = "',
           contract$prediction_profile$sha256, '"'),
    app_config, fixed = TRUE)))
  expect_true(any(grepl("validation-model-path-b64", app_config, fixed = TRUE)))
  expect_true(any(grepl("validation-profile-path-b64", app_config, fixed = TRUE)))
  expect_false(any(grepl("model-spec-b64", app_config, fixed = TRUE)))
})

test_that("pooled validation reader refuses a per-node result", {
  path <- withr::local_tempdir()
  jsonlite::write_json(list(
    pooled_only = TRUE, privacy = "node-dp-pooled-postprocessing",
    task = "binary", n_nodes = 1L, available = TRUE,
    metrics = list(accuracy = 0.5),
    per_node = list(site = list(accuracy = 1))),
    file.path(path, "validation.json"), auto_unbox = TRUE)
  expect_error(
    dsFlowerClient:::.read_private_validation_result(path),
    "pooled-only privacy contract")

  jsonlite::write_json(list(
    pooled_only = TRUE, privacy = "node-dp-pooled-postprocessing",
    task = "binary", n_nodes = 1L, available = TRUE,
    metrics = list(accuracy = 0.5),
    statistics = list(histogram = c(1, 2))),
    file.path(path, "validation.json"), auto_unbox = TRUE)
  expect_error(
    dsFlowerClient:::.read_private_validation_result(path),
    "pooled-only privacy contract")
})

test_that("unavailable validation is explicit and never contains fake metrics", {
  path <- withr::local_tempdir()
  jsonlite::write_json(list(
    pooled_only = TRUE, privacy = "node-dp-pooled-postprocessing",
    task = "binary", n_nodes = 2L, available = FALSE),
    file.path(path, "validation.json"), auto_unbox = TRUE)
  result <- dsFlowerClient:::.read_private_validation_result(path)
  expect_false(result$available)
  expect_null(result$metrics)

  jsonlite::write_json(list(
    pooled_only = TRUE, privacy = "node-dp-pooled-postprocessing",
    task = "binary", n_nodes = 2L, available = FALSE,
    metrics = list(accuracy = 0)),
    file.path(path, "validation.json"), auto_unbox = TRUE)
  expect_error(
    dsFlowerClient:::.read_private_validation_result(path),
    "pooled-only privacy contract")
})
