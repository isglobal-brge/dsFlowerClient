native_tree_training_fixture <- function() {
  path <- withr::local_tempdir(.local_envir = parent.frame())
  spec <- dsFlowerClient:::.native_tree_release_spec("random_forest")
  file.copy(test_path("fixtures", "native-tree-training.json"),
            file.path(path, spec$artifact_file))
  file.copy(test_path("fixtures", "native-tree-training.profile.json"),
            file.path(path, spec$profile_file))
  profile <- jsonlite::fromJSON(file.path(path, spec$profile_file),
                                simplifyVector = FALSE)
  recipe <- list(model = list(engine = "random_forest", task = "binary"),
    native_tree_request_b64 = profile$native_tree_request_b64,
    native_tree_request_sha256 = profile$native_tree_request_sha256,
    public_schema_sha256 = profile$public_schema_sha256)
  meta <- list(public_schema_sha256 = profile$public_schema_sha256,
    artifact = c(list(file = spec$artifact_file), profile$artifact),
    sanitization = dsFlowerClient:::.native_tree_sanitization_attestation("random_forest"))
  list(path = path, recipe = recipe, meta = meta)
}

test_that("real native training output survives canonical release validation", {
  fixture <- native_tree_training_fixture()
  path <- file.path(fixture$path, fixture$meta$artifact$file)
  bytes <- readBin(path, "raw", n = file.info(path)$size)
  value <- jsonlite::fromJSON(rawToChar(bytes), simplifyVector = FALSE)
  # The generated two-leaf forest exercises the 0.5.0 precision failure.
  expect_false(identical(bytes, dsFlowerClient:::.native_tree_json(value)))
  expect_identical(dsFlowerClient:::.native_tree_ensemble_json(bytes), bytes)
  expect_no_error(dsFlowerClient:::.native_tree_release_metadata(
    fixture$recipe, fixture$path))
  expect_no_error(dsFlowerClient:::.validate_native_tree_ensemble_artifact(
    fixture$meta, fixture$path, "binary", "random_forest"))
})

test_that("native containers retain exact canonical bytes and identity checks", {
  fixture <- native_tree_training_fixture()
  path <- file.path(fixture$path, fixture$meta$artifact$file)
  original <- readChar(path, file.info(path)$size, useBytes = TRUE)
  first_leaf <- regmatches(original, regexpr('"leaf_values":\\[[^,]+', original))
  spelling <- sub('"leaf_values":\\[', "", first_leaf)
  noncanonical <- if (grepl("[eE]", spelling)) {
    sub("([eE])", "0\\1", spelling)
  } else if (grepl(".", spelling, fixed = TRUE)) paste0(spelling, "0") else paste0(spelling, ".0")
  expect_equal(as.numeric(spelling), as.numeric(noncanonical))
  mutations <- list(
    whitespace = paste0(" ", original),
    newline = paste0(original, "\n"),
    duplicate = sub('"version":1}', '"version":1,"version":1}', original, fixed = TRUE),
    reordered = sub('"aggregation":"mean_prediction","contract":"dsflower-forest-ensemble-v1"',
      '"contract":"dsflower-forest-ensemble-v1","aggregation":"mean_prediction"', original, fixed = TRUE),
    float_spelling = sub(first_leaf, paste0('"leaf_values":[', noncanonical), original, fixed = TRUE),
    version = sub('"task":"binary","version":1}', '"task":"binary","version":1.0}', original, fixed = TRUE),
    extra_field = sub('"aggregation":', '"added":0,"aggregation":', original, fixed = TRUE),
    engine = sub('"engine":"random_forest"', '"engine":"extra_trees"', original, fixed = TRUE),
    task = sub('"task":"binary","version":1}', '"task":"regression","version":1}', original, fixed = TRUE),
    contract = sub("dsflower-forest-ensemble-v1", "wrong", original, fixed = TRUE))
  for (name in names(mutations)) {
    expect_false(identical(mutations[[name]], original), info = name)
    bytes <- charToRaw(mutations[[name]])
    writeBin(bytes, path)
    meta <- fixture$meta
    meta$artifact$size_bytes <- length(bytes)
    meta$artifact$sha256 <- digest::digest(bytes, algo = "sha256", serialize = FALSE)
    expect_error(dsFlowerClient:::.validate_native_tree_ensemble_artifact(
      meta, fixture$path, "binary", "random_forest"), "canonical container", info = name)
  }
  writeBin(charToRaw(original), path)
  for (name in c("hash", "size", "schema", "attestation")) {
    meta <- fixture$meta
    if (name == "hash") meta$artifact$sha256 <- strrep("0", 64)
    if (name == "size") meta$artifact$size_bytes <- 64 * 1024^2 + 1
    if (name == "schema") meta$public_schema_sha256 <- strrep("0", 64)
    if (name == "attestation") meta$sanitization$contains_raw_records <- TRUE
    expect_error(dsFlowerClient:::.validate_native_tree_ensemble_artifact(
      meta, fixture$path, "binary", "random_forest"),
      "SHA-256|size|attestation", info = name)
  }
})

test_that("run wrapper returns the verified released native bytes across replay", {
  fixture <- native_tree_training_fixture()
  client_env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old_superlink <- client_env$.superlink
  withr::defer(client_env$.superlink <- old_superlink)
  client_env$.superlink <- list(process = list(is_alive = function() TRUE),
                              flwr_home = withr::local_tempdir())
  recipe <- fixture$recipe
  recipe$model <- c(recipe$model, list(
    name = "random_forest", framework = "native_tree", track = "native_tree"))
  recipe$strategy <- list(name = "mean_prediction", params = list())
  recipe$num_rounds <- 1L
  recipe$data_kind <- "tabular"
  recipe$features <- "x"
  recipe$feature_lower <- -1
  recipe$feature_upper <- 1
  recipe$target_levels <- c("no", "yes")
  class(recipe) <- "dsflower_recipe"
  output_root <- withr::local_tempdir()
  count <- 0L
  tamper <- FALSE
  available <- TRUE
  local_mocked_bindings(
    .require_flwr_cli = function() TRUE,
    .client_flwr_cmd = function() "flwr",
    .client_venv_env = function(...) character(),
    .run_flwr_with_artifact_watchdog = function(..., results_dir) {
      count <<- count + 1L
      if (available) {
        file.copy(list.files(fixture$path, full.names = TRUE), results_dir)
        if (tamper) {
          path <- file.path(results_dir, fixture$meta$artifact$file)
          bytes <- readBin(path, "raw", n = file.info(path)$size)
          writeBin(c(bytes, charToRaw("\n")), path)
        }
      }
      jsonlite::write_json(data.frame(round = 1L, available = available),
        file.path(results_dir, "history.json"), auto_unbox = TRUE)
      list(status = 0L, stdout = paste0("run_id=replay-", count), stderr = "")
    }, .package = "dsFlowerClient")
  run <- function(name) ds.flower.run.start(
    recipe, conns = list(site = TRUE), app_dir = withr::local_tempdir(),
    output_dir = output_root, output_name = name, silent = TRUE)
  first <- run("first")
  replay <- run("replay")
  expect_true(first$available)
  expect_null(first$weights)
  expect_type(first$artifact$sha256, "character")
  expect_identical(first$artifact, replay$artifact)
  expect_identical(first$sanitization, replay$sanitization)
  expect_false(identical(first$run_id, replay$run_id))
  expect_false(identical(first$output_dir, replay$output_dir))
  for (result in list(first, replay)) {
    path <- file.path(result$output_dir, result$artifact$file)
    bytes <- readBin(path, "raw", n = file.info(path)$size)
    expect_identical(result$artifact$sha256,
      digest::digest(bytes, algo = "sha256", serialize = FALSE))
    expect_identical(result$artifact$size_bytes, as.integer(length(bytes)))
    saved <- readRDS(result$saved_path)
    expect_identical(result$artifact, saved$artifact)
    expect_identical(result$sanitization, saved$sanitization)
  }
  tamper <- TRUE
  expect_error(run("tampered"), "canonical container")
  expect_false(dir.exists(file.path(output_root, "tampered")))
  available <- FALSE
  unavailable <- run("unavailable")
  expect_false(unavailable$available)
  expect_null(unavailable$artifact)
  expect_null(unavailable$sanitization)
})
