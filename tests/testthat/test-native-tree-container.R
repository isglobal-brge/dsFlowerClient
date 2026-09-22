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
  mutations <- list(
    whitespace = paste0(" ", original),
    newline = paste0(original, "\n"),
    duplicate = sub('"version":1}', '"version":1,"version":1}', original, fixed = TRUE),
    reordered = sub('"aggregation":"mean_prediction","contract":"dsflower-forest-ensemble-v1"',
      '"contract":"dsflower-forest-ensemble-v1","aggregation":"mean_prediction"', original, fixed = TRUE),
    float_spelling = sub("0.0038312680927895396", "0.00383126809278953960", original, fixed = TRUE),
    version = sub('"task":"binary","version":1}', '"task":"binary","version":1.0}', original, fixed = TRUE),
    extra_field = sub('"aggregation":', '"added":0,"aggregation":', original, fixed = TRUE),
    engine = sub('"engine":"random_forest"', '"engine":"extra_trees"', original, fixed = TRUE),
    task = sub('"task":"binary","version":1}', '"task":"regression","version":1}', original, fixed = TRUE),
    contract = sub("dsflower-forest-ensemble-v1", "wrong", original, fixed = TRUE))
  for (name in names(mutations)) {
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
