.native_tree_expected_json <- paste0(
  '{"contract":"dsflower-native-tree-request-v1","engine":"xgboost",',
  '"mode":"native-tight","task":"binary","public_schema":{',
  '"version":1,"features":["age","marker"],"lower":[0.0,-5.0],',
  '"upper":[100.0,5.0],"cuts":[[18.0,40.0,65.0],[-1.0,0.0,1.0]],',
  '"target":{"name":"outcome","kind":"binary","lower":0.0,"upper":1.0},',
  '"sha256":"a24299d5ccba8a1af70f0c2d5afa06937d9632a75bc69d20d3e1520ec96d5733"},',
  '"parameters":[{"name":"max_depth","type":"integer","value":6},',
  '{"name":"monotone_constraints","type":"integer_array","value":[1,-1]},',
  '{"name":"subsample","type":"number","value":0.8}],',
  '"resources":{"max_features":4096,"max_trees":4096,"max_depth":8,',
  '"max_bins":8,"max_threads":32,"memory_mb":32768,',
  '"timeout_seconds":21600}}')

.native_tree_binary_target <- function() {
  list(name = "outcome", kind = "binary", lower = 0, upper = 1)
}

.native_tree_continuous_target <- function() {
  list(name = "outcome", kind = "continuous", lower = -10, upper = 10)
}

.build_native_tree_fixture <- function(...) {
  dsFlowerClient:::.build_native_tree_manifest(
    engine = "xgboost", mode = "native-tight", task = "binary",
    features = c("age", "marker"),
    bounds = list(lower = c(0, -5), upper = c(100, 5)),
    target = .native_tree_binary_target(),
    cuts = list(c(18, 40, 65), c(-1, 0, 1)),
    parameters = list(
      subsample = 0.8,
      monotone_constraints = c(1L, -1L),
      max_depth = 6L),
    resources = list(max_bins = 8L, max_depth = 8L), ...)
}

test_that("native-tree builder emits the canonical cross-package wire", {
  manifest <- .build_native_tree_fixture()

  expect_identical(manifest$json, .native_tree_expected_json)
  expect_identical(
    manifest$sha256,
    "6b80230e762a3ab73c3f4d655ae3b3ff8304d05a6076a3975f065a227ee177bb")
  expect_identical(
    rawToChar(jsonlite::base64_dec(manifest$b64)), manifest$json)
  expect_identical(
    vapply(manifest$value$parameters, `[[`, character(1), "type"),
    c("integer", "integer_array", "number"))

  reordered <- dsFlowerClient:::.build_native_tree_manifest(
    engine = "xgboost", mode = "native-tight", task = "binary",
    features = c("age", "marker"),
    bounds = list(upper = c(100, 5), lower = c(0, -5)),
    target = .native_tree_binary_target(),
    cuts = list(c(18, 40, 65), c(-1, 0, 1)),
    parameters = list(
      max_depth = 6L, subsample = 0.8,
      monotone_constraints = c(1L, -1L)),
    resources = list(max_depth = 8L, max_bins = 8L))
  expect_identical(reordered$sha256, manifest$sha256)
})

test_that("native-tree wire preserves one-element arrays", {
  manifest <- dsFlowerClient:::.build_native_tree_manifest(
    "xgboost", "native-tight", "binary", "x",
    list(lower = 0, upper = 1), .native_tree_binary_target(), list(0.5),
    parameters = list(monotone_constraints = list(
      type = "integer_array", value = 1L)))

  expect_match(manifest$json, '"features":\\["x"\\]')
  expect_match(manifest$json, '"lower":\\[0.0\\]')
  expect_match(manifest$json, '"cuts":\\[\\[0.5\\]\\]')
  expect_match(manifest$json, '"value":\\[1\\]')
  expect_identical(
    manifest$value$public_schema$sha256,
    "fb4a74228657414d935f4dc4f068f0c53f83743d237c100db49c40dab9c622b7")
})

test_that("native-tight requires public cuts and blocks adaptive code paths", {
  expect_error(
    dsFlowerClient:::.build_native_tree_manifest(
      "xgboost", "native-tight", "binary", "x",
      list(lower = 0, upper = 1), .native_tree_binary_target()),
    "requires public cuts")

  forbidden <- c("eval_metric", "early_stopping_rounds", "seed", "tree_method")
  for (name in forbidden) {
    params <- stats::setNames(list(if (name == "early_stopping_rounds") 10L else "x"),
                             name)
    expect_error(
      dsFlowerClient:::.build_native_tree_manifest(
        "xgboost", "native-tight", "binary", "x",
        list(lower = 0, upper = 1), .native_tree_binary_target(),
        list(0.5), params),
      "native-tight forbids")
  }
  for (name in c(
      "callbacks", "custom_objective", "custom_objective_fn", "plugin_path",
      "max_rows_per_unit", "unit_canonicalization")) {
    expect_error(
      dsFlowerClient:::.build_native_tree_manifest(
        "xgboost", "synopsis-flex", "binary", "x",
        list(lower = 0, upper = 1),
        .native_tree_binary_target(),
        parameters = stats::setNames(list("x"), name)),
      "reserved by the server")
  }

  flex <- dsFlowerClient:::.build_native_tree_manifest(
    "catboost", "synopsis-flex", "regression", "x",
    list(lower = 0, upper = 1),
    .native_tree_continuous_target(),
    parameters = list(
      objective = "RMSE", eval_metric = "MAE",
      early_stopping_rounds = 20L))
  expect_identical(flex$value$mode, "synopsis-flex")
  expect_null(flex$value$public_schema$cuts)
})

test_that("native-tree schema hashes and resource ceilings fail closed", {
  manifest <- .build_native_tree_fixture()
  expect_error(
    dsFlowerClient:::.build_native_tree_manifest(
      "xgboost", "native-tight", "binary", c("age", "marker"),
      list(lower = c(0, -5), upper = c(100, 5)),
      .native_tree_binary_target(),
      list(c(18, 40, 65), c(-1, 0, 1)),
      schema_sha256 = strrep("0", 64L)),
    "schema SHA-256")
  expect_error(
    dsFlowerClient:::.build_native_tree_manifest(
      "xgboost", "native-tight", "binary", "x",
      list(lower = 0, upper = 1), .native_tree_binary_target(),
      list(c(0.5, 0.4))),
    "strictly increasing")
  expect_error(
    dsFlowerClient:::.build_native_tree_manifest(
      "xgboost", "native-tight", "binary", "x",
      list(lower = 0, upper = 1), .native_tree_binary_target(), list(0.5),
      parameters = list(n_estimators = 101L),
      resources = list(max_trees = 100L)),
    "exceeds resources\\$max_trees")
  expect_error(
    dsFlowerClient:::.build_native_tree_manifest(
      "xgboost", "native-tight", "binary", "x",
      list(lower = 0, upper = 1), .native_tree_binary_target(), list(0.5),
      parameters = list(epsilon = 1)),
    "reserved by the server")
  expect_match(manifest$value$public_schema$sha256, "^[0-9a-f]{64}$")
})

test_that("native-tree target schema is task-bound", {
  invalid_binary <- list(
    list(name = "outcome", kind = "continuous", lower = 0, upper = 1),
    list(name = "outcome", kind = "binary", lower = -1, upper = 1),
    list(name = "x", kind = "binary", lower = 0, upper = 1))
  for (target in invalid_binary) {
    expect_error(
      dsFlowerClient:::.build_native_tree_manifest(
        "xgboost", "native-tight", "binary", "x",
        list(lower = 0, upper = 1), target, list(0.5)),
      "target|binary task")
  }
  regression <- dsFlowerClient:::.build_native_tree_manifest(
    "lightgbm", "synopsis-flex", "regression", "x",
    list(lower = 0, upper = 1), .native_tree_continuous_target())
  expect_identical(regression$value$public_schema$target$kind, "continuous")

  changed <- .build_native_tree_fixture()
  expect_false(identical(
    changed$value$public_schema$sha256,
    dsFlowerClient:::.build_native_tree_manifest(
      "xgboost", "native-tight", "binary", c("age", "marker"),
      list(lower = c(0, -5), upper = c(100, 5)),
      list(name = "case", kind = "binary", lower = 0, upper = 1),
      list(c(18, 40, 65), c(-1, 0, 1)))$value$public_schema$sha256))
})

test_that("native-tree request has bounded cuts and canonical bytes", {
  expect_error(
    dsFlowerClient:::.build_native_tree_manifest(
      "xgboost", "native-tight", "binary", "x",
      list(lower = 0, upper = 20000), .native_tree_binary_target(),
      list(seq_len(16385L)), resources = list(max_bins = 20000L)),
    "16384-cut contract cap")

  values <- stats::setNames(
    rep(list(strrep("x", 512L)), 128L),
    sprintf("parameter_%03d", seq_len(128L)))
  expect_error(
    dsFlowerClient:::.build_native_tree_manifest(
      "xgboost", "synopsis-flex", "binary", "x",
      list(lower = 0, upper = 1), .native_tree_binary_target(),
      parameters = values),
    "exceeds 65536 bytes")
})
