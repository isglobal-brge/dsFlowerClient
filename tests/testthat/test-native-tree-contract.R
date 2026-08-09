.native_tree_expected_json <- paste0(
  '{"contract":"dsflower-native-tree-request-v1","engine":"xgboost",',
  '"mode":"native-tight","task":"binary","public_schema":{',
  '"version":1,"features":["age","marker"],"lower":[0.0,-5.0],',
  '"upper":[100.0,5.0],"cuts":[[18.0,40.0,65.0],[-1.0,0.0,1.0]],',
  '"target":{"name":"outcome","kind":"binary","levels":',
  '[{"type":"string","value":"control"},{"type":"string","value":"case"}],',
  '"lower":0.0,"upper":1.0},',
  '"sha256":"77a6e8d46a174381b8b4da168b833b2ee75f09f8ca8ac55f2c954be642ba9073"},',
  '"parameters":[{"name":"learning_rate","type":"number","value":0.25},',
  '{"name":"max_delta_step","type":"number","value":1.0},',
  '{"name":"max_depth","type":"integer","value":2},',
  '{"name":"min_child_weight","type":"number","value":1.0},',
  '{"name":"min_split_loss","type":"number","value":0.0},',
  '{"name":"num_boost_round","type":"integer","value":8},',
  '{"name":"reg_alpha","type":"number","value":0.0},',
  '{"name":"reg_lambda","type":"number","value":1.0}],',
  '"resources":{"max_features":2,"max_trees":8,"max_depth":2,',
  '"max_bins":4,"max_threads":32,"memory_mb":32768,',
  '"timeout_seconds":21600}}')

.native_tree_binary_target <- function() {
  list(
    name = "outcome", kind = "binary",
    levels = list(
      list(type = "boolean", value = FALSE),
      list(type = "boolean", value = TRUE)),
    lower = 0, upper = 1)
}

.native_tree_continuous_target <- function() {
  list(name = "outcome", kind = "continuous", levels = NULL,
       lower = -10, upper = 10)
}

.build_native_tree_fixture <- function(target_levels = c("control", "case"), ...) {
  model <- ds.flower.model.xgboost()
  dsFlowerClient:::.build_xgboost_request(
    params = model$params,
    features = c("age", "marker"),
    feature_bounds = list(lower = c(0, -5), upper = c(100, 5)),
    target_name = "outcome",
    target_levels = target_levels,
    feature_cuts = list(c(18, 40, 65), c(-1, 0, 1)), ...)
}

test_that("native-tree builder emits the canonical cross-package wire", {
  manifest <- .build_native_tree_fixture()

  expect_identical(manifest$json, .native_tree_expected_json)
  expect_identical(
    manifest$sha256,
    "193390a92a076bf9d4cdac0686e6542990b5948809cdc8f1dbbc9ccaac787692")
  expect_identical(
    rawToChar(jsonlite::base64_dec(manifest$b64)), manifest$json)
  expect_identical(
    vapply(manifest$value$parameters, `[[`, character(1), "type"),
    c("number", "number", "integer", "number", "number", "integer",
      "number", "number"))

  reordered <- dsFlowerClient:::.build_xgboost_request(
    params = ds.flower.model(
      "xgboost", gamma = 0, eta = 0.25, n_estimators = 8L)$params,
    features = c("age", "marker"),
    feature_bounds = list(upper = c(100, 5), lower = c(0, -5)),
    target_name = "outcome",
    target_levels = c("control", "case"),
    feature_cuts = list(c(18, 40, 65), c(-1, 0, 1)))
  expect_identical(reordered$sha256, manifest$sha256)
})

test_that("native-tree wire preserves one-element arrays", {
  manifest <- dsFlowerClient:::.build_native_tree_manifest(
    "catboost", "native-tight", "binary", "x",
    list(lower = 0, upper = 1), .native_tree_binary_target(), list(0.5),
    parameters = list(one_element_array = list(
      type = "integer_array", value = 1L)))

  expect_match(manifest$json, '"features":\\["x"\\]')
  expect_match(manifest$json, '"lower":\\[0.0\\]')
  expect_match(manifest$json, '"cuts":\\[\\[0.5\\]\\]')
  expect_match(manifest$json, '"value":\\[1\\]')
  expect_identical(
    manifest$value$public_schema$sha256,
    "c42dbee96310e9be8fa0d61ce747716e20057146687ebd5e49a1c0f79fab8628")
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
        "catboost", "native-tight", "binary", "x",
        list(lower = 0, upper = 1),
        .native_tree_binary_target(),
        cuts = list(0.5),
        parameters = stats::setNames(list("x"), name)),
      "reserved by the server")
  }

  expect_error(
    dsFlowerClient:::.build_native_tree_manifest(
      "catboost", "unsupported", "regression", "x",
      list(lower = 0, upper = 1),
      .native_tree_continuous_target(), list(0.5)),
    "Unsupported tree mode")
})

test_that("native-tree schema hashes and resource ceilings fail closed", {
  manifest <- .build_native_tree_fixture()
  expect_error(
    dsFlowerClient:::.build_native_tree_manifest(
      "catboost", "native-tight", "binary", c("age", "marker"),
      list(lower = c(0, -5), upper = c(100, 5)),
      .native_tree_binary_target(),
      list(c(18, 40, 65), c(-1, 0, 1)),
      schema_sha256 = strrep("0", 64L)),
    "schema SHA-256")
  expect_error(
    dsFlowerClient:::.build_native_tree_manifest(
      "catboost", "native-tight", "binary", "x",
      list(lower = 0, upper = 1), .native_tree_binary_target(),
      list(c(0.5, 0.4))),
    "strictly increasing")
  expect_error(
    dsFlowerClient:::.build_native_tree_manifest(
      "catboost", "native-tight", "binary", "x",
      list(lower = 0, upper = 1), .native_tree_binary_target(), list(0.5),
      parameters = list(n_estimators = 101L),
      resources = list(max_trees = 100L)),
    "exceeds resources\\$max_trees")
  expect_error(
    dsFlowerClient:::.build_native_tree_manifest(
      "catboost", "native-tight", "binary", "x",
      list(lower = 0, upper = 1), .native_tree_binary_target(), list(0.5),
      parameters = list(epsilon = 1)),
    "reserved by the server")
  expect_match(manifest$value$public_schema$sha256, "^[0-9a-f]{64}$")
})

test_that("native-tree target schema is task-bound", {
  invalid_binary <- list(
    list(name = "outcome", kind = "continuous",
         levels = .native_tree_binary_target()$levels, lower = 0, upper = 1),
    list(name = "outcome", kind = "binary",
         levels = .native_tree_binary_target()$levels, lower = -1, upper = 1),
    list(name = "x", kind = "binary",
         levels = .native_tree_binary_target()$levels, lower = 0, upper = 1))
  for (target in invalid_binary) {
    expect_error(
      dsFlowerClient:::.build_native_tree_manifest(
        "xgboost", "native-tight", "binary", "x",
        list(lower = 0, upper = 1), target, list(0.5)),
      "target|binary task")
  }
  regression <- dsFlowerClient:::.build_native_tree_manifest(
    "lightgbm", "native-tight", "regression", "x",
    list(lower = 0, upper = 1), .native_tree_continuous_target(), list(0.5))
  expect_identical(regression$value$public_schema$target$kind, "continuous")

  changed <- .build_native_tree_fixture()
  renamed <- dsFlowerClient:::.build_xgboost_request(
    ds.flower.model.xgboost()$params, c("age", "marker"),
    list(lower = c(0, -5), upper = c(100, 5)),
    list(c(18, 40, 65), c(-1, 0, 1)), "case",
    target_levels = c("control", "case"))
  expect_false(identical(
    changed$value$public_schema$sha256,
    renamed$value$public_schema$sha256))
})

test_that("XGBoost request v1 uses an exact typed allowlist", {
  model <- ds.flower.model.xgboost()
  expect_identical(model$track, "native_tree")
  expect_identical(model$framework, "xgboost")
  expect_identical(model$loss, "binary_logistic")
  expect_false(ds.flower.list_models()$available[
    ds.flower.list_models()$name == "xgboost"])
  expect_error(ds.flower.model("xgboost", subsample = 0.8),
               "Unknown parameter.*subsample")
  expect_error(ds.flower.model("xgboost", objective = "binary:logistic"),
               "Unknown parameter.*objective")
  expect_error(ds.flower.model.xgboost(reg_lambda = 0), "reg_lambda")
  expect_error(ds.flower.model.xgboost(max_delta_step = 0), "max_delta_step")
  expect_error(ds.flower.model.xgboost(max_depth = 31L), "max_depth")
  expect_error(ds.flower.model.xgboost(learning_rate = 1e-300),
               "learning_rate.*remain positive as float32")
  direct <- model$params
  direct$subsample <- 0.8
  expect_error(
    dsFlowerClient:::.native_tree_xgboost_parameter_values(direct),
    "Unsupported XGBoost parameter.*subsample")
  direct <- model$params
  direct$max_depth <- 6.5
  expect_error(
    dsFlowerClient:::.build_xgboost_request(
      direct, c("age", "marker"),
      list(lower = c(0, -5), upper = c(100, 5)),
      list(c(18, 40, 65), c(-1, 0, 1)), "outcome",
      target_levels = c("control", "case")),
    "integer parameter must contain only integers")
  expect_error(
    dsFlowerClient:::.build_xgboost_request(
      model$params, c("age", "marker"),
      list(lower = c(0, -5), upper = c(100, 5)),
      list(c(18, 18 + 1e-10, 65), c(-1, 0, 1)), "outcome",
      target_levels = c("control", "case")),
    "cuts and bounds must remain strict as float32")
  expect_error(
    dsFlowerClient:::.build_xgboost_request(
      model$params, c("age", "marker"),
      list(lower = c(0, -5), upper = c(100, 5)),
      list(c(18, 40, 65), c(-1, 0, 1)), "outcome",
      target_levels = c("control", "case"),
      target_bounds = list(lower = 0, upper = 1)),
    "Binary XGBoost fixes")

  regression <- ds.flower.model.xgboost(task = "regression")
  request <- dsFlowerClient:::.build_xgboost_request(
    regression$params, c("age", "marker"),
    list(lower = c(0, -5), upper = c(100, 5)),
    list(c(18, 40, 65), c(-1, 0, 1)), "outcome",
    target_bounds = list(lower = -10, upper = 10))
  expect_identical(request$value$task, "regression")
  expect_equal(request$value$public_schema$target[c("lower", "upper")],
               list(lower = -10, upper = 10))
  expect_null(request$value$public_schema$target$levels)
})

test_that("XGBoost defaults are task-aware and explicit values win", {
  expected_binary <- list(
    num_boost_round = 8L, max_depth = 2L, learning_rate = 0.25)
  expected_regression <- list(
    num_boost_round = 5L, max_depth = 2L, learning_rate = 0.30)

  for (model in list(
      ds.flower.model.xgboost(),
      ds.flower.model("xgboost", task = "binary"))) {
    expect_equal(model$params[names(expected_binary)], expected_binary)
  }
  for (model in list(
      ds.flower.model.xgboost(task = "regression"),
      ds.flower.model("xgboost", task = "regression"))) {
    expect_equal(model$params[names(expected_regression)], expected_regression)
  }

  generic <- ds.flower.model(
    "xgboost", task = "regression", n_estimators = 11L,
    max_depth = 4L, eta = 0.05)
  concrete <- ds.flower.model.xgboost(
    task = "regression", n_estimators = 11L,
    max_depth = 4L, learning_rate = 0.05)
  for (model in list(generic, concrete)) {
    expect_identical(model$params$num_boost_round, 11L)
    expect_identical(model$params$max_depth, 4L)
    expect_equal(model$params$learning_rate, 0.05)
  }
})

test_that("XGBoost target levels preserve their ordered public scalar type", {
  build <- function(levels) .build_native_tree_fixture(target_levels = levels)
  strings <- build(c("control", "case"))
  booleans <- build(c(FALSE, TRUE))
  numbers <- build(c(0L, 1L))

  expect_identical(
    vapply(strings$value$public_schema$target$levels, `[[`, character(1), "type"),
    c("string", "string"))
  expect_identical(
    vapply(booleans$value$public_schema$target$levels, `[[`, character(1), "type"),
    c("boolean", "boolean"))
  expect_identical(
    vapply(numbers$value$public_schema$target$levels, `[[`, character(1), "type"),
    c("number", "number"))
  expect_false(identical(strings$value$public_schema$sha256,
                         booleans$value$public_schema$sha256))
  expect_false(identical(booleans$value$public_schema$sha256,
                         numbers$value$public_schema$sha256))
  expect_false(identical(
    strings$value$public_schema$sha256,
    build(c("case", "control"))$value$public_schema$sha256))
  expect_error(build(NULL), "exactly two distinct ordered")
  expect_error(build(c("case", "case")), "exactly two distinct ordered")
})

test_that("XGBoost submission validates its one-round public request pre-DSI", {
  reached_cli <- FALSE
  local_mocked_bindings(
    .require_flwr_cli = function() {
      reached_cli <<- TRUE
      stop("CLI must not be reached")
    },
    .package = "dsFlowerClient")
  args <- list(
    conns = list(site = TRUE), model = ds.flower.model.xgboost(),
    target = "outcome", features = c("age", "marker"), symbol = "D",
    num_rounds = 1L,
    feature_bounds = list(lower = c(0, -5), upper = c(100, 5)),
    feature_cuts = list(c(18, 40, 65), c(-1, 0, 1)),
    target_levels = c(0, 1))
  expect_error(do.call(ds.flower.submit, args), "capability is not enabled")
  expect_false(reached_cli)

  args$num_rounds <- 2L
  expect_error(do.call(ds.flower.submit, args), "exactly one Flower round")
  args$num_rounds <- 1L
  args$feature_cuts <- NULL
  expect_error(do.call(ds.flower.submit, args), "complete public feature_cuts")
  args$feature_cuts <- list(c(18, 40, 65), c(-1, 0, 1))
  args$target_levels <- NULL
  expect_error(do.call(ds.flower.submit, args), "two ordered public target_levels")
  expect_false(reached_cli)
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
      "catboost", "native-tight", "binary", "x",
      list(lower = 0, upper = 1), .native_tree_binary_target(),
      cuts = list(0.5), parameters = values),
    "exceeds 65536 bytes")
})
