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
  pinned <- dsFlowerClient:::.validate_native_tree_request_wire(
    manifest$b64, manifest$sha256)
  expect_identical(pinned$json, manifest$json)
  expect_error(dsFlowerClient:::.validate_native_tree_request_wire(
    manifest$b64, strrep("0", 64L)), "do not match")

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

test_that("native-tree wire preserves valid closing-tag labels", {
  manifest <- .build_native_tree_fixture(c("</control>", "case"))

  expect_match(manifest$json, '"value":"</control>"', fixed = TRUE)
  expect_false(grepl("<\\/control>", manifest$json, fixed = TRUE))

  literal <- "<\\/control>"
  literal_manifest <- .build_native_tree_fixture(c(literal, "case"))
  decoded <- jsonlite::fromJSON(
    literal_manifest$json, simplifyVector = FALSE)
  expect_identical(
    decoded$public_schema$target$levels[[1L]]$value, literal)
})

test_that("native-tree wire preserves one-element arrays", {
  manifest <- dsFlowerClient:::.build_native_tree_manifest(
    "catboost", "native-tight", "binary", "x",
    list(lower = 0, upper = 1), .native_tree_binary_target(), list(0.5),
    parameters = list(depth = 2L, iterations = 3L, l2_leaf_reg = 1,
                      learning_rate = 0.1, max_delta_step = 2))

  expect_match(manifest$json, '"features":\\["x"\\]')
  expect_match(manifest$json, '"lower":\\[0.0\\]')
  expect_match(manifest$json, '"cuts":\\[\\[0.5\\]\\]')
  parameter <- dsFlowerClient:::.native_tree_parameter_record(list(
    name = "one_element_array", type = "integer_array", value = list(1L)))
  expect_match(rawToChar(dsFlowerClient:::.native_tree_json(parameter)),
               '"value":\\[1\\]')
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
      "extra_trees", "native-tight", "binary", "x",
      list(lower = 0, upper = 1), .native_tree_binary_target(), list(0.5),
      parameters = list(max_depth = 2L, n_estimators = 101L),
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

.boosting_manifest_fixture <- function(engine) {
  manifest <- jsonlite::fromJSON(
    .native_tree_expected_json, simplifyVector = FALSE)
  manifest$engine <- engine
  values <- if (identical(engine, "lightgbm")) {
    list(lambda_l1 = 0, lambda_l2 = 1, learning_rate = 0.1,
         max_delta_step = 2, max_depth = 2L, min_data_in_leaf = 1L,
         min_gain_to_split = 0, num_iterations = 3L, num_leaves = 2L)
  } else {
    list(depth = 2L, iterations = 3L, l2_leaf_reg = 1,
         learning_rate = 0.1, max_delta_step = 2)
  }
  manifest$parameters <- lapply(names(values), function(name) list(
    name = name,
    type = if (is.integer(values[[name]])) "integer" else "number",
    value = values[[name]]))
  manifest
}

test_that("client enforces exact LightGBM-style and CatBoost-style profiles", {
  for (engine in c("lightgbm", "catboost")) {
    manifest <- .boosting_manifest_fixture(engine)
    expect_no_error(dsFlowerClient:::.canonical_native_tree_manifest(manifest))

    extra <- manifest
    extra$parameters[[length(extra$parameters) + 1L]] <- list(
      name = "subsample", type = "number", value = 0.5)
    expect_error(
      dsFlowerClient:::.canonical_native_tree_manifest(extra),
      paste0("Unsupported ", if (engine == "lightgbm") {
        "LightGBM-style"
      } else {
        "CatBoost-style"
      }, " parameter"))

    wrong <- manifest
    index <- which(vapply(
      wrong$parameters, `[[`, character(1), "name") == "learning_rate")
    wrong$parameters[[index]]$type <- "integer"
    wrong$parameters[[index]]$value <- 1L
    expect_error(
      dsFlowerClient:::.canonical_native_tree_manifest(wrong),
      "learning_rate.*wrong declared type")
  }

  collapsed <- .boosting_manifest_fixture("catboost")
  collapsed$public_schema$cuts[[1L]][[2L]] <-
    collapsed$public_schema$cuts[[1L]][[1L]] + 1e-10
  core <- collapsed$public_schema[c(
    "version", "features", "lower", "upper", "cuts", "target")]
  collapsed$public_schema$sha256 <- digest::digest(
    dsFlowerClient:::.native_tree_json(core),
    algo = "sha256", serialize = FALSE)
  expect_error(
    dsFlowerClient:::.canonical_native_tree_manifest(collapsed), "float32")
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
    "catboost", "native-tight", "regression", "x",
    list(lower = 0, upper = 1), .native_tree_continuous_target(), list(0.5),
    parameters = list(depth = 2L, iterations = 3L, l2_leaf_reg = 1,
                      learning_rate = 0.1, max_delta_step = 2))
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
  expect_true(ds.flower.list_models()$available[
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

test_that("ExtraTrees request uses the data-independent exact profile", {
  params <- list(task = "binary", n_estimators = 32L, max_depth = 4L)
  request <- dsFlowerClient:::.build_extra_trees_request(
    params, c("age", "marker"),
    list(lower = c(0, -5), upper = c(100, 5)),
    list(c(18, 40, 65), c(-1, 0, 1)), "outcome",
    target_levels = c("control", "case"))

  expect_identical(request$value$engine, "extra_trees")
  expect_identical(request$value$task, "binary")
  expect_identical(
    vapply(request$value$parameters, `[[`, character(1), "name"),
    c("max_depth", "n_estimators"))
  expect_identical(
    vapply(request$value$parameters, `[[`, character(1), "type"),
    c("integer", "integer"))
  expect_identical(request$value$resources$max_trees, 32L)
  expect_identical(request$value$resources$max_depth, 4L)
  expect_identical(
    dsFlowerClient:::.validate_native_tree_request_wire(
      request$b64, request$sha256)$json,
    request$json)

  for (changed in list(
      list(task = "binary", n_estimators = 513L, max_depth = 4L),
      list(task = "binary", n_estimators = 32L, max_depth = 13L),
      list(task = "binary", n_estimators = 32L, max_depth = 4L,
           criterion = "gini"))) {
    expect_error(
      dsFlowerClient:::.build_extra_trees_request(
        changed, c("age", "marker"),
        list(lower = c(0, -5), upper = c(100, 5)),
        list(c(18, 40, 65), c(-1, 0, 1)), "outcome",
        target_levels = c("control", "case")),
      "ExtraTrees")
  }
  expect_error(
    dsFlowerClient:::.build_extra_trees_request(
      params, c("age", "marker"),
      list(lower = c(0, -5), upper = c(100, 5)),
      list(c(18, 18 + 1e-10, 65), c(-1, 0, 1)), "outcome",
      target_levels = c("control", "case")),
    "strict as float32")
})

test_that("Random Forest resolves task-aware defaults and public mtry exactly", {
  binary <- ds.flower.model.random_forest()
  regression <- ds.flower.model.random_forest(task = "regression")
  expect_equal(
    binary$params[c("n_estimators", "max_depth", "max_features")],
    list(n_estimators = 8L, max_depth = 4L, max_features = "auto"))
  expect_equal(
    regression$params[c("n_estimators", "max_depth", "max_features")],
    list(n_estimators = 4L, max_depth = 4L, max_features = "auto"))

  build <- function(model, features) dsFlowerClient:::.build_random_forest_request(
    model$params, features,
    list(lower = rep(0, length(features)), upper = rep(1, length(features))),
    lapply(features, function(...) 0.5), "outcome",
    target_levels = if (identical(model$params$task, "binary"))
      c("control", "case") else NULL,
    target_bounds = if (identical(model$params$task, "regression"))
      list(lower = -1, upper = 1) else NULL)
  parameter <- function(request, name) {
    index <- which(vapply(
      request$value$parameters, `[[`, character(1), "name") == name)
    request$value$parameters[[index]]$value
  }
  binary_request <- build(binary, paste0("x", seq_len(5L)))
  regression_request <- build(regression, paste0("x", seq_len(5L)))
  expect_identical(parameter(binary_request, "max_features"), 3L)
  expect_identical(parameter(regression_request, "max_features"), 2L)
  expect_identical(parameter(binary_request, "n_estimators"), 8L)
  expect_identical(parameter(regression_request, "n_estimators"), 4L)
  expect_false(any(vapply(
    binary_request$value$parameters, function(record) is.character(record$value),
    logical(1))))

  explicit <- build(
    ds.flower.model.random_forest(max_features = 2L), paste0("x", 1:10))
  expect_identical(parameter(explicit, "max_features"), 2L)
  expect_error(
    build(ds.flower.model.random_forest(max_features = 11L), paste0("x", 1:10)),
    "cannot exceed the public feature count")
  expect_error(ds.flower.model.random_forest(max_features = "sqrt"),
               "NULL, 'auto', or a positive integer")
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
  reached_private_path <- FALSE
  local_mocked_bindings(
    .validate_dsi_transport_security = function(...) TRUE,
    .assert_native_tree_capability = function(...) {
      stop("Verified native-tree capability for 'xgboost' is unavailable on: site.",
           call. = FALSE)
    },
    .require_flwr_cli = function() {
      reached_cli <<- TRUE
      stop("CLI must not be reached")
    },
    ds.flower.connect = function(...) {
      reached_private_path <<- TRUE
      stop("private path must not be reached")
    },
    .package = "dsFlowerClient")
  args <- list(
    conns = list(site = TRUE), model = ds.flower.model.xgboost(),
    target = "outcome", features = c("age", "marker"), symbol = "D",
    num_rounds = 1L,
    feature_bounds = list(lower = c(0, -5), upper = c(100, 5)),
    feature_cuts = list(c(18, 40, 65), c(-1, 0, 1)),
    target_levels = c(0, 1))
  expect_error(do.call(ds.flower.submit, args), "capability.*is unavailable")
  expect_false(reached_cli)
  expect_false(reached_private_path)

  args$num_rounds <- 2L
  expect_error(do.call(ds.flower.submit, args), "exactly one Flower round")
  args$num_rounds <- 1L
  args$feature_cuts <- NULL
  expect_error(do.call(ds.flower.submit, args), "complete public feature_cuts")
  args$feature_cuts <- list(c(18, 40, 65), c(-1, 0, 1))
  args$target_levels <- NULL
  expect_error(do.call(ds.flower.submit, args), "two ordered public target_levels")
  expect_false(reached_cli)
  expect_false(reached_private_path)
})

test_that("native-tree submission reuses its targeted capability response", {
  events <- character()
  local_conn <- structure(list(), class = "DSLiteConnection")
  flower <- structure(
    list(conns = list(site = local_conn), symbol = "flower"),
    class = "dsflower_connection")
  capabilities <- list(site = list(
    runner_abi = 3L,
    runner_sha256 = paste(rep("a", 64L), collapse = "")))
  local_mocked_bindings(
    .validate_dsi_transport_security = function(...) TRUE,
    .assert_native_tree_capability = function(conns, engine) {
      events <<- c(events, paste0("targeted:", engine))
      capabilities
    },
    .assert_runner_compatibility = function(...) {
      events <<- c(events, "unexpected-common-capability")
      stop("second capability query", call. = FALSE)
    },
    .require_flwr_cli = function() TRUE,
    .validate_declarative_model_preflight = function(...) TRUE,
    ds.flower.connect = function(...) {
      events <<- c(events, "connect")
      flower
    },
    ds.flower.nodes.prepare = function(...) {
      events <<- c(events, "prepare")
      stop("prepared sentinel", call. = FALSE)
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
    .package = "dsFlowerClient")

  expect_error(
    ds.flower.submit(
      conns = list(site = local_conn), model = ds.flower.model.xgboost(),
      target = "outcome", features = c("age", "marker"), symbol = "D",
      feature_bounds = list(lower = c(0, -5), upper = c(100, 5)),
      feature_cuts = list(c(18, 40, 65), c(-1, 0, 1)),
      target_levels = c(0, 1)),
    "prepared sentinel")
  expect_false("unexpected-common-capability" %in% events)
  expect_identical(
    events,
    c("targeted:xgboost", "connect", "prepare", "link", "nodes",
      "disconnect"))
})

test_that("native-tree FAB uses isolated entrypoints and no torch dependency", {
  runner <- file.path(withr::local_tempdir(), "dsflower_runner")
  dir.create(runner)
  file.create(file.path(runner, "__init__.py"))
  local_mocked_bindings(
    .runner_skeleton_dir = function() runner,
    .package = "dsFlowerClient")
  app <- dsFlowerClient:::.build_submission_app(
    list(track = "native_tree", pkg_dir = NULL),
    c('dp-track = "native_tree"'), withr::local_tempdir())
  toml <- paste(readLines(file.path(app, "pyproject.toml"), warn = FALSE),
                collapse = "\n")
  expect_match(toml, "native_tree_server_app:app", fixed = TRUE)
  expect_match(toml, "native_tree_client_app:app", fixed = TRUE)
  expect_match(toml, "flwr==1.31.0", fixed = TRUE)
  expect_match(toml, "numpy==2.4.6", fixed = TRUE)
  expect_match(toml, "pandas==3.0.3", fixed = TRUE)
  expect_match(toml, "pyarrow==23.0.1", fixed = TRUE)
  expect_match(toml, "cryptography==46.0.7", fixed = TRUE)
  expect_false(grepl("torch", toml, fixed = TRUE))
  expect_false(grepl("opacus", toml, fixed = TRUE))

  validation_app <- dsFlowerClient:::.build_submission_app(
    list(track = "native_tree_validation", pkg_dir = NULL),
    c('dp-track = "validation"'), withr::local_tempdir())
  validation_toml <- paste(readLines(
    file.path(validation_app, "pyproject.toml"), warn = FALSE),
    collapse = "\n")
  expect_match(validation_toml,
               "native_tree_validation_server_app:app", fixed = TRUE)
  expect_match(validation_toml,
               "native_tree_validation_client_app:app", fixed = TRUE)
  expect_match(validation_toml, "flwr==1.31.0", fixed = TRUE)
  expect_match(validation_toml, "numpy==2.4.6", fixed = TRUE)
  expect_false(grepl("torch", validation_toml, fixed = TRUE))
  expect_false(grepl("opacus", validation_toml, fixed = TRUE))
})

test_that("native-tree admission uses the selected engine's fresh probe", {
  expected <- paste(rep("a", 64L), collapse = "")
  calls <- 0L
  captured <- NULL
  result <- list(site = list(
    runner_abi = 3L,
    runner_sha256 = expected,
    native_tree = list(
      contract = "dsflower-native-tree-request-v1",
      probed_engines = "random_forest",
      random_forest_native_tight_available = FALSE)))
  local_mocked_bindings(
    .compute_local_runner_hash = function(...) expected,
    .package = "dsFlowerClient")
  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      calls <<- calls + 1L
      captured <<- expr
      result
    },
    .package = "DSI")
  expect_error(
    dsFlowerClient:::.assert_native_tree_capability(
      list(site = TRUE), "random_forest"),
    "unavailable on: site")
  expect_identical(calls, 1L)
  expect_identical(as.character(captured[[1L]]), "flowerGetCapabilitiesDS")
  expect_identical(as.character(captured[[2L]]), "random_forest")
  result$site$native_tree$random_forest_native_tight_available <- TRUE
  expect_no_error(
    dsFlowerClient:::.assert_native_tree_capability(
      list(site = TRUE), "random_forest"))
  expect_identical(calls, 2L)

  result$site$runner_abi <- 2L
  expect_error(
    dsFlowerClient:::.assert_native_tree_capability(
      list(site = TRUE), "random_forest"),
    "Incompatible dsFlower runner")
  before_invalid <- calls
  expect_error(
    dsFlowerClient:::.assert_native_tree_capability(
      list(site = TRUE), "Random_Forest"),
    "no implemented capability probe")
  expect_identical(calls, before_invalid)
})

test_that("native-tree request has bounded cuts and canonical bytes", {
  expect_error(
    dsFlowerClient:::.build_native_tree_manifest(
      "xgboost", "native-tight", "binary", "x",
      list(lower = 0, upper = 20000), .native_tree_binary_target(),
      list(seq_len(16385L)), resources = list(max_bins = 20000L)),
    "16384-cut contract cap")

  features <- vapply(seq_len(400L), function(i) {
    paste0(sprintf("f%03d_", i), strrep("x", 190L))
  }, character(1))
  values <- dsFlowerClient:::.native_tree_xgboost_parameter_values(
    ds.flower.model.xgboost()$params)
  expect_error(
    dsFlowerClient:::.build_native_tree_manifest(
      "xgboost", "native-tight", "binary", features,
      list(lower = rep(0, length(features)),
           upper = rep(1, length(features))), .native_tree_binary_target(),
      cuts = lapply(features, function(...) 0.5), parameters = values,
      resources = list(max_features = length(features))),
    "exceeds 65536 bytes")
})
