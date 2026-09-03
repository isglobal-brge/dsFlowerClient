image_test_connection <- function() {
  structure(list(
    conns = list(site = NULL), symbol = ".dsfl_test",
    data = "PROJECT.images", data_kind = "resource", labels = data.frame()
  ), class = "dsflower_connection")
}

test_that("resource connect consumes the ResourceClient assigned by DataSHIELD", {
  assigned <- list()
  aggregated <- list()
  labels <- data.frame(
    name = "diagnosis", type = "categorical", columns = "diagnosis",
    description = "Public schema", stringsAsFactors = FALSE
  )

  local_mocked_bindings(
    datashield.assign.resource = function(conns, symbol, resource, success, ...) {
      for (node in names(conns)) success(node)
      invisible(NULL)
    },
    datashield.assign.expr = function(conns, symbol, expr, success, ...) {
      assigned[[length(assigned) + 1L]] <<- list(symbol = symbol, expr = expr)
      for (node in names(conns)) success(node)
      invisible(NULL)
    },
    datashield.aggregate = function(conns, expr) {
      aggregated[[length(aggregated) + 1L]] <<- expr
      list(site = labels)
    },
    .package = "DSI"
  )

  flower <- ds.flower.connect(
    conns = list(site = NULL), resource = "PROJECT.images")

  expect_length(assigned, 1L)
  expect_identical(as.character(assigned[[1L]]$expr[[1L]]), "flowerInitDS")
  expect_identical(as.character(aggregated[[1L]][[1L]]), "flowerImageLabelsDS")
  expect_identical(aggregated[[1L]][[2L]], flower$symbol)
  expect_equal(flower$labels, labels)
})

test_that("an initialized dsImaging symbol is consumed without reinitialization", {
  assigned <- list()
  resources <- 0L
  labels <- data.frame(
    name = "diagnosis", type = "categorical", columns = "diagnosis",
    description = "Public schema", stringsAsFactors = FALSE
  )

  local_mocked_bindings(
    datashield.assign.resource = function(...) {
      resources <<- resources + 1L
      invisible(NULL)
    },
    datashield.assign.expr = function(conns, symbol, expr, success, ...) {
      assigned[[length(assigned) + 1L]] <<- list(symbol = symbol, expr = expr)
      for (node in names(conns)) success(node)
      invisible(NULL)
    },
    datashield.aggregate = function(conns, expr) list(site = labels),
    .package = "DSI"
  )

  flower <- ds.flower.connect(list(site = NULL), symbol = "img")

  expect_equal(resources, 0L)
  expect_length(assigned, 1L)
  expect_identical(as.character(assigned[[1L]]$expr[[1L]]), "flowerInitDS")
  expect_identical(assigned[[1L]]$expr[[2L]], "img")
  expect_equal(flower$labels, labels)
})

test_that("low-level resource initialization uses the assigned ResourceClient", {
  assigned <- list()
  resources <- 0L
  local_mocked_bindings(
    datashield.assign.resource = function(conns, symbol, resource, success, ...) {
      resources <<- resources + 1L
      for (node in names(conns)) success(node)
      invisible(NULL)
    },
    datashield.assign.expr = function(conns, symbol, expr, success, ...) {
      assigned[[length(assigned) + 1L]] <<- list(symbol = symbol, expr = expr)
      for (node in names(conns)) success(node)
      invisible(NULL)
    },
    datashield.aggregate = function(conns, expr) list(site = list(status = "ok")),
    .package = "DSI"
  )

  ds.flower.nodes.init(
    list(site = NULL), resource = "PROJECT.images", symbol = "flower")

  expect_equal(resources, 1L)
  expect_length(assigned, 1L)
  expect_identical(as.character(assigned[[1L]]$expr[[1L]]), "flowerInitDS")
  expect_identical(assigned[[1L]]$expr[[2L]], "flower_res")
})

test_that("fit forwards an imaging symbol through the vision path", {
  submitted <- NULL
  local_mocked_bindings(
    ds.flower.submit = function(...) {
      submitted <<- list(...)
      structure(list(), class = "dsflower_run")
    },
    .package = "dsFlowerClient"
  )

  ds.flower.fit(
    conns = list(site = NULL), symbol = "img", target = "label",
    model = "pytorch_resnet18", rounds = 1L)

  expect_identical(submitted$symbol, "img")
  expect_identical(submitted$data_kind, "image")
  expect_null(submitted$features)
})

test_that("connect requires exactly one data source", {
  expect_error(ds.flower.connect(list(site = NULL)), "exactly one")
  expect_error(
    ds.flower.connect(
      list(site = NULL), resource = "PROJECT.resource", symbol = "D"),
    "exactly one")
})

test_that("image helpers call dsFlower endpoints with Flower handle signatures", {
  flower <- image_test_connection()
  calls <- list()
  responses <- list(
    flowerImageLabelsDS = data.frame(
      name = "diagnosis", type = "categorical", columns = "diagnosis",
      description = NA_character_, stringsAsFactors = FALSE
    ),
    flowerImageAssetsDS = data.frame(
      alias = c("images", "radiomics"),
      kind = c("image_root", "feature_table"),
      provider = c("pacs", "pyradiomics"), stringsAsFactors = FALSE
    ),
    flowerImageMasksDS = data.frame(
      alias = "tumour_masks", provider = "nnunet", status = "declared",
      stringsAsFactors = FALSE
    )
  )

  local_mocked_bindings(
    datashield.aggregate = function(conns, expr) {
      calls[[length(calls) + 1L]] <<- expr
      list(site = responses[[as.character(expr[[1L]])]])
    },
    .package = "DSI"
  )

  labels <- ds.flower.labels(flower)
  features <- ds.flower.features(flower)
  masks <- ds.flower.masks(flower)

  expect_equal(vapply(calls, function(expr) as.character(expr[[1L]]), character(1)),
               c("flowerImageLabelsDS", "flowerImageAssetsDS", "flowerImageMasksDS"))
  expect_true(all(vapply(calls, length, integer(1)) == 2L))
  expect_true(all(vapply(calls, function(expr) identical(expr[[2L]], flower$symbol),
                         logical(1))))
  expect_equal(labels$site$name, "diagnosis")
  expect_equal(features$alias, "radiomics")
  expect_equal(features$kind, "feature_table")
  expect_equal(masks$alias, "tumour_masks")
})

test_that("image helper fallbacks keep canonical public schemas", {
  flower <- image_test_connection()
  local_mocked_bindings(
    datashield.aggregate = function(...) stop("unavailable"),
    .package = "DSI"
  )

  expect_named(ds.flower.features(flower), c("alias", "kind", "provider"))
  expect_named(ds.flower.masks(flower), c("alias", "provider", "status"))
  expect_equal(nrow(ds.flower.features(flower)), 0L)
  expect_equal(nrow(ds.flower.masks(flower)), 0L)
})
