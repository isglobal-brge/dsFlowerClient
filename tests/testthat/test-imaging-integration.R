image_test_connection <- function() {
  structure(list(
    conns = list(site = NULL), symbol = ".dsfl_test",
    data = "PROJECT.images", data_kind = "resource", labels = data.frame()
  ), class = "dsflower_connection")
}

test_that("resource connect uses registered dsFlower imaging discovery", {
  assigned <- list()
  aggregated <- list()
  labels <- data.frame(
    name = "diagnosis", type = "categorical", columns = "diagnosis",
    description = "Public schema", stringsAsFactors = FALSE
  )

  local_mocked_bindings(
    datashield.assign.resource = function(conns, symbol, resource) invisible(NULL),
    datashield.assign.expr = function(conns, symbol, expr) {
      assigned[[length(assigned) + 1L]] <<- list(symbol = symbol, expr = expr)
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

  expect_identical(as.character(assigned[[1L]]$expr[[1L]]), "imagingInitDS")
  expect_identical(as.character(assigned[[2L]]$expr[[1L]]), "flowerInitDS")
  expect_identical(as.character(aggregated[[1L]][[1L]]), "flowerImageLabelsDS")
  expect_identical(aggregated[[1L]][[2L]], flower$symbol)
  expect_equal(flower$labels, labels)
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
