image_test_connection <- function() {
  structure(list(
    conns = list(site = NULL), symbol = "dsf_test",
    data = "PROJECT.images", data_kind = "resource", labels = data.frame()
  ), class = "dsflower_connection")
}

test_that("resource connect admits dsImaging before dsFlower", {
  assigned <- list()
  aggregated <- list()
  state <- list(site = character())
  labels <- data.frame(
    name = "diagnosis", type = "categorical", columns = "diagnosis",
    description = "Public schema", stringsAsFactors = FALSE
  )

  local_mocked_bindings(
    datashield.symbols = function(conns, ...) state[names(conns)],
    datashield.assign.resource = function(conns, symbol, resource, success, ...) {
      for (node in names(conns)) {
        state[[node]] <<- c(state[[node]], symbol)
        success(node)
      }
      invisible(NULL)
    },
    datashield.assign.expr = function(conns, symbol, expr, success, ...) {
      assigned[[length(assigned) + 1L]] <<- list(symbol = symbol, expr = expr)
      for (node in names(conns)) {
        state[[node]] <<- c(state[[node]], symbol)
        success(node)
      }
      invisible(NULL)
    },
    datashield.aggregate = function(conns, expr) {
      aggregated[[length(aggregated) + 1L]] <<- expr
      list(site = labels)
    },
    datashield.rm = function(conns, symbol, ...) {
      host <- names(conns)[[1L]]
      state[[host]] <<- setdiff(state[[host]], symbol)
      invisible(NULL)
    },
    .package = "DSI"
  )

  flower <- ds.flower.connect(
    conns = list(site = NULL), resource = "PROJECT.images")

  expect_match(flower$symbol, "^dsf_[0-9a-f]{32}$")
  expect_length(assigned, 2L)
  expect_identical(as.character(assigned[[1L]]$expr[[1L]]), "imagingInitDS")
  expect_identical(assigned[[1L]]$expr[[2L]],
    dsFlowerClient:::.dsi_init_resource_symbol(flower$symbol))
  expect_identical(as.character(assigned[[2L]]$expr[[1L]]), "flowerInitDS")
  expect_identical(assigned[[2L]]$expr[[2L]], paste0(flower$symbol, "_img"))
  expect_identical(as.character(aggregated[[1L]][[1L]]), "flowerImageLabelsDS")
  expect_identical(aggregated[[1L]][[2L]], flower$symbol)
  expect_equal(flower$labels, labels)
})

test_that("resource connect cleans provider transients across backends", {
  provider <- "opal"
  state <- list(site = character())
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) state[names(conns)],
    datashield.assign.resource = function(conns, symbol, resource, success, ...) {
      for (node in names(conns)) {
        transients <- if (identical(provider, "armadillo")) {
          c("R", "rds")
        } else {
          NULL
        }
        state[[node]] <<- unique(c(state[[node]], symbol, transients))
        success(node)
      }
      invisible(NULL)
    },
    datashield.assign.expr = function(conns, symbol, expr, success, ...) {
      for (node in names(conns)) {
        state[[node]] <<- unique(c(state[[node]], symbol))
        success(node)
      }
      invisible(NULL)
    },
    datashield.aggregate = function(conns, expr) list(site = data.frame()),
    datashield.rm = function(conns, symbol, ...) {
      node <- names(conns)[[1L]]
      state[[node]] <<- setdiff(state[[node]], symbol)
      invisible(NULL)
    },
    .package = "DSI"
  )

  for (backend in c("opal", "armadillo")) {
    provider <- backend
    state$site <- character()
    flower <- ds.flower.connect(
      list(site = NULL), resource = "project/folder/images")

    expect_false(any(c("R", "rds") %in% state$site), info = backend)
    expect_setequal(state$site, c(flower$symbol, flower$imaging_symbol))
    expect_true(ds.flower.disconnect(flower))
    expect_identical(state$site, character(), info = backend)
  }
})

test_that("resource paths refuse to overwrite provider transient symbols", {
  state <- list(site = character())
  assignments <- 0L
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) state[names(conns)],
    datashield.assign.resource = function(...) {
      assignments <<- assignments + 1L
    },
    .package = "DSI"
  )

  for (transient in c("R", "rds")) {
    state$site <- transient
    expect_error(
      ds.flower.connect(
        list(site = NULL), resource = "project/folder/images"),
      "already exists", info = transient)
    expect_error(
      ds.flower.nodes.init(
        list(site = NULL), resource = "project/folder/images"),
      "already exists", info = transient)
  }
  expect_identical(assignments, 0L)
})

test_that("resource paths retry provider transient cleanup transactionally", {
  state <- list(site = character())
  remove_r_attempts <- 0L
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) state[names(conns)],
    datashield.assign.resource = function(conns, symbol, resource, success, ...) {
      for (node in names(conns)) {
        state[[node]] <<- unique(c(state[[node]], symbol, "R", "rds"))
        success(node)
      }
      invisible(NULL)
    },
    datashield.assign.expr = function(conns, symbol, expr, success, ...) {
      for (node in names(conns)) {
        state[[node]] <<- unique(c(state[[node]], symbol))
        success(node)
      }
      invisible(NULL)
    },
    datashield.aggregate = function(conns, expr) list(site = list(status = "ok")),
    datashield.rm = function(conns, symbol, ...) {
      node <- names(conns)[[1L]]
      if (identical(symbol, "R")) {
        remove_r_attempts <<- remove_r_attempts + 1L
        if (remove_r_attempts == 1L) {
          stop("simulated transient removal failure")
        }
      }
      state[[node]] <<- setdiff(state[[node]], symbol)
      invisible(NULL)
    },
    .package = "DSI"
  )

  for (path in c("connect", "nodes.init")) {
    state$site <- character()
    remove_r_attempts <- 0L
    call <- if (identical(path, "connect")) {
      quote(ds.flower.connect(
        list(site = NULL), resource = "project/folder/images"))
    } else {
      quote(ds.flower.nodes.init(
        list(site = NULL), resource = "project/folder/images"))
    }

    expect_error(eval(call), "Temporary imaging resource cleanup failed",
                 info = path)
    expect_identical(remove_r_attempts, 2L, info = path)
    expect_identical(state$site, character(), info = path)
  }
})

test_that("resource connect routes tabular resources explicitly", {
  assigned <- list()
  state <- list(site = character())
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) state[names(conns)],
    datashield.assign.resource = function(conns, symbol, resource, success, ...) {
      state$site <<- c(state$site, symbol, "R", "rds")
      success("site")
      invisible(NULL)
    },
    datashield.assign.expr = function(conns, symbol, expr, success, ...) {
      assigned[[length(assigned) + 1L]] <<- list(symbol = symbol, expr = expr)
      state$site <<- unique(c(state$site, symbol))
      success("site")
      invisible(NULL)
    },
    datashield.aggregate = function(conns, expr) list(site = data.frame()),
    datashield.rm = function(conns, symbol, ...) {
      state$site <<- setdiff(state$site, symbol)
      invisible(NULL)
    },
    .package = "DSI"
  )

  flower <- ds.flower.connect(
    list(site = NULL), resource = "PROJECT.table",
    resource_kind = "tabular")

  expect_identical(
    vapply(assigned, function(item) as.character(item$expr[[1L]]), character(1)),
    c("as.resource.client", "flowerInitDS"))
  expect_identical(assigned[[1L]]$symbol,
    dsFlowerClient:::.dsi_init_resource_symbol(flower$symbol))
  expect_identical(as.character(assigned[[1L]]$expr[[2L]]),
    assigned[[1L]]$symbol)
  expect_identical(assigned[[2L]]$expr[[2L]], assigned[[1L]]$symbol)
  expect_identical(flower$resource_kind, "tabular")
  expect_null(flower$imaging_symbol)
  expect_false(any(c("R", "rds") %in% state$site))
  expect_false(any(vapply(assigned, function(item) {
    identical(as.character(item$expr[[1L]]), "imagingInitDS")
  }, logical(1))))
})

test_that("resource routing is validated before any DSI call", {
  calls <- 0L
  local_mocked_bindings(
    datashield.symbols = function(...) {
      calls <<- calls + 1L
      list(site = character())
    },
    .package = "DSI"
  )
  expect_error(
    ds.flower.connect(
      list(site = NULL), resource = "PROJECT.table",
      resource_kind = "auto"),
    "exactly 'imaging' or 'tabular'")
  expect_identical(calls, 0L)
})

test_that("resource connect removes partial session state when admission fails", {
  removed <- character(0)
  state <- list(site_1 = character(), site_2 = character())
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) {
      state[names(conns)]
    },
    datashield.assign.resource = function(conns, symbol, resource, success, ...) {
      for (node in names(conns)) {
        state[[node]] <<- c(state[[node]], symbol)
        success(node)
      }
      invisible(NULL)
    },
    datashield.assign.expr = function(conns, symbol, expr, success, error, ...) {
      method <- as.character(expr[[1L]])
      if (identical(method, "imagingInitDS")) {
        state$site_1 <<- c(state$site_1, symbol)
        success("site_1")
        error("site_2", "admission failed")
      } else if (identical(method, "imagingDestroyDS")) {
        success(names(conns)[[1L]])
      }
      invisible(NULL)
    },
    datashield.rm = function(conns, symbol, ...) {
      host <- names(conns)[[1L]]
      state[[host]] <<- setdiff(state[[host]], symbol)
      removed <<- c(removed, paste(host, symbol, sep = ":"))
      invisible(NULL)
    },
    .package = "DSI"
  )

  expect_error(
    ds.flower.connect(
      list(site_1 = NULL, site_2 = NULL), resource = "PROJECT.images"),
    "privacy admission failed"
  )
  expect_length(removed, 3L)
  expect_length(grep("_img$", removed), 1L)
  expect_length(grep(":dsFres\\.", removed), 2L)
  expect_true(all(grepl("(_img|:dsFres\\.)", removed)))
})

test_that("disconnect destroys private registries before removing symbols", {
  methods <- character()
  removed <- character()
  resource_symbol <- dsFlowerClient:::.dsi_init_resource_symbol("dsf_test")
  state <- list(site = c("dsf_test", "dsf_test_img", resource_symbol))
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) state[names(conns)],
    datashield.assign.expr = function(conns, symbol, expr, success, ...) {
      methods <<- c(methods, as.character(expr[[1L]]))
      success(names(conns)[[1L]])
      invisible(NULL)
    },
    datashield.rm = function(conns, symbol, ...) {
      host <- names(conns)[[1L]]
      state[[host]] <<- setdiff(state[[host]], symbol)
      removed <<- c(removed, symbol)
      invisible(NULL)
    },
    .package = "DSI"
  )

  expect_true(ds.flower.disconnect(image_test_connection()))
  expect_identical(methods, c("flowerDestroyDS", "imagingDestroyDS"))
  expect_setequal(removed, c("dsf_test", "dsf_test_img", resource_symbol))
})

test_that("disconnect does not destroy a caller-owned imaging handle", {
  methods <- character()
  state <- list(site = "dsf_test")
  flower <- image_test_connection()
  flower$data_kind <- "symbol"
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) state[names(conns)],
    datashield.assign.expr = function(conns, symbol, expr, success, ...) {
      methods <<- c(methods, as.character(expr[[1L]]))
      success(names(conns)[[1L]])
      invisible(NULL)
    },
    datashield.rm = function(conns, symbol, ...) {
      state[[names(conns)[[1L]]]] <<- character()
      invisible(NULL)
    },
    .package = "DSI"
  )

  expect_true(ds.flower.disconnect(flower))
  expect_identical(methods, "flowerDestroyDS")
})

test_that("automatic disconnect reports a retryable partial failure", {
  flower <- image_test_connection()
  local_mocked_bindings(
    ds.flower.disconnect = function(...) {
      stop(
        'Node session destruction failed. Retry ds.flower.nodes.destroy(',
        'conns = conns, symbol = "dsf_test", ',
        'imaging_symbol = "dsf_test_img").',
        call. = FALSE)
    },
    .package = "dsFlowerClient"
  )

  expect_message(
    cleaned <- dsFlowerClient:::.dsflower_disconnect_on_exit(flower),
    "Automatic dsFlower session cleanup was incomplete.*Retry")
  expect_false(cleaned)
})

test_that("low-level node init never overwrites an existing handle", {
  assignments <- 0L
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) list(site = "flower"),
    datashield.assign.expr = function(...) {
      assignments <<- assignments + 1L
    },
    .package = "DSI"
  )

  expect_error(
    ds.flower.nodes.init(list(site = NULL), data = "D", symbol = "flower"),
    "already exists")
  expect_identical(assignments, 0L)
})

test_that("low-level lifecycle rejects symbols hidden from DSI enumeration", {
  assignments <- 0L
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) list(site = character()),
    datashield.assign.expr = function(...) assignments <<- assignments + 1L,
    .package = "DSI"
  )

  expect_error(
    ds.flower.nodes.init(
      list(site = NULL), data = "D", symbol = ".flower"),
    "Visible DataSHIELD symbols")
  expect_error(
    ds.flower.nodes.destroy(list(site = NULL), symbol = ".flower"),
    "visible DataSHIELD symbols")
  expect_identical(assignments, 0L)
})

test_that("high-level connect never overwrites a colliding session handle", {
  assignments <- 0L
  session_symbol <- paste0("dsf_", strrep("a", 32L))
  local_mocked_bindings(
    .new_capability_token = function(prefix) paste0(prefix, "_", strrep("a", 32L)),
    .package = "dsFlowerClient"
  )
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) list(site = session_symbol),
    datashield.assign.expr = function(...) {
      assignments <<- assignments + 1L
    },
    .package = "DSI"
  )

  expect_error(
    ds.flower.connect(list(site = NULL), symbol = "img"),
    "already exists")
  expect_identical(assignments, 0L)
})

test_that("an initialized dsImaging symbol is consumed without reinitialization", {
  assigned <- list()
  resources <- 0L
  labels <- data.frame(
    name = "diagnosis", type = "categorical", columns = "diagnosis",
    description = "Public schema", stringsAsFactors = FALSE
  )

  local_mocked_bindings(
    datashield.symbols = function(conns, ...) list(site = character()),
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

test_that("low-level resource initialization uses dsImaging admission", {
  assigned <- list()
  resources <- 0L
  removed <- character()
  state <- list(site = character())
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) state[names(conns)],
    datashield.assign.resource = function(conns, symbol, resource, success, ...) {
      resources <<- resources + 1L
      for (node in names(conns)) {
        state[[node]] <<- c(state[[node]], symbol)
        success(node)
      }
      invisible(NULL)
    },
    datashield.assign.expr = function(conns, symbol, expr, success, ...) {
      assigned[[length(assigned) + 1L]] <<- list(symbol = symbol, expr = expr)
      for (node in names(conns)) {
        state[[node]] <<- c(state[[node]], symbol)
        success(node)
      }
      invisible(NULL)
    },
    datashield.aggregate = function(conns, expr) list(site = list(status = "ok")),
    datashield.rm = function(conns, symbol, ...) {
      host <- names(conns)[[1L]]
      state[[host]] <<- setdiff(state[[host]], symbol)
      removed <<- c(removed, symbol)
      invisible(NULL)
    },
    .package = "DSI"
  )

  ds.flower.nodes.init(
    list(site = NULL), resource = "PROJECT.images", symbol = "flower")

  expect_equal(resources, 1L)
  expect_length(assigned, 2L)
  expect_identical(as.character(assigned[[1L]]$expr[[1L]]), "imagingInitDS")
  expect_match(assigned[[1L]]$expr[[2L]], "^dsFres\\.")
  expect_identical(as.character(assigned[[2L]]$expr[[1L]]), "flowerInitDS")
  expect_identical(assigned[[2L]]$expr[[2L]], "flower_img")
  expect_true(assigned[[1L]]$expr[[2L]] %in% removed)
})

test_that("low-level init resolves an explicitly tabular resource", {
  assigned <- list()
  state <- list(site = character())
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) state[names(conns)],
    datashield.assign.resource = function(conns, symbol, resource, success, ...) {
      state$site <<- c(state$site, symbol, "R", "rds")
      success("site")
      invisible(NULL)
    },
    datashield.assign.expr = function(conns, symbol, expr, success, ...) {
      assigned[[length(assigned) + 1L]] <<- list(symbol = symbol, expr = expr)
      state$site <<- unique(c(state$site, symbol))
      success("site")
      invisible(NULL)
    },
    datashield.aggregate = function(conns, expr) list(site = list(status = "ok")),
    datashield.rm = function(conns, symbol, ...) {
      state$site <<- setdiff(state$site, symbol)
      invisible(NULL)
    },
    .package = "DSI"
  )

  result <- ds.flower.nodes.init(
    list(site = NULL), resource = "PROJECT.table", symbol = "flower",
    resource_kind = "tabular")

  expect_identical(
    vapply(assigned, function(item) as.character(item$expr[[1L]]), character(1)),
    c("as.resource.client", "flowerInitDS"))
  expect_null(result$meta$imaging_symbol)
  expect_identical(result$meta$resource_kind, "tabular")
  expect_false(any(c("R", "rds") %in% state$site))
})

test_that("low-level destroy remembers the imaging handle it owns", {
  conns <- list(site = NULL)
  state <- list(site = character())
  destroyed <- character()
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) state[names(conns)],
    datashield.assign.resource = function(conns, symbol, resource, success, ...) {
      state$site <<- unique(c(state$site, symbol))
      success("site")
      invisible(NULL)
    },
    datashield.assign.expr = function(conns, symbol, expr, success, ...) {
      method <- as.character(expr[[1L]])
      if (method %in% c("flowerDestroyDS", "imagingDestroyDS")) {
        destroyed <<- c(destroyed, method)
      }
      state$site <<- unique(c(state$site, symbol))
      success("site")
      invisible(NULL)
    },
    datashield.aggregate = function(conns, expr) list(site = list(status = "ok")),
    datashield.rm = function(conns, symbol, ...) {
      state$site <<- setdiff(state$site, symbol)
      invisible(NULL)
    },
    .package = "DSI"
  )

  init <- ds.flower.nodes.init(
    conns, resource = "PROJECT.images", symbol = "flower")
  expect_identical(init$meta$imaging_symbol, "flower_img")
  result <- ds.flower.nodes.destroy(conns, symbol = "flower")

  expect_identical(destroyed, c("flowerDestroyDS", "imagingDestroyDS"))
  expect_identical(result$per_site$site$imaging$symbol, "flower_img")
  expect_null(dsFlowerClient:::.owned_imaging_handle(conns, "flower"))
})

test_that("a doubly failed dsFres removal remains API-retryable", {
  state <- list(site = character())
  resource_symbol <- dsFlowerClient:::.dsi_init_resource_symbol("flower")
  resource_remove_attempts <- 0L
  fail_resource_remove <- TRUE
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) state[names(conns)],
    datashield.assign.resource = function(conns, symbol, resource, success, ...) {
      expect_identical(symbol, resource_symbol)
      state$site <<- unique(c(state$site, symbol))
      success("site")
      invisible(NULL)
    },
    datashield.assign.expr = function(conns, symbol, expr, success, ...) {
      state$site <<- unique(c(state$site, symbol))
      success("site")
      invisible(NULL)
    },
    datashield.rm = function(conns, symbol, ...) {
      if (identical(symbol, resource_symbol)) {
        resource_remove_attempts <<- resource_remove_attempts + 1L
        if (isTRUE(fail_resource_remove)) {
          stop("simulated resource removal failure")
        }
      }
      state$site <<- setdiff(state$site, symbol)
      invisible(NULL)
    },
    .package = "DSI"
  )

  expect_warning(
    expect_error(
      ds.flower.nodes.init(
        list(site = NULL), resource = "PROJECT.images", symbol = "flower"),
      "Temporary imaging resource cleanup failed"),
    "Temporary resource cleanup was incomplete")
  expect_identical(resource_remove_attempts, 2L)
  expect_identical(state$site, resource_symbol)

  fail_resource_remove <- FALSE
  result <- ds.flower.nodes.destroy(
    list(site = NULL), symbol = "flower", imaging_symbol = "flower_img")
  expect_identical(result$per_site$site$resource$state, "removed")
  expect_identical(resource_remove_attempts, 3L)
  expect_identical(state$site, character())
})

test_that("low-level node init removes partial imaging state on failure", {
  removed <- character()
  state <- list(site_1 = character(), site_2 = character())
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) {
      state[names(conns)]
    },
    datashield.assign.resource = function(conns, symbol, resource, success, ...) {
      for (node in names(conns)) {
        state[[node]] <<- c(state[[node]], symbol)
        success(node)
      }
      invisible(NULL)
    },
    datashield.assign.expr = function(conns, symbol, expr, success, error, ...) {
      method <- as.character(expr[[1L]])
      if (identical(method, "imagingInitDS")) {
        for (node in names(conns)) {
          state[[node]] <<- c(state[[node]], symbol)
          success(node)
        }
      } else if (identical(method, "flowerInitDS") &&
                 identical(names(conns), "site_1")) {
        state$site_1 <<- c(state$site_1, symbol)
        success("site_1")
      } else if (identical(method, "flowerInitDS")) {
        error("site_2", "flower admission failed")
      } else {
        success(names(conns)[[1L]])
      }
      invisible(NULL)
    },
    datashield.rm = function(conns, symbol, ...) {
      host <- names(conns)[[1L]]
      state[[host]] <<- setdiff(state[[host]], symbol)
      removed <<- c(removed, paste(host, symbol, sep = ":"))
      invisible(NULL)
    },
    .package = "DSI"
  )

  expect_error(
    ds.flower.nodes.init(
      list(site_1 = NULL, site_2 = NULL), resource = "PROJECT.images"),
    "site_2")
  expect_true(any(grepl("site_1:flower$", removed)))
  expect_length(grep("flower_img$", removed), 2L)
  expect_true(any(grepl(":dsFres\\.", removed)))
})

test_that("node init rollback never replaces the initialization error", {
  state <- list(site = character())
  withr::local_options(list(warn = 2L))
  local_mocked_bindings(
    datashield.symbols = function(conns, ...) state[names(conns)],
    datashield.assign.expr = function(conns, symbol, expr, success, error,
                                      ...) {
      method <- as.character(expr[[1L]])
      if (identical(method, "flowerInitDS")) {
        state$site <<- c(state$site, symbol)
        error("site", "original initialization failure")
      } else {
        error("site", "rollback failure")
      }
      invisible(NULL)
    },
    datashield.rm = function(...) stop("must not remove without destroy ACK"),
    .package = "DSI"
  )

  failure <- tryCatch(
    ds.flower.nodes.init(
      list(site = NULL), data = "D", symbol = "flower"),
    error = identity)
  expect_s3_class(failure, "error")
  expect_match(conditionMessage(failure), "Flower handle initialization")
  expect_false(grepl("Node initialization failed", conditionMessage(failure)))
  expect_identical(state$site, "flower")
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
