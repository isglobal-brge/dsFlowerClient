test_that("native tree requests retain concrete model parameters through fit and submit", {
  captured <- NULL
  build <- dsFlowerClient:::.build_native_tree_request
  local_mocked_bindings(
    .build_native_tree_request = function(...) {
      captured <<- build(...)
      captured
    },
    .validate_dsi_transport_security = function(...) TRUE,
    .assert_native_tree_capability = function(...) stop("captured before transport"),
    .package = "dsFlowerClient")
  for (engine in c("random_forest", "extra_trees", "lightgbm", "catboost", "xgboost")) {
    depth <- if (engine == "catboost") "depth" else "max_depth"
    count <- switch(engine, lightgbm = "num_iterations", catboost = "iterations",
                    xgboost = "num_boost_round", "n_estimators")
    for (task in c("binary", "regression")) {
      params <- list(task = task, n_estimators = 2L)
      params[[depth]] <- 3L
      model <- do.call(ds.flower.model, c(list(name = engine), params))
      common <- list(conns = list(site = TRUE), target = "y", features = "x",
        symbol = "D", feature_bounds = list(lower = -1, upper = 1),
        feature_cuts = list(c(-0.5, 0, 0.5)))
      if (task == "binary") common$target_levels <- c(0, 1) else
        common$target_bounds <- list(lower = -1, upper = 1)
      cases <- list(
        list(fun = ds.flower.fit, args = list(model = engine,
          model_params = params, rounds = 1L)),
        list(fun = ds.flower.submit, args = list(model = model, num_rounds = 1L)),
        list(fun = ds.flower.submit, args = list(model = engine,
          model_params = params, num_rounds = 1L)),
        list(fun = ds.flower.fit, args = list(model = model,
          model_params = list(n_estimators = 3L), rounds = 1L)),
        list(fun = ds.flower.submit, args = list(model = model,
          model_params = list(n_estimators = 3L), num_rounds = 1L)))
      for (i in seq_along(cases)) {
        captured <- NULL
        expect_error(do.call(cases[[i]]$fun, c(common, cases[[i]]$args)),
                     "captured before transport")
        wire <- captured$value
        values <- setNames(lapply(wire$parameters, `[[`, "value"),
                           vapply(wire$parameters, `[[`, character(1), "name"))
        expect_equal(values[[depth]], 3L, info = paste(engine, task, i))
        expect_equal(values[[count]], if (i > 3L) 3L else 2L,
                     info = paste(engine, task, i))
        expect_identical(wire$task, task)
      }
      # A model supplied by name still chooses that task's construction defaults.
      expect_error(do.call(ds.flower.submit, c(common, list(model = engine,
        model_params = list(task = task), num_rounds = 1L))),
        "captured before transport")
      defaults <- ds.flower.model(engine, task = task)$params
      values <- setNames(lapply(captured$value$parameters, `[[`, "value"),
        vapply(captured$value$parameters, `[[`, character(1), "name"))
      expect_equal(values[[count]], defaults[[count]])
    }
  }
})
