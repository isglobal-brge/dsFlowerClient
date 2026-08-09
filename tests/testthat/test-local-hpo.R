test_that("local HPO dimensions are typed and bounded", {
  expect_equal(
    unclass(ds.flower.hpo.float(0.01, 1, log = TRUE)),
    list(type = "float", lower = 0.01, upper = 1, log = TRUE, step = NULL)
  )
  expect_equal(
    unclass(ds.flower.hpo.integer(2L, 10L, step = 2L)),
    list(type = "integer", lower = 2, upper = 10, log = FALSE, step = 2)
  )
  expect_equal(
    unclass(ds.flower.hpo.categorical(c("a", "b"))),
    list(type = "categorical", values = c("a", "b"))
  )

  expect_error(ds.flower.hpo.float(1, 1), "strictly smaller")
  expect_error(ds.flower.hpo.float(0, 1, log = TRUE), "positive")
  expect_error(ds.flower.hpo.float(0.01, 1, log = TRUE, step = 0.1),
               "cannot be combined")
  expect_error(ds.flower.hpo.float(0, 1, step = 0.3), "divide")
  expect_error(ds.flower.hpo.integer(0L, 10L, log = TRUE), "lower >= 1")
  expect_error(ds.flower.hpo.integer(1L, 10L, step = 4L), "divide")
  expect_error(ds.flower.hpo.categorical(c("a", "a")), "unique")
  expect_error(ds.flower.hpo.categorical(c(1, Inf)), "finite")
})

test_that("local HPO request is canonical and has no storage field", {
  first <- list(
    z = ds.flower.hpo.categorical("only"),
    a = ds.flower.hpo.integer(1L, 3L)
  )
  second <- rev(first)
  one <- dsFlowerClient:::.hpo_request(first, 1000000L, "maximize", 42L)
  two <- dsFlowerClient:::.hpo_request(second, 1000000L, "maximize", 42L)

  expect_identical(one, two)
  expect_identical(names(one$space), c("a", "z"))
  expect_identical(one$space$z$values, list("only"))
  expect_false("storage" %in% names(one))
  expect_error(
    dsFlowerClient:::.hpo_request(first, 1000001L, "maximize", 42L),
    "1000000"
  )
  expect_error(
    dsFlowerClient:::.hpo_request(unname(first), 1L, "minimize", 0L),
    "parameter names"
  )
  expect_error(
    dsFlowerClient:::.hpo_request(list(x = list(type = "float")),
                                 1L, "minimize", 0L),
    "dimension constructor"
  )
})

test_that("local HPO backend is provisioned at the tested exact version", {
  expect_identical(
    grep("^optuna", dsFlowerClient:::.DSFLOWER_CLIENT_PYTHON_DEPS,
         value = TRUE),
    "optuna==4.8.0"
  )
})

test_that("Optuna HPO is reproducible and leaves no local study files", {
  python <- tryCatch(
    dsFlowerClient:::.local_hpo_python_cmd(), error = function(e) "")
  skip_if(!nzchar(python), "Optuna 4.8.0 is required")
  local_mocked_bindings(
    .local_hpo_python_cmd = function() python,
    .package = "dsFlowerClient"
  )
  directory <- withr::local_tempdir()
  withr::local_dir(directory)
  calls <- 0L
  objective <- function(params) {
    calls <<- calls + 1L
    (params$x - 0.2)^2 + (params$depth - 4)^2
  }
  space <- list(
    x = ds.flower.hpo.float(0.01, 1, log = TRUE),
    depth = ds.flower.hpo.integer(2L, 8L),
    engine = ds.flower.hpo.categorical(c("xgboost", "random_forest"))
  )

  first <- ds.flower.hpo(objective, space, n_trials = 12L,
                         direction = "minimize", seed = 71L)
  second <- ds.flower.hpo(objective, rev(space), n_trials = 12L,
                          direction = "minimize", seed = 71L)
  tied <- ds.flower.hpo(
    function(params) {
      calls <<- calls + 1L
      1
    },
    list(only = ds.flower.hpo.categorical("value")),
    n_trials = 3L, direction = "maximize", seed = 71L
  )

  expect_s3_class(first, "dsflower_hpo")
  expect_identical(first$trials, second$trials)
  expect_identical(first$best_params, second$best_params)
  expect_identical(first$best_value, second$best_value)
  expect_identical(first$optuna_version, "4.8.0")
  expect_identical(first$sampler, "TPESampler")
  expect_false("storage" %in% names(first))
  expect_identical(tied$best_number, 0L)
  expect_identical(tied$best_value, 1)
  expect_identical(calls, 27L)
  expect_identical(list.files(directory, all.files = TRUE, no.. = TRUE),
                   character())
})

test_that("objective failures are bounded and terminate the helper", {
  python <- tryCatch(
    dsFlowerClient:::.local_hpo_python_cmd(), error = function(e) "")
  skip_if(!nzchar(python), "Optuna 4.8.0 is required")
  local_mocked_bindings(
    .local_hpo_python_cmd = function() python,
    .package = "dsFlowerClient"
  )
  long_message <- paste(rep("private-detail", 100L), collapse = "\n")
  error <- tryCatch(
    ds.flower.hpo(
      function(params) stop(long_message),
      list(x = ds.flower.hpo.float(0, 1)), n_trials = 2L
    ),
    error = identity
  )
  expect_s3_class(error, "error")
  expect_lte(nchar(conditionMessage(error)), 560L)
  expect_false(grepl("\n", conditionMessage(error), fixed = TRUE))

  expect_error(
    ds.flower.hpo(
      function(params) Inf,
      list(x = ds.flower.hpo.float(0, 1)), n_trials = 1L
    ),
    "finite number"
  )
})

test_that("packaged HPO helper has no persistence or code transport", {
  path <- system.file("python", "local_hpo_helper.py",
                      package = "dsFlowerClient")
  source <- paste(readLines(path, warn = FALSE), collapse = "\n")
  r_source <- paste(deparse(body(ds.flower.hpo)), collapse = "\n")

  expect_match(source, "storage=None", fixed = TRUE)
  expect_false(grepl("sqlite|RDBStorage|JournalStorage|pickle|cloudpickle",
                     source, ignore.case = TRUE))
  expect_false(grepl("serialize|deparse|body\\(", r_source))
  expect_match(r_source, "kill_tree", fixed = TRUE)
})
