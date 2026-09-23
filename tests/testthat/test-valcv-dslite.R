test_that("DSLite executes checkpoint validation and initialized patient CV with private pooling", {
  skip_if_not_installed("dsFlower", minimum_version = "0.7.0")
  skip_if_not_installed("DSLite")
  skip_if_not_installed("resourcer")
  python <- dsFlowerClient:::.client_python_cmd()
  encoder <- Sys.getenv("DSFLOWER_TEST_ENCODER", unset = path.expand(
    "~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"))
  skip_if_not(file.exists(encoder), "requires the custodian-preseeded pinned encoder")
  helper <- normalizePath(test_path("..", "..", "tools", "integration",
    "validation-cv-fixture.py"), mustWork = TRUE)
  runner <- dirname(system.file("flower_app", "dsflower_runner", package = "dsFlower"))
  root <- normalizePath(withr::local_tempdir("valcv-dslite-"), winslash = "/")
  fixture <- file.path(root, "fixture")
  made <- processx::run(python, c("-I", helper, "fixture", "--runner", runner,
    "--root", fixture, "--encoder", encoder), error_on_status = FALSE, timeout = 180)
  expect_identical(made$status, 0L, info = made$stderr)
  if (made$status != 0L) return(invisible(NULL))
  identities <- jsonlite::fromJSON(made$stdout)
  secret_dir <- file.path(root, "privacy")
  dir.create(secret_dir, mode = "0700")
  withr::local_envvar(c(DSFLOWER_NODE_SECRET_FILE = file.path(secret_dir, "noise_root"),
    DSFLOWER_TEST_ALLOW_EPHEMERAL_SECRET = "1", OMP_NUM_THREADS = "1",
    OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1"))
  withr::local_options(list(dsflower.dp_unit = "patient", dsflower.patient_column = "patient_id",
    dsflower.image_data_root = file.path(fixture, "images"),
    dsflower.mask_data_root = file.path(fixture, "masks"),
    dsflower.staging_root = file.path(root, "staging"),
    dsflower.checkpoint_cache_dir = file.path(root, "checkpoint-cache"),
    dsflower.public_initialisation = "analyst_or_resource",
    datashield.progress = FALSE, datashield.errors.print = FALSE))
  local_mocked_bindings(.resolve_framework_runtime = function(...) list(
    python = python, venv_path = dirname(dirname(python))), .package = "dsFlower")
  dsFlower:::.ensure_node_secret()
  resource <- function(file, hash) resourcer::newResource(name = "synthetic",
    url = paste0("file://", file.path(fixture, file)),
    format = paste0("dsflower-checkpoint-v1:", hash))
  resources <- list(PublicModels.tabular = resource("tabular.zip", identities$tabular$bundle_sha256),
    PublicModels.segmentation = resource("bundle.zip", identities$segmentation$bundle_sha256))
  servers <- conns <- list()
  description <- utils::packageDescription("dsFlower")
  for (site in c("alpha", "beta")) {
    id <- seq_len(32L)
    data <- data.frame(patient_id = paste0(site, "p", id), image_id = paste0("i", id),
      relative_path = paste0(id %% 2L, ".png"), mask_path = paste0(id %% 2L, ".png"),
      x1 = (id %% 9L) / 5 - 0.8, x2 = (id %% 7L) / 4 - 0.7,
      outcome = as.integer(id %% 2L), time = as.numeric(id %% 10L + 1L),
      event = as.integer(id %% 3L != 0L))
    server <- DSLite::newDSLiteServer(tables = list(training = data), resources = resources,
      config = list(), strict = TRUE, home = file.path(root, paste0("dslite-", site)))
    for (field in c("AssignMethods", "AggregateMethods")) {
      methods <- trimws(strsplit(description[[field]], ",")[[1L]])
      for (method in methods) {
        if (field == "AssignMethods") server$assignMethod(method, paste0("dsFlower::", method))
        else server$aggregateMethod(method, paste0("dsFlower::", method))
      }
    }
    server_name <- paste0("valcv_dslite_", site, "_", Sys.getpid())
    assign(server_name, server, .GlobalEnv)
    servers[[site]] <- server
    conns[[site]] <- DSI::dsConnect(DSLite::DSLite(), name = site, url = server_name)
    secret <- file.path(secret_dir, site)
    writeLines(strrep(if (site == "alpha") "31" else "42", 32L), secret)
    Sys.chmod(secret, "0600")
  }
  withr::defer({
    for (connection in conns) DSI::dsDisconnect(connection)
    rm(list = paste0("valcv_dslite_", c("alpha", "beta"), "_", Sys.getpid()), envir = .GlobalEnv)
  })
  DSI::datashield.assign.table(conns, "D", "training")
  DSI::datashield.assign.resource(conns, "TAB_R", "PublicModels.tabular")
  DSI::datashield.assign.expr(conns, "TAB", quote(flowerCheckpointInitDS("TAB_R")))
  DSI::datashield.assign.resource(conns, "SEG_R", "PublicModels.segmentation")
  DSI::datashield.assign.expr(conns, "SEG", quote(flowerCheckpointInitDS("SEG_R")))

  active <- conns
  reports <- list()
  client_env <- dsFlowerClient:::.dsflower_client_env
  old_superlink <- client_env$.superlink
  withr::defer(client_env$.superlink <- old_superlink)
  client_env$.superlink <- list(flwr_home = root, process = list(is_alive = function() TRUE))
  # Real high-level client calls and DSI preparation; only local Flower transport
  # and runtime provisioning are substituted. No training/prediction/DP mocks.
  local_mocked_bindings(
    .require_flwr_cli = function(...) TRUE,
    .ensure_client_framework = function(...) TRUE,
    ds.flower.link.up = function(...) TRUE,
    ds.flower.link.down = function(...) TRUE,
    ds.flower.nodes.ensure = function(...) TRUE,
    .run_flwr_with_artifact_watchdog = function(args, ...) {
      nodes <- lapply(names(active), function(site) {
        session <- servers[[site]]$getSession(active[[site]]@sid)
        state <- dsFlower:::.flower_session_state(session)
        handles <- lapply(state$handles, function(entry) entry$handle)
        staged <- Filter(function(handle) !is.null(handle$staging_dir) &&
          file.exists(file.path(handle$staging_dir, "manifest.json")), handles)
        stopifnot(length(staged) == 1L)
        list(directory = staged[[1L]]$staging_dir, secret = file.path(secret_dir, site))
      })
      node_file <- file.path(root, "nodes.json")
      report_file <- file.path(root, "execution.json")
      jsonlite::write_json(nodes, node_file, auto_unbox = TRUE)
      result <- processx::run(python, c("-I", helper, "execute", "--runner", runner,
        "--app", args[[2L]], "--nodes", node_file, "--report", report_file),
        error_on_status = FALSE, timeout = 300)
      expect_identical(result$status, 0L, info = paste(result$stdout, result$stderr))
      if (result$status == 0L) reports[[length(reports) + 1L]] <<- jsonlite::fromJSON(report_file)
      result
    }, .package = "dsFlowerClient")

  tabular_bundle <- paste0("client:", file.path(fixture, "tabular.zip"))
  segmentation_bundle <- paste0("client:", file.path(fixture, "bundle.zip"))
  for (selection in list(conns, conns["alpha"])) {
    active <- selection
    result <- ds.flower.validate(active, model = tabular_bundle, target = "outcome", symbol = "D", silent = TRUE)
    expect_s3_class(result, "dsflower_validation")
    expect_true(result$available)
    expect_equal(result$n_nodes, length(active))
    expect_true(is.finite(result$metrics$accuracy))
    expect_true(result$pooled_only)
  }
  active <- conns
  for (bundle in list(list(model = segmentation_bundle, target = "mask_path", metric = "foreground_dice"),
                      list(model = paste0("client:", file.path(fixture, "survival.zip")),
                           target = c("time", "event"), metric = "negative_log_likelihood"))) {
    result <- ds.flower.validate(active, model = bundle$model, target = bundle$target, symbol = "D", silent = TRUE)
    expect_true(result$available)
    expect_true(is.finite(result$metrics[[bundle$metric]]))
    expect_false("concordance" %in% names(result$metrics))
  }
  for (policy in c("resource_only", "none")) {
    staged_before <- list.files(file.path(root, "staging"), recursive = TRUE, all.files = TRUE)
    withr::with_options(list(dsflower.public_initialisation = policy), {
      expect_error(ds.flower.validate(conns, model = tabular_bundle, target = "outcome", symbol = "D", silent = TRUE),
                   "policy|permit|admit|disabled|allowed|forbid|initialisation|DataSHIELD errors")
    })
    expect_identical(list.files(file.path(root, "staging"), recursive = TRUE, all.files = TRUE), staged_before)
  }
  bad <- file.path(root, "changed-bundle")
  dir.create(bad)
  file.copy(list.files(file.path(fixture, "tabular"), full.names = TRUE), bad)
  manifest_path <- file.path(bad, "manifest.json")
  manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
  manifest$checkpoint$sha256 <- strrep("e", 64L)
  jsonlite::write_json(manifest, manifest_path, auto_unbox = TRUE)
  expect_error(ds.flower.validate(conns, model = paste0("client:", bad), target = "outcome", symbol = "D"),
               "digest|checkpoint|Checkpoint")
  staged_before <- list.files(file.path(root, "staging"), recursive = TRUE, all.files = TRUE)
  expect_error(ds.flower.cross_validate(conns, symbol = "D", target = "outcome",
    features = "x1", model = ds.flower.model.pytorch_logreg(),
    feature_bounds = list(lower = -1, upper = 1), target_levels = c(0, 1),
    public_initialisation = tabular_bundle, folds = 2L, rounds = 2L, silent = TRUE),
    "geometry|contract|checkpoint|preparation")
  expect_identical(list.files(file.path(root, "staging"), recursive = TRUE, all.files = TRUE), staged_before)

  for (route in c("client", "resource")) {
    init <- if (route == "client") tabular_bundle else "resource:TAB"
    cv <- ds.flower.cross_validate(conns, symbol = "D", target = "outcome",
      features = c("x1", "x2"), model = ds.flower.model.pytorch_logreg(batch_size = 8L),
      feature_bounds = list(lower = c(-1, -1), upper = c(1, 1)), target_levels = c(0, 1),
      public_initialisation = init,
      public_checkpoint_file = if (route == "resource") file.path(fixture, "tabular", "checkpoint.npz") else NULL,
      folds = 2L, rounds = 2L, silent = TRUE)
    expect_s3_class(cv, "dsflower_cv")
    expect_equal(cv$folds, 2L)
    expect_identical(list.files(cv$results_dir), "cv.json")
    init <- if (route == "client") segmentation_bundle else "resource:SEG"
    model <- ds.flower.model.pytorch_resnet18_segmentation(decoder = "narrow", decoder_init = init,
      batch_size = 8L, local_epochs = 1L, learning_rate = 0.01)
    cv <- ds.flower.cross_validate(conns, symbol = "D", target = "mask_path", model = model,
      task = "segmentation", data_kind = "image", folds = 2L, rounds = 2L, silent = TRUE,
      public_checkpoint_file = if (route == "resource") file.path(fixture, "bundle", "checkpoint.npz") else NULL)
    expect_s3_class(cv, "dsflower_cv")
    expect_true(is.finite(cv$metrics$foreground_dice))
    expect_identical(list.files(cv$results_dir), "cv.json")
  }
  expect_length(reports, 8L)
  expect_true(all(vapply(reports, `[[`, logical(1), "identity_and_partition_checks")))
  expect_true(all(vapply(tail(reports, 4L), `[[`, logical(1), "folds_start_from_same_checkpoint")))
  expect_equal(vapply(tail(reports, 4L), `[[`, integer(1), "dp_training_rounds"), rep(8L, 4L))
  expect_true(all(vapply(reports[3:4], `[[`, logical(1), "plain_metrics_match")))
  expect_true(all(vapply(reports[3:4], `[[`, logical(1), "pooled_noise_within_six_sigma")))

  # Exercise atomic holdout's final model (one training round catches accidental
  # reapplication of the starting checkpoint during evaluation) and saved artifacts.
  tabular_holdout <- ds.flower.fit(conns, symbol = "D", target = "outcome",
    features = c("x1", "x2"), model = ds.flower.model.pytorch_logreg(batch_size = 8L),
    feature_bounds = list(lower = c(-1, -1), upper = c(1, 1)), target_levels = c(0, 1),
    public_initialisation = "resource:TAB",
    public_checkpoint_file = file.path(fixture, "tabular", "checkpoint.npz"),
    holdout = 0.2, rounds = 1L, output_dir = file.path(root, "tabular-holdout"), silent = TRUE)
  expect_s3_class(tabular_holdout, "dsflower_run")
  expect_true(is.finite(tabular_holdout$holdout$accuracy))
  seg_model <- ds.flower.model.pytorch_resnet18_segmentation(decoder = "narrow",
    decoder_init = "resource:SEG", batch_size = 8L, local_epochs = 1L)
  seg_holdout <- ds.flower.fit(conns, symbol = "D", target = "mask_path", model = seg_model,
    task = "segmentation", data_kind = "image", holdout = 0.2, rounds = 1L,
    public_checkpoint_file = file.path(fixture, "bundle", "checkpoint.npz"),
    output_dir = file.path(root, "segmentation-holdout"), silent = TRUE)
  expect_true(is.finite(seg_holdout$holdout$foreground_dice))
  saved_segmentation <- ds.flower.validate(conns, model = seg_holdout, target = "mask_path",
    symbol = "D", silent = TRUE)
  expect_true(saved_segmentation$available)
  survival_model <- ds.flower.model.pytorch_aft(horizon = 10, batch_size = 8L)
  survival_holdout <- ds.flower.fit(conns, symbol = "D", target = c("time", "event"),
    features = c("x1", "x2"), model = survival_model, task = "survival",
    feature_bounds = list(lower = c(-1, -1), upper = c(1, 1)),
    public_initialisation = paste0("client:", file.path(fixture, "survival.zip")),
    survival_horizons = c(5, 10), holdout = 0.2, rounds = 1L,
    output_dir = file.path(root, "survival-holdout"), silent = TRUE)
  # A small holdout can have a nonpositive noised valid count. Its documented
  # missing NLL is a valid pooled release, not an execution failure.
  holdout_nll <- survival_holdout$holdout$negative_log_likelihood
  expect_identical(is.null(holdout_nll), survival_holdout$holdout$n <= 0)
  expect_true(is.null(holdout_nll) || (is.finite(holdout_nll) && abs(holdout_nll) <= 20))
  expect_length(survival_holdout$holdout$brier$scores, 2L)
  saved_survival <- ds.flower.validate(conns, model = survival_holdout,
    target = c("time", "event"), symbol = "D", survival_horizons = c(5, 10), silent = TRUE)
  expect_true(saved_survival$available)
  survival_cv <- ds.flower.cross_validate(conns, symbol = "D", target = c("time", "event"),
    features = c("x1", "x2"), model = survival_model, task = "survival",
    feature_bounds = list(lower = c(-1, -1), upper = c(1, 1)),
    public_initialisation = paste0("client:", file.path(fixture, "survival.zip")),
    survival_horizons = c(5, 10), folds = 2L, rounds = 2L, silent = TRUE)
  expect_s3_class(survival_cv, "dsflower_cv")
  expect_length(survival_cv$metrics$brier$scores, 2L)
  expect_false("concordance" %in% names(survival_cv$metrics))
  expect_length(reports, 14L)
})
