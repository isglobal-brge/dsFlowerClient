test_that("both public initialization routes run two DP rounds through native DSLite admission", {
  skip_if_not_installed("dsFlower", minimum_version = "0.7.0")
  skip_if_not_installed("DSLite")
  skip_if_not_installed("resourcer")
  skip_if_not_installed("httr")
  python <- dsFlowerClient:::.client_python_cmd()
  encoder <- Sys.getenv("DSFLOWER_TEST_ENCODER", unset = path.expand(
    "~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"))
  skip_if_not(file.exists(encoder), "requires the custodian-preseeded pinned encoder")
  helper <- normalizePath(test_path("..", "..", "tools", "integration",
    "public-initialisation-fixture.py"), mustWork = TRUE)
  runner <- dirname(system.file("flower_app", "dsflower_runner", package = "dsFlower"))
  root <- normalizePath(withr::local_tempdir("public-init-dslite-"), winslash = "/")
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
  # Select this already-installed runtime; all bundle/encoder checks and DP
  # preparation/training execute the production implementation.
  local_mocked_bindings(.resolve_framework_runtime = function(...) list(
    python = python, venv_path = dirname(dirname(python))), .package = "dsFlower")
  dsFlower:::.ensure_node_secret()
  resource <- resourcer::newResource(name = "synthetic",
    url = paste0("file://", file.path(fixture, "bundle.zip")),
    format = paste0("dsflower-checkpoint-v1:", identities$bundle_sha256))
  certificate <- file.path(root, "localhost.crt")
  key <- file.path(root, "localhost.key")
  openssl <- Sys.which("openssl")
  expect_true(nzchar(openssl))
  ssl_config <- file.path(root, "localhost.cnf")
  writeLines(c("[req]", "distinguished_name=dn", "x509_extensions=ext", "prompt=no",
    "[dn]", "CN=localhost", "[ext]", "subjectAltName=DNS:localhost",
    "basicConstraints=CA:TRUE"), ssl_config)
  cert <- processx::run(openssl, c("req", "-x509", "-newkey", "rsa:2048", "-nodes",
    "-keyout", key, "-out", certificate, "-days", "1", "-config", ssl_config),
    error_on_status = FALSE, timeout = 30)
  expect_identical(cert$status, 0L, info = cert$stderr)
  https <- processx::process$new(python, c("-I", helper, "serve", "--runner", runner,
    "--root", fixture, "--certificate", certificate, "--key", key),
    stdout = "|", stderr = "|")
  withr::defer(https$kill())
  port_file <- file.path(fixture, "https-port.json")
  for (attempt in seq_len(100L)) {
    if (file.exists(port_file) || !https$is_alive()) break
    Sys.sleep(0.05)
  }
  expect_true(file.exists(port_file))
  https_url <- paste0("https://localhost:", jsonlite::fromJSON(port_file), "/")
  https_resource <- function(file, hash = identities$bundle_sha256) resourcer::newResource(
    name = "synthetic_https", url = paste0(https_url, file),
    format = paste0("dsflower-checkpoint-v1:", hash))
  data <- data.frame(patient_id = paste0("p", 1:4), image_id = paste0("i", 1:4),
    relative_path = rep(c("0.png", "1.png"), 2L), mask_path = rep(c("0.png", "1.png"), 2L))
  server <- DSLite::newDSLiteServer(tables = list(training = data),
    resources = list(PublicModels.synthetic = resource,
      PublicModels.https = https_resource("bundle.zip"),
      PublicModels.missing = https_resource("missing.zip"),
      PublicModels.changed = https_resource("bundle.zip", strrep("e", 64L))),
    config = list(), strict = TRUE,
    home = file.path(root, "dslite"))
  for (method in c("flowerInitDS", "flowerCheckpointInitDS", "flowerPrepareRunDS",
                   "flowerCleanupRunDS", "flowerDestroyDS")) {
    server$assignMethod(method, paste0("dsFlower::", method))
  }
  for (method in c("flowerStatusDS", "flowerCheckpointStatusDS", "flowerCheckpointUploadDS")) {
    server$aggregateMethod(method, paste0("dsFlower::", method))
  }
  server_name <- paste0("public_initialisation_dslite_", Sys.getpid())
  assign(server_name, server, .GlobalEnv)
  withr::defer(rm(list = server_name, envir = .GlobalEnv))
  connection <- DSI::dsConnect(DSLite::DSLite(), name = "site", url = server_name)
  withr::defer(DSI::dsDisconnect(connection))
  conns <- list(site = connection)
  DSI::datashield.assign.table(conns, "D", "training")
  DSI::datashield.assign.resource(conns, "CKPT_R", "PublicModels.synthetic")
  DSI::datashield.assign.expr(conns, "CKPT", quote(flowerCheckpointInitDS("CKPT_R")))
  httr::with_config(httr::config(cainfo = certificate), {
    DSI::datashield.assign.resource(conns, "HTTPS_R", "PublicModels.https")
    DSI::datashield.assign.expr(conns, "HTTPS_CKPT", quote(flowerCheckpointInitDS("HTTPS_R")))
    https_summary <- DSI::datashield.aggregate(conns, quote(flowerCheckpointStatusDS("HTTPS_CKPT")))
    expect_identical(https_summary$site$manifest_sha256, identities$manifest_sha256)
    expect_error(DSI::dsFetch(DSI::dsAssignResource(connection, "MISSING_R", "PublicModels.missing")),
                 "Checkpoint resource acquisition")
    expect_error(DSI::dsFetch(DSI::dsAssignResource(connection, "CHANGED_R", "PublicModels.changed")),
                 "checkpoint|Checkpoint")
  })
  expect_false(exists("MISSING_R", server$getSession(connection@sid), inherits = FALSE))
  expect_false(exists("CHANGED_R", server$getSession(connection@sid), inherits = FALSE))
  expect_false(dir.exists(file.path(root, "staging", "dsflower")))
  reference <- server$getSessionData(connection@sid, "CKPT")
  expect_s3_class(reference, "dsflower_checkpoint_ref")
  expect_named(reference, "capability")
  route_records <- list()
  for (route in c("client", "resource")) {
    init <- if (route == "client") paste0("client:", file.path(fixture, "bundle.zip")) else "resource:CKPT"
    model <- ds.flower.model.pytorch_resnet18_segmentation(decoder = "narrow", decoder_init = init,
      batch_size = 2L, local_epochs = 1L, learning_rate = 0.01)
    sub <- dsFlowerClient:::.emit_submission(model)
    local <- dsFlowerClient:::.segmentation_client_initialization(conns, sub$params,
      if (route == "resource") file.path(fixture, "bundle", "checkpoint.npz") else NULL)
    if (!is.null(local$uploads)) {
      uploaded_tokens <- local$uploads
      withr::defer(dsFlowerClient:::.segmentation_abort_uploads(conns, uploaded_tokens))
    }
    DSI::datashield.assign.expr(conns, "flower", quote(flowerInitDS("D")))
    p <- sub$params
    config <- c(list("dp-track" = "neural", data_type = "image", "task-type" = "segmentation",
      "loss-name" = sub$loss, "num-features" = 32768L, "num-classes" = 2L,
      "num-server-rounds" = 2L, "batch-size" = 2L, "local-epochs" = 1L,
      "model-spec-b64" = dsFlowerClient:::.spec_to_b64(sub$spec),
      backbone = p$backbone, "image-size" = p$image_size,
      "vision-extractor-profile" = p$vision_extractor_profile,
      image_asset = "images", mask_asset = "masks", image_path_col = "relative_path",
      mask_path_col = "mask_path", sample_id_col = "image_id"),
      dsFlowerClient:::.segmentation_public_config(p),
      dsFlowerClient:::.neural_training_config(p, sub$loss))
    prepared <- dsFlowerClient:::.segmentation_prepare_nodes(conns, "flower", "mask_path",
      NULL, config, local)
    matched <- dsFlowerClient:::.segmentation_server_initialization(prepared, p, "site", local)
    expect_identical(matched$provenance$provenance$manifest_sha256, identities$manifest_sha256)
    expect_false(any(c("checkpoint_base64", "local_arrays_b64", "snapshot_directory") %in%
      names(prepared$per_site$site$public_initialisation)))
    # Fixture custodian access to DSLite's session retrieves the private staging
    # directory. This is deliberately absent from every aggregate result.
    session <- server$getSession(connection@sid)
    handle_ref <- get("flower", session)
    state <- dsFlower:::.flower_session_state(session)
    handle <- state$handles[[handle_ref$capability]]$handle
    manifest <- jsonlite::fromJSON(file.path(handle$staging_dir, "manifest.json"), simplifyVector = FALSE)
    expected_origin <- if (route == "client") "analyst-declared" else paste0("resource:", identities$manifest_sha256)
    expect_identical(manifest$initialisation, expected_origin)
    expect_identical(manifest[["public-initialisation-policy"]], "analyst_or_resource")
    trained <- processx::run(python, c("-I", helper, "train", "--runner", runner,
      "--root", fixture, "--manifest", handle$staging_dir),
      error_on_status = FALSE, timeout = 240)
    expect_identical(trained$status, 0L, info = trained$stderr)
    if (trained$status == 0L) {
      report <- jsonlite::fromJSON(trained$stdout, simplifyVector = FALSE)
      expect_length(report$rounds, 2L)
      expect_true(report$private_training_executed)
      expect_true(report$identity_changed_with_digest)
      expect_true(report$identity_unchanged_with_locator)
      route_records[[route]] <- report
    }
    DSI::datashield.assign.expr(conns, "flower", quote(flowerDestroyDS("flower")))
    DSI::datashield.rm(conns, "flower")
  }
  expect_named(route_records, c("client", "resource"))
})
