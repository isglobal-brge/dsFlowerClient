#!/usr/bin/env Rscript
# R4 actual DSLite federation. Only the explicitly supplied training collection is read.
args <- commandArgs(TRUE)
stopifnot(length(args) == 2L)
root <- normalizePath(args[[1]])
job_path <- normalizePath(args[[2]])
job <- jsonlite::fromJSON(job_path)
epsilon <- as.numeric(job$epsilon)
seed <- as.integer(job$seed)
contract <- "pytorch_resnet18"
tools <- file.path(root, "src/dsFlowerClient/tools/campaign/vision")
source(file.path(tools, "../campaign_lib.R"))
stopifnot(length(epsilon) == 1L, epsilon %in% c(1, 4, 8),
          length(seed) == 1L, !is.na(seed),
          is.character(job$run_dir), startsWith(job$run_dir, "/"),
          is.character(job$collection_root), startsWith(job$collection_root, "/"),
          length(job$n_patients_per_site) == 3L,
          all(job$n_patients_per_site > 0L),
          all(job$n_patients_per_site == as.integer(job$n_patients_per_site)),
          is.list(job$model_params),
          is.character(job$split_sha256),
          grepl("^[0-9a-f]{64}$", job$split_sha256))
work <- job$run_dir
if (dir.exists(work)) stop("Existing run directory; retain all attempts and refuse overwrite.")
dir.create(work, recursive = TRUE)
file.copy(job_path, file.path(work, "job.json"))
capture <- file.path(work, "public-capture")
dir.create(capture)
Sys.setenv(F_VISION_PUBLIC_BENCHMARK = "1", F_VISION_INIT_SEED = seed,
           F_VISION_CAPTURE_DIR = capture, CUBLAS_WORKSPACE_CONFIG = ":4096:8",
           OMP_NUM_THREADS = "2", MKL_NUM_THREADS = "2", OPENBLAS_NUM_THREADS = "2",
           PYTHONPATH = paste(file.path(tools, "benchmark_hooks"),
             file.path(root, "src/dsFlowerClient/inst/flower_app"), sep = ":"))
collection_root <- normalizePath(job$collection_root)
for (index in seq_len(3L)) {
  samples <- utils::read.csv(file.path(collection_root, paste0("site", index), "samples.csv"),
                             colClasses = "character", check.names = FALSE)
  stopifnot("subject_id" %in% names(samples),
            length(unique(samples$subject_id)) == job$n_patients_per_site[[index]])
}
Sys.setenv(TORCH_HOME = file.path(root, "torch"), XDG_CACHE_HOME = root)
ports <- .campaign_free_ports(6L)
venv <- Sys.getenv("DSFLOWER_VENV_ROOT")
started <- Sys.time()
cluster <- parallel::makePSOCKcluster(3L, outfile = "")
conns <- list()
cleanup_ok <- FALSE
cleanup <- function() {
  if (length(conns)) try(ds.flower.link.down(conns), silent = TRUE)
  if (!is.null(cluster)) {
    drained <- try(parallel::clusterCall(cluster, function() {
      cid <- dsFlower:::.dsflower_env$tunnel_conn_id
      if (!is.null(cid)) try(dsFlower::flowerTunnelDownDS(cid), silent = TRUE)
      nodes <- dsFlower:::.supernode_list()
      for (path in nodes$manifest_dir) try(dsFlower:::.supernode_stop(path), silent = TRUE)
      if (exists(".campaign_dslite_conn", .GlobalEnv)) try(DSI::dsDisconnect(.campaign_dslite_conn), silent = TRUE)
      is.null(dsFlower:::.dsflower_env$tunnel_conn_id) && nrow(dsFlower:::.supernode_list()) == 0L
    }), silent = TRUE)
    cleanup_ok <<- !inherits(drained, "try-error") && all(vapply(drained, isTRUE, logical(1)))
    try(parallel::stopCluster(cluster), silent = TRUE)
    cluster <<- NULL
  }
  try(ds.flower.superlink.stop(), silent = TRUE)
  cleanup_ok <<- isTRUE(cleanup_ok) && !isTRUE(ds.flower.superlink.status()$running)
}
result <- tryCatch({
  parallel::clusterMap(cluster, function(index, libpaths, venv, work, epsilon, capture, seed, collection_root) {
    .libPaths(libpaths)
    secret_dir <- file.path(tempdir(), "vision-node-state")
    dir.create(secret_dir, mode = "0700")
    observer <- file.path(secret_dir, "vision-public-benchmark.json")
    jsonlite::write_json(list(public_fixture_only = TRUE, dataset = "busbra", capture_dir = capture, seed = seed),
                         observer, auto_unbox = TRUE)
    Sys.chmod(observer, "0600")
    worker_venv <- file.path(secret_dir, "venvs")
    dir.create(worker_venv, mode = "0700")
    for (framework in c("pytorch", "pytorch-gpu")) {
      stopifnot(file.symlink(normalizePath(file.path(venv, framework)), file.path(worker_venv, framework)))
    }
    Sys.setenv(DSFLOWER_VENV_ROOT = worker_venv,
      DSFLOWER_NODE_SECRET_FILE = file.path(secret_dir, paste0("secret-site", index)),
      DSFLOWER_TEST_ALLOW_EPHEMERAL_SECRET = "1")
    options(dsflower.venv_root = worker_venv, dsflower.dp_unit = "patient",
      dsflower.patient_column = "subject_id", dsflower.dp_per_training_epsilon = epsilon,
      dsflower.dp_per_training_delta = 1e-6, dsflower.dp_clipping_norm = 1)
    collection <- file.path(collection_root, paste0("site", index))
    options(dsimaging.registry_path = file.path(collection, "registry.yaml"))
    suppressPackageStartupMessages({library(DSI); library(DSLite); library(dsFlower); library(dsImaging)})
    resource <- resourcer::newResource(name = "images", url = paste0("imaging+dataset://busbra.site", index))
    server <- DSLite::newDSLiteServer(resources = list(images = resource),
      config = DSLite::defaultDSConfiguration(include = c("dsFlower", "dsImaging")),
      home = file.path(work, paste0("dslite", index)))
    symbol <- paste0("vision_site", index, "_", Sys.getpid())
    assign(symbol, server, .GlobalEnv)
    .campaign_dslite_conn <<- DSLite::dsConnect(DSLite::DSLite(), name = paste0("site", index), url = symbol)
    DSI::dsFetch(DSI::dsAssignResource(.campaign_dslite_conn, "images_resource", "images", async = FALSE))
    DSI::dsFetch(DSI::dsAssignExpr(.campaign_dslite_conn, "D", quote(imagingInitDS("images_resource")), async = FALSE))
    TRUE
  }, index = seq_len(3L), MoreArgs = list(libpaths = .libPaths(), venv = venv,
       work = work, epsilon = epsilon, capture = capture, seed = seed, collection_root = collection_root), SIMPLIFY = FALSE)
  dummy <- DSLite::newDSLiteServer(tables = list())
  conns <- setNames(lapply(seq_len(3), function(i) methods::new("CampaignDSLiteConnection",
    name = paste0("site", i), sid = paste0("remote-site", i), server = dummy, worker = cluster[i])), paste0("site", 1:3))
  policies <- lapply(conns, function(conn) DSI::dsFetch(DSI::dsAggregate(conn, quote(flowerPrivacyPolicyDS()), async = FALSE)))
  capabilities <- lapply(conns, function(conn) DSI::dsFetch(DSI::dsAggregate(conn, quote(flowerGetCapabilitiesDS()), async = FALSE)))
  jsonlite::write_json(list(policies = policies, capabilities = capabilities), file.path(work, "node-contract.json"),
                       pretty = TRUE, auto_unbox = TRUE, digits = NA)
  options(dsflower.tunnel_port = ports[4:6], dsflower.superlink_insecure = TRUE,
          dsflower.tunnel_loss_tolerance = 30, dsflower.supernode_term_grace = 10)
  tryCatch({
    ds.flower.superlink.start(fleet_port = ports[1], control_port = ports[2], serverappio_port = ports[3], insecure = TRUE)
  }, error = function(e) {
    if (!startsWith(conditionMessage(e), "SuperLink did not become ready within 15 seconds.\nLog:\n")) stop(e)
    info <- dsFlowerClient:::.dsflower_client_env$.superlink
    jsonlite::write_json(list(reason = conditionMessage(e), default_wait_seconds = 15,
      additional_wait_seconds = 90, model_configuration_changed = FALSE,
      action = "Continue waiting for the same SuperLink using the unchanged readiness helper."),
      file.path(work, "startup-wait.json"), pretty = TRUE, auto_unbox = TRUE)
    dsFlowerClient:::.wait_superlink_ready(info$process, ports[1], info$log_path, timeout = 90)
    dsFlowerClient:::.write_superlink_pid(info)
  })
  fit <- ds.flower.fit(conns, symbol = "D", target = "pathology", data_kind = "image",
    model = contract, model_params = job$model_params, strategy = "fedavg", rounds = 5L,
    target_levels = c("benign", "malignant"), torch_backend = "cuda",
    output_dir = file.path(work, "artifact"), silent = TRUE)
  writeLines(fit$stdout, file.path(work, "flower-stdout.log"))
  writeLines(fit$stderr, file.path(work, "flower-stderr.log"))
  if (!isTRUE(fit$available)) stop("The nodes completed without publishing a private model.")
  stopifnot(file.exists(file.path(capture, "public-initial-arrays.npz")))
  metadata <- jsonlite::fromJSON(file.path(fit$output_dir, "metadata.json"))
  history <- jsonlite::fromJSON(file.path(fit$output_dir, "history.json"))
  if (sum(history$n_failures) != 0L || nrow(history) != 5L) stop("Incomplete or failed node rounds.")
  list(status = "trained_unscored", output_dir = fit$output_dir,
       model_sha256 = digest::digest(file = file.path(fit$output_dir, "model.pt"), algo = "sha256"),
       metadata = metadata, history = history)
}, error = function(e) {
  logs <- try(parallel::clusterCall(cluster, function() {
    paths <- list.files(file.path(tempdir(), "dsflower", "supernodes"), pattern = "\\.log$", recursive = TRUE, full.names = TRUE)
    lapply(paths, function(path) tail(readLines(path, warn = FALSE), 250L))
  }), silent = TRUE)
  if (!inherits(logs, "try-error")) jsonlite::write_json(logs, file.path(work, "public-node-diagnostics.json"), pretty = TRUE)
  list(status = "failed", error = conditionMessage(e))
})
cleanup()
result$cleanup_ok <- cleanup_ok
result$elapsed_s <- as.numeric(difftime(Sys.time(), started, units = "secs"))
result$epsilon <- epsilon
result$split_sha256 <- job$split_sha256
result$job_sha256 <- digest::digest(file = job_path, algo = "sha256")
result$n_patients_per_site <- job$n_patients_per_site
result$test_accessed <- FALSE
result$seed <- seed
result$contract <- contract
jsonlite::write_json(result, file.path(work, "federation-status.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(result$status, "epsilon", epsilon, "seed", seed, "elapsed", result$elapsed_s, "\n")
if (result$status == "failed" || !cleanup_ok) quit(status = 1L)
