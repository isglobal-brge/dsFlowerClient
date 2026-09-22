#!/usr/bin/env Rscript
# Three DSLite sites; training only. The test archive is never opened here.
args <- commandArgs(TRUE)
stopifnot(length(args) == 4L)
root <- normalizePath(args[[1]])
epsilon <- as.numeric(args[[2]])
seed <- as.integer(args[[3]])
contract <- args[[4]]
tools <- file.path(root, "src/dsFlowerClient/tools/campaign/sequence")
source(file.path(tools, "../campaign_lib.R"))
protocol <- jsonlite::fromJSON(file.path(tools, "protocol.json"))
stopifnot(epsilon %in% protocol$epsilon_order, seed %in% protocol$seeds,
          contract %in% c("pytorch_lstm", "pytorch_gru"))
work <- file.path(root, "runs", sprintf("%s-eps%g-seed%d", contract, epsilon, seed))
if (dir.exists(work)) stop("Existing run directory; retain all attempts and refuse overwrite.")
dir.create(work, recursive = TRUE)
capture <- file.path(work, "public-capture")
dir.create(capture)
Sys.setenv(F_SEQUENCE_PUBLIC_BENCHMARK = "1", F_SEQUENCE_INIT_SEED = seed,
           F_SEQUENCE_CAPTURE_DIR = capture, CUBLAS_WORKSPACE_CONFIG = ":4096:8",
           OMP_NUM_THREADS = "2", MKL_NUM_THREADS = "2", OPENBLAS_NUM_THREADS = "2",
           PYTHONPATH = paste(file.path(tools, "benchmark_hooks"),
             system.file("flower_app", package = "dsFlowerClient"), sep = ":"))
audit <- jsonlite::fromJSON(file.path(root, "prepared/audit.json"))
train <- utils::read.csv(file.path(root, "prepared/train.csv"), check.names = FALSE)
sites <- lapply(seq_len(3), function(i) train[train$subject %in% audit$site_subjects[i, ], ])
stopifnot(all(vapply(sites, function(x) length(unique(x$subject)), integer(1)) == 7L))
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
  parallel::clusterMap(cluster, function(index, data, libpaths, venv, work, epsilon, capture, seed) {
    .libPaths(libpaths)
    secret_dir <- file.path(tempdir(), "sequence-node-state")
    dir.create(secret_dir, mode = "0700")
    observer <- file.path(secret_dir, "sequence-public-benchmark.json")
    jsonlite::write_json(list(public_fixture_only = TRUE, dataset = "uci_har", capture_dir = capture, seed = seed),
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
      dsflower.patient_column = "subject", dsflower.dp_per_training_epsilon = epsilon,
      dsflower.dp_per_training_delta = 1e-6, dsflower.dp_clipping_norm = 1)
    suppressPackageStartupMessages({library(DSI); library(DSLite); library(dsFlower)})
    server <- DSLite::newDSLiteServer(tables = list(training = data),
      config = DSLite::defaultDSConfiguration(include = c("dsBase", "dsFlower")),
      home = file.path(work, paste0("dslite", index)))
    symbol <- paste0("sequence_site", index, "_", Sys.getpid())
    assign(symbol, server, .GlobalEnv)
    .campaign_dslite_conn <<- DSLite::dsConnect(DSLite::DSLite(), name = paste0("site", index), url = symbol)
    DSLite::dsAssignTable(.campaign_dslite_conn, "D", "training")
    TRUE
  }, index = seq_len(3L), data = sites, MoreArgs = list(libpaths = .libPaths(), venv = venv,
       work = work, epsilon = epsilon, capture = capture, seed = seed), SIMPLIFY = FALSE)
  dummy <- DSLite::newDSLiteServer(tables = list())
  conns <- setNames(lapply(seq_len(3), function(i) methods::new("CampaignDSLiteConnection",
    name = paste0("site", i), sid = paste0("remote-site", i), server = dummy, worker = cluster[i])), paste0("site", 1:3))
  policies <- lapply(conns, function(conn) DSI::dsFetch(DSI::dsAggregate(conn, quote(flowerPrivacyPolicyDS()), async = FALSE)))
  capabilities <- lapply(conns, function(conn) DSI::dsFetch(DSI::dsAggregate(conn, quote(flowerGetCapabilitiesDS()), async = FALSE)))
  jsonlite::write_json(list(policies = policies, capabilities = capabilities), file.path(work, "node-contract.json"),
                       pretty = TRUE, auto_unbox = TRUE, digits = NA)
  options(dsflower.tunnel_port = ports[4:6], dsflower.superlink_insecure = TRUE,
          dsflower.tunnel_loss_tolerance = 30, dsflower.supernode_term_grace = 10)
  ds.flower.superlink.start(fleet_port = ports[1], control_port = ports[2], serverappio_port = ports[3], insecure = TRUE)
  fit <- ds.flower.fit(conns, symbol = "D", target = "target", features = audit$features,
    model = contract, model_params = protocol$model_params, strategy = "fedavg", rounds = 5L,
    target_levels = as.character(0:5), feature_bounds = audit$bounds, torch_backend = "cuda",
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
result$seed <- seed
result$contract <- contract
jsonlite::write_json(result, file.path(work, "federation-status.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(result$status, "epsilon", epsilon, "seed", seed, "elapsed", result$elapsed_s, "\n")
if (result$status == "failed" || !cleanup_ok) quit(status = 1L)
