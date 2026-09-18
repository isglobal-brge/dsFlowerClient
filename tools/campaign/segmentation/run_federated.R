#!/usr/bin/env Rscript
# Public-only, three actual DSLite workers and Flower SuperNodes.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6L) stop("Usage: run_federated.R PREPARED_DIR SPLIT_JSON EPSILON SEED WORK_DIR TOOLS_DIR")
prepared <- normalizePath(args[[1L]])
split_path <- normalizePath(args[[2L]])
epsilon <- as.numeric(args[[3L]])
seed <- as.integer(args[[4L]])
work_dir <- normalizePath(args[[5L]], mustWork = FALSE)
tools_dir <- normalizePath(args[[6L]])
stopifnot(epsilon %in% c(1, 4, 8), seed %in% c(20260919L, 20260920L, 20260921L))
source(file.path(tools_dir, "..", "campaign_lib.R"))
# main's transactional initialization queries/removes symbols on the real worker.
methods::setMethod("dsListSymbols", "CampaignDSLiteConnection", function(conn) {
  .campaign_remote_call(conn, function() DSI::dsListSymbols(.campaign_dslite_conn))
})
methods::setMethod("dsRmSymbol", "CampaignDSLiteConnection", function(conn, symbol) {
  .campaign_remote_call(conn, function(symbol) DSI::dsRmSymbol(.campaign_dslite_conn, symbol), symbol)
})
started_at <- Sys.time()
synthetic <- identical(Sys.getenv("F_SEG_SYNTHETIC"), "1")
nominal_batch <- as.integer(Sys.getenv("F_SEG_BATCH_SIZE", "16"))
stopifnot(nominal_batch %in% c(16L, 64L))
gate_path <- Sys.getenv("F_SEG_GATES_JSON")
if (!synthetic) {
  if (!file.exists(gate_path)) stop("F_SEG_GATES_JSON must identify blocking-gate evidence.")
  gates <- jsonlite::fromJSON(gate_path)
  if (!all(vapply(paste0("segmentation_6_1_", 1:7), function(name) isTRUE(gates[[name]]), logical(1)))) {
    stop("All seven blocking mechanism/API gates must pass before public scored cells.")
  }
}
if (startsWith(work_dir, "/workspace/")) {
  markers <- c("/workspace/logs/install_r.log", "/workspace/segmentation/r-stack-ready.log")
  ready <- any(vapply(markers, function(p) file.exists(p) &&
    any(readLines(p, warn = FALSE) == "R_STACK_DONE"), logical(1)))
  if (!ready) stop("Pod R stack completion gate missing.")
}
for (name in c("DSFLOWER_VENV_ROOT", "DSFLOWER_CLIENT_VENV_ROOT", "TORCH_HOME", "F_SEG_RUNNER_PARENT")) {
  if (!nzchar(Sys.getenv(name))) stop("Missing environment: ", name)
}
# The trusted node environment inherits XDG_CACHE_HOME, not TORCH_HOME.
stopifnot(identical(basename(Sys.getenv("TORCH_HOME")), "torch"))
Sys.setenv(XDG_CACHE_HOME = dirname(Sys.getenv("TORCH_HOME")))
dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
capture_dir <- file.path(work_dir, "public-capture")
if (dir.exists(capture_dir) && length(list.files(capture_dir, all.files = TRUE, no.. = TRUE))) {
  stop("A new cell requires an empty capture directory; retain previous attempts separately.")
}
Sys.setenv(F_SEG_PUBLIC_BENCHMARK = "1", F_SEG_INIT_SEED = seed,
           F_SEG_CAPTURE_DIR = capture_dir,
           CUBLAS_WORKSPACE_CONFIG = ":4096:8",
           OMP_NUM_THREADS = "2", MKL_NUM_THREADS = "2",
           PYTHONPATH = paste(file.path(tools_dir, "benchmark_hooks"),
                              Sys.getenv("F_SEG_RUNNER_PARENT"), sep = .Platform$path.sep))
audit <- jsonlite::fromJSON(file.path(prepared, "audit.json"), simplifyVector = FALSE)
if (synthetic && !identical(audit$dataset, "synthetic")) stop("Synthetic mode requires the synthetic fixture.")
if (!synthetic && !audit$dataset %in% c("breast", "busbra")) stop("Unknown public cohort.")
source_split <- file.path(work_dir, "source-split.json")
stopifnot(file.copy(split_path, source_split, overwrite = FALSE))
source_split_sha256 <- digest::digest(file = source_split, algo = "sha256")
if (!synthetic) {
  frozen_audit <- jsonlite::fromJSON(file.path(tools_dir, "..", "..", "..", "inst", "extdata",
      "campaign", "segmentation", "provenance", paste0(audit$dataset, "-audit.json")))
  if (!identical(source_split_sha256, frozen_audit$split_hashes[[as.character(seed)]])) {
    stop("Source split differs from the archived preregistration.")
  }
}
split <- jsonlite::fromJSON(source_split, simplifyVector = FALSE)
stopifnot(identical(as.integer(split$seed), seed))
variant <- Sys.getenv("F_SEG_VARIANT", "full")
if (!variant %in% c("full", "small192", "heterogeneous", "bce")) stop("Unknown preregistered variant.")
if (!identical(variant, "full") && !identical(audit$dataset, "busbra")) stop("Extensions are preregistered for BUS-BRA only.")
if (variant %in% c("heterogeneous", "bce") && epsilon != 8) stop("This extension is preregistered at epsilon8 only.")
if (identical(variant, "small192")) {
  split$train <- split$small_train
  split$sites <- split$small_sites
}
if (identical(variant, "heterogeneous")) split$sites <- split$heterogeneous_sites
split$variant <- variant
jsonlite::write_json(split, file.path(work_dir, "effective-split.json"), auto_unbox = TRUE, pretty = TRUE)
frame <- utils::read.csv(file.path(prepared, "samples.csv"), stringsAsFactors = FALSE,
                         colClasses = "character")
site_data <- lapply(split$sites, function(ids) frame[frame$subject_id %in% unlist(ids), , drop = FALSE])
stopifnot(length(site_data) == 3L,
          sum(vapply(site_data, function(x) length(unique(x$subject_id)), integer(1))) == length(split$train))
venv <- Sys.getenv("DSFLOWER_VENV_ROOT")
ports <- .campaign_free_ports(6L)
cluster <- parallel::makePSOCKcluster(3L, outfile = file.path(work_dir, "workers.log"))
conns <- list()
cleanup_ok <- FALSE
cleanup <- function() {
  if (length(conns)) try(ds.flower.link.down(conns), silent = TRUE)
  if (!is.null(cluster)) {
    drained <- try(parallel::clusterCall(cluster, function() {
      cid <- dsFlower:::.dsflower_env$tunnel_conn_id
      if (!is.null(cid)) try(dsFlower::flowerTunnelDownDS(cid), silent = TRUE)
      nodes <- dsFlower:::.supernode_list()
      for (manifest in nodes$manifest_dir) try(dsFlower:::.supernode_stop(manifest), silent = TRUE)
      if (exists(".campaign_dslite_conn", .GlobalEnv, inherits = FALSE)) {
        try(DSI::dsDisconnect(.campaign_dslite_conn), silent = TRUE)
      }
      is.null(dsFlower:::.dsflower_env$tunnel_conn_id) && nrow(dsFlower:::.supernode_list()) == 0L
    }), silent = TRUE)
    cleanup_ok <<- !inherits(drained, "try-error") && all(vapply(drained, isTRUE, logical(1)))
    try(parallel::stopCluster(cluster), silent = TRUE)
    cluster <<- NULL
  }
  try(ds.flower.superlink.stop(), silent = TRUE)
  cleanup_ok <<- isTRUE(cleanup_ok) && tryCatch(
    !isTRUE(ds.flower.superlink.status()$running), error = function(e) FALSE)
}
result <- tryCatch({
  parallel::clusterMap(cluster, function(index, data, libpaths, venv, work_dir, epsilon, audit, capture_dir, seed) {
    .libPaths(libpaths)
    # RunPod's /workspace FUSE volume ignores chmod; OS temp storage enforces it.
    # These are isolated public-fixture node identities, stable across all rounds.
    secret_dir <- file.path(tempdir(), "segmentation-node-state")
    dir.create(secret_dir, mode = "0700", showWarnings = FALSE)
    observer_config <- file.path(secret_dir, "segmentation-public-benchmark.json")
    jsonlite::write_json(list(public_fixture_only = TRUE, dataset = audit$dataset,
                              capture_dir = capture_dir, seed = seed),
                         observer_config, auto_unbox = TRUE)
    Sys.chmod(observer_config, "0600")
    Sys.setenv(DSFLOWER_VENV_ROOT = venv,
               DSFLOWER_NODE_SECRET_FILE = file.path(secret_dir, paste0("secret-site", index)),
               DSFLOWER_TEST_ALLOW_EPHEMERAL_SECRET = "1")
    options(dsflower.venv_root = venv,
            dsflower.dp_unit = "patient", dsflower.patient_column = "subject_id",
            dsflower.dp_per_training_epsilon = epsilon, dsflower.dp_per_training_delta = 1e-5,
            dsflower.dp_clipping_norm = 1,
            dsflower.image_data_root = audit$image_root, dsflower.mask_data_root = audit$mask_root)
    suppressPackageStartupMessages({library(DSI); library(DSLite); library(dsFlower)})
    server <- DSLite::newDSLiteServer(tables = list(training = data),
      config = DSLite::defaultDSConfiguration(include = c("dsBase", "dsFlower")),
      home = file.path(work_dir, paste0("dslite", index)))
    symbol <- paste0("segmentation_site", index, "_", Sys.getpid())
    assign(symbol, server, .GlobalEnv)
    .campaign_dslite_conn <<- DSLite::dsConnect(DSLite::DSLite(), name = paste0("site", index), url = symbol)
    DSLite::dsAssignTable(.campaign_dslite_conn, "D", "training")
    TRUE
  }, index = seq_len(3L), data = site_data, MoreArgs = list(libpaths = .libPaths(),
      venv = venv, work_dir = work_dir, epsilon = epsilon, audit = audit,
      capture_dir = capture_dir, seed = seed), SIMPLIFY = FALSE)
  dummy <- DSLite::newDSLiteServer(tables = list())
  conns <- stats::setNames(lapply(seq_len(3L), function(i) methods::new("CampaignDSLiteConnection",
    name = paste0("site", i), sid = paste0("remote-site", i), server = dummy, worker = cluster[i])), paste0("site", 1:3))
  options(dsflower.tunnel_port = ports[4:6], dsflower.superlink_insecure = TRUE,
          dsflower.tunnel_loss_tolerance = 30, dsflower.supernode_term_grace = 10)
  ds.flower.superlink.start(fleet_port = ports[1], control_port = ports[2], serverappio_port = ports[3], insecure = TRUE)
  fit <- ds.flower.fit(conns, symbol = "D", target = "mask_path", task = "segmentation",
      model = "pytorch_resnet18_segmentation", data_kind = "image", strategy = "fedavg",
      model_params = list(alpha = if (identical(variant, "bce")) 1 else .5, mask_values = "0,255", sample_id_col = "image_id",
          image_path_col = "relative_path", mask_empty_col = "mask_empty", subject_id_col = "subject_id",
          learning_rate = .01, batch_size = if (synthetic) 8L else nominal_batch,
          local_epochs = if (synthetic) 1L else 2L),
      rounds = if (synthetic) 2L else 5L, torch_backend = "cuda", output_dir = file.path(work_dir, "artifact"), silent = TRUE)
  writeLines(fit$stdout, file.path(work_dir, "flower-stdout.log"))
  writeLines(fit$stderr, file.path(work_dir, "flower-stderr.log"))
  if (!isTRUE(fit$available)) stop("No available released segmentation model.")
  metadata <- jsonlite::fromJSON(file.path(fit$output_dir, "metadata.json"))
  history <- jsonlite::fromJSON(file.path(fit$output_dir, "history.json"))
  stopifnot(identical(metadata$status, "success"), as.integer(metadata$n_clients) == 3L,
            nrow(history) == if (synthetic) 2L else 5L,
            all(history$n_failures == 0L), inherits(fit, "dsflower_run"), fit$status == 0L)
  stopifnot(file.exists(file.path(work_dir, "public-capture", "public-initial-arrays.npz")))
  # Canonical subject/image ordering matches the independently cached public masks.
  test <- frame[frame$subject_id %in% unlist(split$test), , drop = FALSE]
  test <- test[order(test$subject_id, test$image_id), , drop = FALSE]
  test <- test[!duplicated(test$subject_id), , drop = FALSE]
  paths <- file.path(audit$image_root, test$relative_path)
  # Seven internal batches16 fit beneath the predictor's 2M-cell request bound.
  blocks <- base::split(seq_along(paths), ceiling(seq_along(paths) / 112L))
  probability <- do.call(rbind, lapply(blocks, function(ix) {
    values <- ds.flower.predict(fit, paths[ix], type = "prob")
    do.call(rbind, lapply(seq_along(ix), function(i) as.vector(t(values[i, 1, , ]))))
  }))
  stopifnot(identical(dim(probability), c(nrow(test), 16384L)),
            all(is.finite(probability)), all(probability >= 0 & probability <= 1))
  if (synthetic) {
    repeated <- ds.flower.predict(fit$output_dir, paths, type = "prob")
    repeated <- do.call(rbind, lapply(seq_along(paths), function(i) as.vector(t(repeated[i, 1, , ]))))
    stopifnot(identical(probability, repeated))
  }
  utils::write.table(probability, file.path(work_dir, "public-probabilities.csv"),
                     row.names = FALSE, col.names = FALSE, sep = ",")
  list(status = "predicted_pending_public_metric_summary", output_dir = fit$output_dir,
       model_sha256 = digest::digest(file = file.path(fit$output_dir, "model.pt"), algo = "sha256"))
}, error = function(e) {
  # Preserve public-fixture node diagnostics before worker temp directories vanish.
  logs <- try(parallel::clusterCall(cluster, function() {
    paths <- list.files(file.path(tempdir(), "dsflower", "supernodes"),
                        pattern = "\\.log$", recursive = TRUE, full.names = TRUE)
    stats::setNames(lapply(paths, function(path) tail(readLines(path, warn = FALSE), 200L)),
                    basename(paths))
  }), silent = TRUE)
  if (!inherits(logs, "try-error")) {
    jsonlite::write_json(logs, file.path(work_dir, "public-node-diagnostics.json"), pretty = TRUE)
  }
  list(status = "failed", error = conditionMessage(e))
})
cleanup()
result$cleanup_ok <- cleanup_ok
result$dataset <- audit$dataset
result$synthetic <- synthetic
result$elapsed_s <- as.numeric(difftime(Sys.time(), started_at, units = "secs"))
result$variant <- variant
result$seed <- seed
result$epsilon <- epsilon
result$split_sha256 <- source_split_sha256
jsonlite::write_json(result, file.path(work_dir, "federation-status.json"), pretty = TRUE, auto_unbox = TRUE)
if (identical(result$status, "failed") || !cleanup_ok) stop("Federation failed; see federation-status.json.")
