#!/usr/bin/env Rscript

# Real local integration smoke: two isolated DSLite data nodes, two Flower
# SuperNodes, one SuperLink, and one round of DP-SGD logistic regression.

suppressPackageStartupMessages({
  library(DSI)
  library(DSLite)
  library(dsFlower)
  library(dsFlowerClient)
  library(jsonlite)
})

started_at <- Sys.time()
setTimeLimit(elapsed = 260, transient = FALSE)

required_env <- c("DSFLOWER_VENV_ROOT", "DSFLOWER_NODE_SECRET_FILE")
missing_env <- required_env[!nzchar(Sys.getenv(required_env, unset = ""))]
if (length(missing_env)) {
  stop("Missing required environment variable(s): ",
       paste(missing_env, collapse = ", "), call. = FALSE)
}

work_dir <- Sys.getenv("DSFLOWER_SMOKE_WORKDIR", unset = "")
if (!nzchar(work_dir)) work_dir <- tempfile("dsflower-multinode-smoke-")
dir.create(work_dir, recursive = TRUE, showWarnings = FALSE, mode = "0700")
if (!dir.exists(work_dir)) stop("Could not create smoke work directory.", call. = FALSE)

output_dir <- file.path(work_dir, "artifact")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

free_port <- function(exclude = integer()) {
  for (i in seq_len(100L)) {
    port <- dsFlower:::.random_available_port()
    if (!port %in% exclude) return(port)
  }
  stop("Could not reserve a distinct dynamic port.", call. = FALSE)
}

ports <- integer()
for (i in seq_len(5L)) ports <- c(ports, free_port(ports))
names(ports) <- c("fleet", "control", "serverappio", "site1", "site2")

cat("dsFlower multinode smoke\n")
cat("  R:", R.version.string, "\n")
cat("  dsFlower:", as.character(packageVersion("dsFlower")), "\n")
cat("  dsFlowerClient:", as.character(packageVersion("dsFlowerClient")), "\n")
python <- file.path(Sys.getenv("DSFLOWER_VENV_ROOT"), "pytorch", "bin", "python")
flower_version <- processx::run(
  python, c("-c", "import flwr; print(flwr.__version__)"),
  error_on_status = TRUE)$stdout
cat("  Flower:", trimws(flower_version), "\n")
cat("  ports:", paste(ports, collapse = ","), "\n")

set.seed(20260811)
make_site <- function(offset) {
  n <- 48L
  x1 <- seq(-2.5, 2.5, length.out = n) + offset
  x2 <- sin(seq_len(n) / 4 + offset)
  data.frame(
    x1 = x1,
    x2 = x2,
    outcome = as.integer(x1 + 0.7 * x2 > median(x1 + 0.7 * x2))
  )
}

# A DSLiteServer is normally in-process. Two of them would therefore share the
# dsFlower package namespace and its singleton tunnel state. Keep the actual
# DSLite servers in separate local R workers, as two real data-node processes
# are, and proxy only the standard synchronous DSI calls used by this smoke.
methods::setClass(
  "SmokeDSLiteConnection",
  contains = "DSLiteConnection",
  slots = c(worker = "ANY")
)
methods::setClass(
  "SmokeDSLiteResult",
  contains = "DSResult",
  slots = c(conn = "SmokeDSLiteConnection", rval = "list")
)

remote_call <- function(conn, fun, ...) {
  tryCatch(
    parallel::clusterCall(conn@worker, fun, ...)[[1L]],
    error = function(e) {
      message(conn@name, " worker error: ", conditionMessage(e))
      stop(e)
    }
  )
}
methods::setMethod("dsIsAsync", "SmokeDSLiteConnection", function(conn) {
  list(aggregate = FALSE, assignTable = FALSE, assignExpr = FALSE,
       assignResource = FALSE)
})
methods::setMethod("dsKeepAlive", "SmokeDSLiteConnection", function(conn) NULL)
methods::setMethod(
  "dsAggregate", "SmokeDSLiteConnection",
  function(conn, expr, async = TRUE) {
    value <- remote_call(conn, function(expr) {
      DSI::dsFetch(DSI::dsAggregate(.smoke_dslite_conn, expr, async = FALSE))
    }, expr)
    methods::new(
      "SmokeDSLiteResult", conn = conn,
      rval = list(status = "COMPLETED", result = value))
  }
)
methods::setMethod(
  "dsAssignExpr", "SmokeDSLiteConnection",
  function(conn, symbol, expr, async = TRUE) {
    remote_call(conn, function(symbol, expr) {
      DSI::dsFetch(DSI::dsAssignExpr(
        .smoke_dslite_conn, symbol, expr, async = FALSE))
      NULL
    }, symbol, expr)
    methods::new(
      "SmokeDSLiteResult", conn = conn,
      rval = list(status = "COMPLETED", result = NULL))
  }
)
methods::setMethod(
  "dsDisconnect", "SmokeDSLiteConnection",
  function(conn, save = NULL) {
    remote_call(conn, function(save) {
      DSI::dsDisconnect(.smoke_dslite_conn, save = save)
      TRUE
    }, save)
    invisible(TRUE)
  }
)
methods::setMethod("dsFetch", "SmokeDSLiteResult", function(res) res@rval$result)
methods::setMethod(
  "dsGetInfo", "SmokeDSLiteResult", function(dsObj, ...) {
    list(status = dsObj@rval$status)
  }
)

cluster <- parallel::makePSOCKcluster(2L, outfile = "")
cluster_for_early_exit <- cluster
on.exit(try(parallel::stopCluster(cluster_for_early_exit), silent = TRUE),
        add = TRUE)
worker_libpaths <- .libPaths()
site_data <- list(make_site(-0.25), make_site(0.25))
parallel::clusterMap(
  cluster,
  function(index, data, work_dir, libpaths, venv_root) {
    .libPaths(libpaths)
    Sys.setenv(
      DSFLOWER_VENV_ROOT = venv_root,
      DSFLOWER_NODE_SECRET_FILE = file.path(
        work_dir, paste0("node-secret-site", index)),
      DSFLOWER_TEST_ALLOW_EPHEMERAL_SECRET = "1"
    )
    options(dsflower.venv_root = venv_root)
    suppressPackageStartupMessages({
      library(DSI)
      library(DSLite)
      library(dsFlower)
    })
    config <- DSLite::defaultDSConfiguration(
      include = c("dsBase", "dsFlower"))
    server <- DSLite::newDSLiteServer(
      tables = list(training = data), config = config,
      home = file.path(work_dir, paste0("dslite-site", index)))
    symbol <- paste0("dsflower_smoke_site", index, "_", Sys.getpid())
    assign(symbol, server, envir = .GlobalEnv)
    .smoke_dslite_conn <<- DSLite::dsConnect(
      DSLite::DSLite(), name = paste0("site", index), url = symbol)
    DSLite::dsAssignTable(.smoke_dslite_conn, "D", "training")
    TRUE
  },
  index = seq_len(2L), data = site_data,
  MoreArgs = list(
    work_dir = work_dir,
    libpaths = worker_libpaths,
    venv_root = Sys.getenv("DSFLOWER_VENV_ROOT")),
  SIMPLIFY = FALSE
)

dummy_server <- DSLite::newDSLiteServer(tables = list())
conns <- stats::setNames(lapply(seq_len(2L), function(i) {
  methods::new(
    "SmokeDSLiteConnection",
    name = paste0("site", i), sid = paste0("remote-site", i),
    server = dummy_server, worker = cluster[i])
}), c("site1", "site2"))

cleaned <- FALSE
cleanup_ok <- FALSE
cleanup <- function() {
  if (cleaned) return(invisible(NULL))
  cleaned <<- TRUE
  if (length(conns) &&
      (!is.null(dsFlowerClient:::.dsflower_client_env$.tunnel) ||
       !is.null(dsFlowerClient:::.dsflower_client_env$.superlink))) {
    try(ds.flower.link.down(conns), silent = TRUE)
  }
  if (!is.null(cluster)) {
    drained <- try(parallel::clusterCall(cluster, function() {
      cid <- dsFlower:::.dsflower_env$tunnel_conn_id
      if (!is.null(cid)) try(dsFlower::flowerTunnelDownDS(cid), silent = TRUE)
      nodes <- dsFlower:::.supernode_list()
      if (nrow(nodes)) {
        for (manifest in nodes$manifest_dir) {
          try(dsFlower:::.supernode_stop(manifest), silent = TRUE)
        }
      }
      if (exists(".smoke_dslite_conn", envir = .GlobalEnv, inherits = FALSE)) {
        try(DSI::dsDisconnect(.smoke_dslite_conn), silent = TRUE)
      }
      is.null(dsFlower:::.dsflower_env$tunnel_conn_id) &&
        nrow(dsFlower:::.supernode_list()) == 0L
    }), silent = TRUE)
    cleanup_ok <<- !inherits(drained, "try-error") &&
      length(drained) == 2L && all(vapply(drained, isTRUE, logical(1)))
    try(parallel::stopCluster(cluster), silent = TRUE)
    cluster <<- NULL
  }
  if (!is.null(dsFlowerClient:::.dsflower_client_env$.superlink)) {
    try(ds.flower.superlink.stop(), silent = TRUE)
  }
  invisible(NULL)
}
on.exit(cleanup(), add = TRUE)

options(
  datashield.errors.print = TRUE,
  dsflower.tunnel_port = unname(ports[c("site1", "site2")]),
  dsflower.superlink_insecure = TRUE,
  dsflower.tunnel_loss_tolerance = 30,
  dsflower.supernode_term_grace = 10
)

ds.flower.superlink.start(
  fleet_port = ports[["fleet"]],
  control_port = ports[["control"]],
  serverappio_port = ports[["serverappio"]],
  insecure = TRUE
)

fit <- ds.flower.fit(
  conns,
  symbol = "D",
  target = "outcome",
  features = c("x1", "x2"),
  model = "pytorch_logreg",
  model_params = list(learning_rate = 0.05, batch_size = 16L,
                      local_epochs = 1L),
  rounds = 1L,
  feature_bounds = list(lower = c(-4, -1), upper = c(4, 1)),
  target_levels = c("0", "1"),
  torch_backend = "cpu",
  output_dir = output_dir,
  silent = TRUE,
  verbose = FALSE
)

if (!isTRUE(fit$available)) {
  node_logs <- parallel::clusterCall(cluster, function() {
    paths <- list.files(
      file.path(tempdir(), "dsflower", "supernodes"),
      pattern = "\\.log$", full.names = TRUE)
    stats::setNames(lapply(paths, function(path) {
      utils::tail(readLines(path, warn = FALSE), 120L)
    }), basename(paths))
  })
  for (i in seq_along(node_logs)) {
    cat("\n--- site", i, " SuperNode log ---\n", file = stderr())
    cat(unlist(node_logs[[i]], use.names = FALSE), sep = "\n", file = stderr())
  }
  stop("The nodes completed without publishing a private model.", call. = FALSE)
}

persisted_dir <- fit$output_dir
metadata_path <- file.path(persisted_dir, "metadata.json")
history_path <- file.path(persisted_dir, "history.json")
model_path <- file.path(persisted_dir, "model.pt")
for (path in c(metadata_path, history_path, model_path)) {
  if (!file.exists(path) || file.info(path)$size <= 0) {
    stop("Missing or empty run artifact: ", path, call. = FALSE)
  }
}

metadata <- jsonlite::fromJSON(metadata_path, simplifyVector = TRUE)
history <- jsonlite::fromJSON(history_path, simplifyVector = TRUE)
if (!identical(metadata$status, "success") ||
    !identical(as.integer(metadata$n_clients), 2L)) {
  stop("The run did not record exactly two successful SuperNodes.", call. = FALSE)
}
if (nrow(history) != 1L || !identical(as.integer(history$round), 1L) ||
    !identical(as.integer(history$n_failures), 0L)) {
  stop("The one-round Flower history is incomplete or contains failures.",
       call. = FALSE)
}
if (!inherits(fit, "dsflower_run") || !identical(as.integer(fit$status), 0L)) {
  stop("ds.flower.fit() did not return a completed dsflower_run.", call. = FALSE)
}

cleanup()
Sys.sleep(1)
if (isTRUE(ds.flower.superlink.status()$running) ||
    nrow(dsFlower:::.supernode_list()) != 0L || !isTRUE(cleanup_ok)) {
  stop("Flower process cleanup did not drain the local federation.", call. = FALSE)
}

elapsed <- as.numeric(difftime(Sys.time(), started_at, units = "secs"))
cat("PASS: 2 SuperNodes, 1 round, 0 failures\n")
cat("  artifact:", normalizePath(persisted_dir), "\n")
cat("  model_sha256:", digest::digest(file = model_path, algo = "sha256"), "\n")
cat("  elapsed_seconds:", sprintf("%.1f", elapsed), "\n")
if (elapsed > 300) stop("Smoke exceeded the 300-second budget.", call. = FALSE)
