# Campaign library: N-site DSLite federation plumbing, dataset preparation,
# evaluation metrics, and evidence-artifact emission for the dsFlower utility
# campaign. The federation pattern is copied from
# tools/integration/dslite-multinode-smoke.R and generalised from 2 workers to
# N: one DSLite server per PSOCK R worker process, because dsFlower keeps
# singleton tunnel state per package namespace.

suppressPackageStartupMessages({
  library(DSI)
  library(DSLite)
  library(dsFlower)
  library(dsFlowerClient)
  library(jsonlite)
})

# ---------------------------------------------------------------------------
# DSLite-in-worker proxy classes (identical mechanics to the multinode smoke)
# ---------------------------------------------------------------------------

methods::setClass(
  "CampaignDSLiteConnection",
  contains = "DSLiteConnection",
  slots = c(worker = "ANY")
)
methods::setClass(
  "CampaignDSLiteResult",
  contains = "DSResult",
  slots = c(conn = "CampaignDSLiteConnection", rval = "list")
)

.campaign_remote_call <- function(conn, fun, ...) {
  tryCatch(
    parallel::clusterCall(conn@worker, fun, ...)[[1L]],
    error = function(e) {
      message(conn@name, " worker error: ", conditionMessage(e))
      stop(e)
    }
  )
}
methods::setMethod("dsIsAsync", "CampaignDSLiteConnection", function(conn) {
  list(aggregate = FALSE, assignTable = FALSE, assignExpr = FALSE,
       assignResource = FALSE)
})
methods::setMethod("dsKeepAlive", "CampaignDSLiteConnection", function(conn) NULL)
methods::setMethod(
  "dsAggregate", "CampaignDSLiteConnection",
  function(conn, expr, async = TRUE) {
    value <- .campaign_remote_call(conn, function(expr) {
      DSI::dsFetch(DSI::dsAggregate(.campaign_dslite_conn, expr, async = FALSE))
    }, expr)
    methods::new(
      "CampaignDSLiteResult", conn = conn,
      rval = list(status = "COMPLETED", result = value))
  }
)
methods::setMethod(
  "dsAssignExpr", "CampaignDSLiteConnection",
  function(conn, symbol, expr, async = TRUE) {
    .campaign_remote_call(conn, function(symbol, expr) {
      DSI::dsFetch(DSI::dsAssignExpr(
        .campaign_dslite_conn, symbol, expr, async = FALSE))
      NULL
    }, symbol, expr)
    methods::new(
      "CampaignDSLiteResult", conn = conn,
      rval = list(status = "COMPLETED", result = NULL))
  }
)
methods::setMethod(
  "dsDisconnect", "CampaignDSLiteConnection",
  function(conn, save = NULL) {
    .campaign_remote_call(conn, function(save) {
      DSI::dsDisconnect(.campaign_dslite_conn, save = save)
      TRUE
    }, save)
    invisible(TRUE)
  }
)
methods::setMethod("dsFetch", "CampaignDSLiteResult", function(res) res@rval$result)
methods::setMethod(
  "dsGetInfo", "CampaignDSLiteResult", function(dsObj, ...) {
    list(status = dsObj@rval$status)
  }
)

# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

CAMPAIGN_COHORT_SEED <- 20260819L  # cdc9k cohort subsampling seed (fixed)

# Load a prepared cohort: data.frame with numeric feature columns plus an
# integer `target` in {0, 1}. Returns list(df, meta).
campaign_load_cohort <- function(name, cache_dir) {
  if (identical(name, "breast")) {
    path <- file.path(cache_dir, "breast-cancer-wisconsin.data")
    cols <- c("id", "clump_thickness", "cell_size_uniformity",
              "cell_shape_uniformity", "marginal_adhesion",
              "epithelial_cell_size", "bare_nuclei", "bland_chromatin",
              "normal_nucleoli", "mitoses", "class")
    df <- utils::read.csv(path, header = FALSE, col.names = cols,
                          na.strings = "?")
    df <- df[stats::complete.cases(df), ]
    df$target <- as.integer(df$class == 4L)
    df <- df[, c(setdiff(cols, c("id", "class")), "target")]
    source_url <- paste0("https://archive.ics.uci.edu/ml/",
                         "machine-learning-databases/breast-cancer-wisconsin/",
                         "breast-cancer-wisconsin.data")
  } else if (identical(name, "heart")) {
    path <- file.path(cache_dir, "processed.cleveland.data")
    cols <- c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
              "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num")
    df <- utils::read.csv(path, header = FALSE, col.names = cols,
                          na.strings = "?")
    df <- df[stats::complete.cases(df), ]
    df$target <- as.integer(df$num > 0)
    df <- df[, c(setdiff(cols, "num"), "target")]
    source_url <- paste0("https://archive.ics.uci.edu/ml/",
                         "machine-learning-databases/heart-disease/",
                         "processed.cleveland.data")
  } else if (grepl("^cdc[0-9]+k$", name)) {
    path <- file.path(cache_dir, "cdc_diabetes_health_indicators.csv")
    full <- utils::read.csv(path)
    full$target <- as.integer(full$Diabetes_binary)
    full <- full[, c(setdiff(names(full), c("ID", "Diabetes_binary", "target")),
                     "target")]
    # Fixed-seed stratified subsample of N thousand rows (same cohort for every
    # replicate; per-replicate variation comes from the train/test split seed).
    # `cdc9k` is the pilot cohort; other sizes (cdc2k, cdc45k, cdc90k, ...)
    # support the n-scaling curve of the DP validation contract.
    set.seed(CAMPAIGN_COHORT_SEED)
    n_target <- as.numeric(sub("^cdc([0-9]+)k$", "\\1", name)) * 1000
    if (n_target > nrow(full)) stop("Requested CDC sample exceeds source rows.")
    idx <- unlist(lapply(split(seq_len(nrow(full)), full$target), function(ix) {
      take <- round(n_target * length(ix) / nrow(full))
      sample(ix, take)
    }), use.names = FALSE)
    idx <- sort(idx)
    if (length(idx) > n_target) idx <- idx[seq_len(n_target)]
    df <- full[idx, ]
    source_url <- "https://archive.ics.uci.edu/static/public/891/data.csv"
  } else {
    stop("Unknown dataset: ", name, call. = FALSE)
  }
  rownames(df) <- NULL
  features <- setdiff(names(df), "target")
  df[features] <- lapply(df[features], as.numeric)
  df$target <- as.integer(df$target)

  prepared_csv <- file.path(cache_dir, paste0("prepared_", name, ".csv"))
  utils::write.csv(df, prepared_csv, row.names = FALSE)
  list(
    df = df,
    meta = list(
      name = name,
      n_total = nrow(df),
      features = features,
      source_url = source_url,
      sha_of_prepared = digest::digest(file = prepared_csv, algo = "sha256")
    )
  )
}

# Stratified 80/20 train/test split, then the train side dealt evenly and
# stratified across n_sites shards. Deterministic given `seed`.
campaign_split <- function(df, seed, n_sites = 3L) {
  set.seed(seed)
  by_class <- split(seq_len(nrow(df)), df$target)
  test_idx <- unlist(lapply(by_class, function(ix) {
    sample(ix, max(1L, round(0.2 * length(ix))))
  }), use.names = FALSE)
  train_idx <- setdiff(seq_len(nrow(df)), test_idx)

  # Round-robin dealing per class keeps shards even in size and prevalence.
  shards <- vector("list", n_sites)
  for (ix in split(train_idx, df$target[train_idx])) {
    ix <- sample(ix)
    for (s in seq_len(n_sites)) {
      shards[[s]] <- c(shards[[s]], ix[seq_along(ix) %% n_sites == (s - 1L)])
    }
  }
  list(
    train = df[train_idx, , drop = FALSE],
    test = df[test_idx, , drop = FALSE],
    sites = lapply(shards, function(ix) df[sort(ix), , drop = FALSE])
  )
}

# Public-schema feature bound declaration: observed train min/max widened by
# 10% of the observed range (or +/-0.5 for degenerate constant columns).
campaign_bounds <- function(train, features) {
  lo <- vapply(train[features], min, numeric(1))
  hi <- vapply(train[features], max, numeric(1))
  pad <- 0.1 * (hi - lo)
  pad[pad <= 0] <- 0.5
  list(lower = unname(lo - pad), upper = unname(hi + pad))
}

# ---------------------------------------------------------------------------
# Metrics and baselines
# ---------------------------------------------------------------------------

campaign_metrics <- function(y, p) {
  y <- as.integer(y)
  p <- as.numeric(p)
  stopifnot(length(y) == length(p), all(y %in% c(0L, 1L)), all(is.finite(p)))
  n1 <- sum(y == 1L); n0 <- sum(y == 0L)
  auc <- if (n1 == 0L || n0 == 0L) NA_real_ else {
    r <- rank(p)
    (sum(r[y == 1L]) - n1 * (n1 + 1) / 2) / (n1 * n0)
  }
  pc <- pmin(pmax(p, 1e-15), 1 - 1e-15)
  list(
    auc = auc,
    acc = mean((p >= 0.5) == (y == 1L)),
    brier = mean((p - y)^2),
    logloss = -mean(y * log(pc) + (1 - y) * log(1 - pc))
  )
}

campaign_central <- function(train, test, features) {
  frame <- train[, c(features, "target")]
  fit <- suppressWarnings(stats::glm(target ~ ., family = stats::binomial(),
                                     data = frame))
  p <- suppressWarnings(
    stats::predict(fit, newdata = test[, features, drop = FALSE],
                   type = "response"))
  campaign_metrics(test$target, p)
}

# Central baseline for the torch MLP contract: the same model class fitted
# centrally without DP or federation, through the campaign venv's torch.
# Mirrors the logreg precedent (central = the model class fitted well on the
# pooled data), so the gap column isolates the DP-federation cost.
campaign_central_torch_mlp <- function(train, test, features, model_params) {
  client_root <- Sys.getenv("DSFLOWER_CLIENT_VENV_ROOT")
  stopifnot(nzchar(client_root))
  python <- file.path(client_root, "venv", "bin", "python")
  stopifnot(file.exists(python))
  tmp <- tempfile("central-mlp-")
  dir.create(tmp, mode = "0700")
  on.exit(unlink(tmp, recursive = TRUE, force = TRUE), add = TRUE)
  utils::write.csv(train[, c(features, "target")],
                   file.path(tmp, "train.csv"), row.names = FALSE)
  utils::write.csv(test[, features, drop = FALSE],
                   file.path(tmp, "test.csv"), row.names = FALSE)
  hidden <- model_params$hidden_layers %||% c(64L, 32L)
  script <- file.path(
    Sys.getenv("CAMPAIGN_TOOLS_DIR", file.path("tools", "campaign")),
    "central_train.py")
  stopifnot(file.exists(script))
  status <- system2(python, c(
    shQuote(script), "--train", shQuote(file.path(tmp, "train.csv")),
    "--test", shQuote(file.path(tmp, "test.csv")),
    "--out", shQuote(file.path(tmp, "probs.csv")),
    "--hidden", paste(hidden, collapse = ","), "--seed", "20260830"))
  stopifnot(identical(status, 0L))
  p <- utils::read.csv(file.path(tmp, "probs.csv"))$prob
  campaign_metrics(test$target, p)
}

`%||%` <- function(a, b) if (is.null(a)) b else a

campaign_trivial <- function(train, test) {
  prevalence <- mean(train$target)
  majority <- as.integer(prevalence >= 0.5)
  m <- campaign_metrics(test$target, rep(prevalence, nrow(test)))
  list(acc = mean(test$target == majority), prevalence = prevalence,
       brier = m$brier, auc = 0.5)
}

# ---------------------------------------------------------------------------
# Federated-DP run: build an N-site federation, fit, predict locally, tear down
# ---------------------------------------------------------------------------

.campaign_free_ports <- function(n) {
  ports <- integer()
  for (i in seq_len(200L)) {
    port <- dsFlower:::.random_available_port()
    if (!port %in% ports) ports <- c(ports, port)
    if (length(ports) == n) return(ports)
  }
  stop("Could not reserve distinct dynamic ports.", call. = FALSE)
}

# Runs one federated DP training over `site_data` (list of data.frames) and
# evaluates the released model on `test` locally (channel B). Everything is
# created and destroyed inside this call so sequential replicates in one R
# process stay isolated.
campaign_run_federated <- function(site_data, test, features, feature_bounds,
                                   epsilon, delta = 1e-6, rounds,
                                   model_params, work_dir, venv_root,
                                   contract = "pytorch_logreg",
                                   cv_folds = NULL) {
  n_sites <- length(site_data)
  dir.create(work_dir, recursive = TRUE, showWarnings = FALSE, mode = "0700")
  output_dir <- file.path(work_dir, "artifact")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  started <- Sys.time()

  ports <- .campaign_free_ports(3L + n_sites)
  names(ports) <- c("fleet", "control", "serverappio",
                    paste0("site", seq_len(n_sites)))

  cluster <- parallel::makePSOCKcluster(n_sites, outfile = "")
  conns <- list()
  cleanup_ok <- FALSE
  cleaned <- FALSE
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
        if (exists(".campaign_dslite_conn", envir = .GlobalEnv,
                   inherits = FALSE)) {
          try(DSI::dsDisconnect(.campaign_dslite_conn), silent = TRUE)
        }
        is.null(dsFlower:::.dsflower_env$tunnel_conn_id) &&
          nrow(dsFlower:::.supernode_list()) == 0L
      }), silent = TRUE)
      cleanup_ok <<- !inherits(drained, "try-error") &&
        length(drained) == n_sites &&
        all(vapply(drained, isTRUE, logical(1)))
      try(parallel::stopCluster(cluster), silent = TRUE)
      cluster <<- NULL
    }
    if (!is.null(dsFlowerClient:::.dsflower_client_env$.superlink)) {
      try(ds.flower.superlink.stop(), silent = TRUE)
    }
    invisible(NULL)
  }
  on.exit(cleanup(), add = TRUE)

  worker_libpaths <- .libPaths()
  parallel::clusterMap(
    cluster,
    function(index, data, work_dir, libpaths, venv_root, epsilon, delta) {
      .libPaths(libpaths)
      Sys.setenv(
        DSFLOWER_VENV_ROOT = venv_root,
        DSFLOWER_NODE_SECRET_FILE = file.path(
          work_dir, paste0("node-secret-site", index)),
        DSFLOWER_TEST_ALLOW_EPHEMERAL_SECRET = "1"
      )
      # Node-side administrator-pinned privacy contract (read by
      # dsFlower:::.privacy_policy() through .dsf_option()).
      options(
        dsflower.venv_root = venv_root,
        dsflower.dp_per_training_epsilon = epsilon,
        dsflower.dp_per_training_delta = delta
      )
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
      symbol <- paste0("dsflower_campaign_site", index, "_", Sys.getpid())
      assign(symbol, server, envir = .GlobalEnv)
      .campaign_dslite_conn <<- DSLite::dsConnect(
        DSLite::DSLite(), name = paste0("site", index), url = symbol)
      DSLite::dsAssignTable(.campaign_dslite_conn, "D", "training")
      TRUE
    },
    index = seq_len(n_sites), data = site_data,
    MoreArgs = list(
      work_dir = work_dir,
      libpaths = worker_libpaths,
      venv_root = venv_root,
      epsilon = epsilon,
      delta = delta),
    SIMPLIFY = FALSE
  )

  dummy_server <- DSLite::newDSLiteServer(tables = list())
  conns <- stats::setNames(lapply(seq_len(n_sites), function(i) {
    methods::new(
      "CampaignDSLiteConnection",
      name = paste0("site", i), sid = paste0("remote-site", i),
      server = dummy_server, worker = cluster[i])
  }), paste0("site", seq_len(n_sites)))

  options(
    datashield.errors.print = TRUE,
    dsflower.tunnel_port = unname(
      ports[paste0("site", seq_len(n_sites))]),
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

  if (!is.null(cv_folds)) {
    # Capability demonstration path: k clean-initialised federated trainings
    # under ONE node-owned per-job privacy contract (80% split across folds,
    # 20% for the single pooled differentially private OOF release).
    cv <- tryCatch(
      ds.flower.cross_validate(
        conns,
        symbol = "D",
        target = "target",
        features = features,
        model = contract,
        model_params = model_params,
        strategy = "fedavg",
        rounds = as.integer(rounds),
        folds = as.integer(cv_folds),
        feature_bounds = feature_bounds,
        target_levels = c("0", "1"),
        torch_backend = "cpu",
        output_dir = output_dir,
        silent = TRUE,
        verbose = isTRUE(nzchar(Sys.getenv("CAMPAIGN_CV_VERBOSE")))
      ),
      error = function(e) e)
    if (inherits(cv, "error")) {
      node_logs <- try(parallel::clusterCall(cluster, function() {
        paths <- list.files(
          file.path(tempdir(), "dsflower", "supernodes"),
          pattern = "\\.log$", full.names = TRUE)
        stats::setNames(lapply(paths, function(path) {
          utils::tail(readLines(path, warn = FALSE), 120L)
        }), basename(paths))
      }), silent = TRUE)
      if (!inherits(node_logs, "try-error")) {
        for (i in seq_along(node_logs)) {
          cat("\n--- site", i, "SuperNode log ---\n", file = stderr())
          cat(unlist(node_logs[[i]], use.names = FALSE), sep = "\n",
              file = stderr())
        }
      }
      stop(conditionMessage(cv), call. = FALSE)
    }
    stopifnot(inherits(cv, "dsflower_cv"))
    elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
    cleanup()
    return(list(
      cv = list(model = cv$model, folds = cv$folds, n_nodes = cv$n_nodes,
                task = cv$task, metrics = as.list(cv$metrics)),
      elapsed_s = elapsed, cleanup_ok = cleanup_ok))
  }

  fit <- ds.flower.fit(
    conns,
    symbol = "D",
    target = "target",
    features = features,
    model = contract,
    model_params = model_params,
    strategy = "fedavg",
    rounds = as.integer(rounds),
    feature_bounds = feature_bounds,
    target_levels = c("0", "1"),
    torch_backend = "cpu",
    output_dir = output_dir,
    silent = TRUE,
    verbose = FALSE
  )

  if (!isTRUE(fit$available)) {
    node_logs <- try(parallel::clusterCall(cluster, function() {
      paths <- list.files(
        file.path(tempdir(), "dsflower", "supernodes"),
        pattern = "\\.log$", full.names = TRUE)
      stats::setNames(lapply(paths, function(path) {
        utils::tail(readLines(path, warn = FALSE), 120L)
      }), basename(paths))
    }), silent = TRUE)
    if (!inherits(node_logs, "try-error")) {
      for (i in seq_along(node_logs)) {
        cat("\n--- site", i, "SuperNode log ---\n", file = stderr())
        cat(unlist(node_logs[[i]], use.names = FALSE), sep = "\n",
            file = stderr())
      }
    }
    stop("The nodes completed without publishing a private model.",
         call. = FALSE)
  }

  persisted_dir <- fit$output_dir
  model_path <- file.path(persisted_dir, "model.pt")
  for (path in file.path(persisted_dir,
                         c("metadata.json", "history.json", "model.pt"))) {
    if (!file.exists(path) || file.info(path)$size <= 0) {
      stop("Missing or empty run artifact: ", path, call. = FALSE)
    }
  }
  metadata <- jsonlite::fromJSON(file.path(persisted_dir, "metadata.json"))
  history <- jsonlite::fromJSON(file.path(persisted_dir, "history.json"))

  # Channel B: consume the released artifact locally on the held-out test set.
  prob <- ds.flower.predict(fit, test[, features, drop = FALSE], type = "prob")
  p1 <- if (is.matrix(prob) || is.data.frame(prob)) {
    prob <- as.matrix(prob)
    if (!is.null(colnames(prob)) && "1" %in% colnames(prob)) {
      as.numeric(prob[, "1"])
    } else {
      as.numeric(prob[, ncol(prob)])
    }
  } else {
    as.numeric(prob)
  }
  metrics <- campaign_metrics(test$target, p1)

  cleanup()
  Sys.sleep(1)
  if (isTRUE(ds.flower.superlink.status()$running) ||
      nrow(dsFlower:::.supernode_list()) != 0L || !isTRUE(cleanup_ok)) {
    stop("Flower process cleanup did not drain the local federation.",
         call. = FALSE)
  }

  list(
    metrics = metrics,
    n_clients = as.integer(metadata$n_clients),
    n_failures = as.integer(sum(history$n_failures)),
    n_rounds_run = nrow(history),
    model_sha256 = digest::digest(file = model_path, algo = "sha256"),
    elapsed_s = as.numeric(difftime(Sys.time(), started, units = "secs"))
  )
}

# ---------------------------------------------------------------------------
# Environment/version capture for the evidence artifact
# ---------------------------------------------------------------------------

campaign_env_info <- function(venv_root) {
  python <- file.path(venv_root, "pytorch", "bin", "python")
  py_versions <- trimws(processx::run(
    python,
    c("-c", paste0("import flwr, torch, opacus; ",
                   "print(flwr.__version__); print(torch.__version__); ",
                   "print(opacus.__version__)")),
    error_on_status = TRUE)$stdout)
  py_versions <- strsplit(py_versions, "\n")[[1]]
  info <- Sys.info()
  list(
    host = list(
      nodename = unname(info[["nodename"]]),
      sysname = unname(info[["sysname"]]),
      release = unname(info[["release"]]),
      machine = unname(info[["machine"]]),
      r_version = R.version.string
    ),
    versions = list(
      dsflower = as.character(utils::packageVersion("dsFlower")),
      dsflowerclient = as.character(utils::packageVersion("dsFlowerClient")),
      flwr = py_versions[1],
      torch = py_versions[2],
      opacus = py_versions[3]
    )
  )
}
