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
methods::setMethod("dsListSymbols", "CampaignDSLiteConnection", function(conn) {
  .campaign_remote_call(conn, function() {
    DSI::dsListSymbols(.campaign_dslite_conn)
  })
})
methods::setMethod("dsRmSymbol", "CampaignDSLiteConnection", function(conn, symbol) {
  .campaign_remote_call(conn, function(symbol) {
    DSI::dsRmSymbol(.campaign_dslite_conn, symbol)
  }, symbol)
})
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
      # Observe only metadata written by the unchanged node package. The fit
      # pipeline removes its manifests during cleanup, so retain public fields
      # immediately after preparation, never raw rows or credentials.
      paths <- file.path(dsFlower:::.supernode_list()$manifest_dir, "manifest.json")
      paths <- paths[file.exists(paths)]
      if (length(paths)) {
        .campaign_node_manifests <<- lapply(paths, function(path) {
          manifest <- jsonlite::fromJSON(path, simplifyVector = FALSE)
          manifest[grepl(paste0("^(privacy-|dp[-_]|patient[-_]|feature[-_]",
            "|target[-_]|num-|loss-|learning-|batch-|local-|optimizer|",
            "scheduler|weight-|l1-|n_units$|n_samples$|fixed_client_sampling$|",
            "allow_per_node_metrics$|allow_exact_num_examples$)"), names(manifest))]
        })
      }
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
                                   target_bounds, target = "target",
                                   patient_column = NULL, score_function = NULL) {
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
    function(index, data, work_dir, libpaths, venv_root, epsilon, delta,
             patient_column) {
      .libPaths(libpaths)
      Sys.setenv(
        DSFLOWER_VENV_ROOT = venv_root,
        DSFLOWER_NODE_SECRET_FILE = file.path(
          work_dir, paste0("node-secret-site", index)),
        DSFLOWER_TEST_ALLOW_EPHEMERAL_SECRET = "1"
      )
      .campaign_node_manifests <<- list()
      # Node-side administrator-pinned privacy contract (read by
      # dsFlower:::.privacy_policy() through .dsf_option()).
      options(
        dsflower.venv_root = venv_root,
        dsflower.dp_per_training_epsilon = epsilon,
        dsflower.dp_per_training_delta = delta
      )
      if (!is.null(patient_column)) {
        options(dsflower.dp_unit = "patient",
                dsflower.patient_column = patient_column,
                dsflower.dp_clipping_norm = 1)
      }
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
      delta = delta, patient_column = patient_column),
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

  fit <- withCallingHandlers(ds.flower.fit(
    conns,
    symbol = "D",
    target = target,
    features = features,
    model = contract,
    model_params = model_params,
    strategy = "fedavg",
    rounds = as.integer(rounds),
    feature_bounds = feature_bounds,
    target_bounds = target_bounds,
    torch_backend = "cpu",
    output_dir = output_dir,
    silent = TRUE,
    verbose = FALSE
  ), error = function(e) {
    if (is.null(patient_column)) return(invisible(NULL))
    # Public regression campaign diagnostics only; preserve the original failure
    # and its cleanup while retaining traces that otherwise disappear with Rtmp.
    try({
    diagnostic <- character()
    for (frame in sys.frames()) {
      for (name in c("clean_stdout", "clean_stderr")) {
        if (exists(name, envir = frame, inherits = FALSE)) {
          value <- get(name, envir = frame, inherits = FALSE)
          if (is.character(value)) diagnostic <- c(diagnostic, name, value)
        }
      }
    }
    link_log <- file.path(tempdir(), "dsflower_superlink", "superlink.log")
    if (file.exists(link_log)) diagnostic <- c(
      diagnostic, "SuperLink", readLines(link_log, warn = FALSE))
    node_logs <- try(parallel::clusterCall(cluster, function() {
      paths <- list.files(file.path(tempdir(), "dsflower", "supernodes"),
                          pattern = "\\.log$", full.names = TRUE)
      unlist(lapply(paths, function(path) {
        utils::tail(readLines(path, warn = FALSE), 300L)
      }), use.names = FALSE)
    }), silent = TRUE)
    if (!inherits(node_logs, "try-error")) {
      diagnostic <- c(diagnostic, "SuperNodes", unlist(node_logs, use.names = FALSE))
    }
    diagnostic <- unlist(strsplit(diagnostic, "\n", fixed = TRUE), use.names = FALSE)
    sensitive <- grepl("token|secret|authorization|bearer", diagnostic, ignore.case = TRUE)
    diagnostic[sensitive] <- "[authentication-related diagnostic line redacted]"
    diagnostic <- gsub("[A-Za-z0-9_-]{20,}\\.[A-Za-z0-9_-]{20,}\\.[A-Za-z0-9_-]{20,}",
                       "[credential redacted]", diagnostic)
    writeLines(diagnostic, file.path(work_dir, "failure-diagnostics.log"))
    }, silent = TRUE)
    invisible(NULL)
  })

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

  # Capture node-authored policy/configuration before federation teardown.
  node_privacy <- parallel::clusterCall(cluster, function() {
    list(policy = dsFlower:::.privacy_policy_status(),
         clipping_norm = dsFlower:::.serverDpClippingNorm(),
         staged_configurations = .campaign_node_manifests)
  })
  metrics <- score_function(fit, test, features)

  cleanup()
  Sys.sleep(1)
  if (isTRUE(ds.flower.superlink.status()$running) ||
      nrow(dsFlower:::.supernode_list()) != 0L || !isTRUE(cleanup_ok)) {
    stop("Flower process cleanup did not drain the local federation.",
         call. = FALSE)
  }

  list(
    metrics = metrics,
    node_privacy = node_privacy,
    artifact_dir = persisted_dir,
    cleanup_ok = cleanup_ok,
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
