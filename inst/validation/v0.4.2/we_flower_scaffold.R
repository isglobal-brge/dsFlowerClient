# Custodian-side scaffold for the chapter-7 worked example: a three-site
# DSLite federation with node-owned privacy contracts, mirroring the
# campaign harness (process-isolated peers; one DSLite server per PSOCK
# worker). Returns analyst-facing DataSHIELD connections.
source(file.path(Sys.getenv("CAMPAIGN_TOOLS_DIR"), "campaign_lib.R"))

we_setup <- function(site_data, work_dir, venv_root,
                     epsilon = 1, delta = 1e-6) {
  n_sites <- length(site_data)
  dir.create(work_dir, recursive = TRUE, showWarnings = FALSE, mode = "0700")
  ports <- .campaign_free_ports(3L + n_sites)
  names(ports) <- c("fleet", "control", "serverappio",
                    paste0("site", seq_len(n_sites)))
  cluster <- parallel::makePSOCKcluster(n_sites, outfile = "")
  worker_libpaths <- .libPaths()
  parallel::clusterMap(
    cluster,
    function(index, data, work_dir, libpaths, venv_root, epsilon, delta) {
      .libPaths(libpaths)
      Sys.setenv(
        DSFLOWER_VENV_ROOT = venv_root,
        DSFLOWER_NODE_SECRET_FILE = file.path(
          work_dir, paste0("node-secret-site", index)),
        DSFLOWER_TEST_ALLOW_EPHEMERAL_SECRET = "1")
      options(
        dsflower.venv_root = venv_root,
        dsflower.dp_per_training_epsilon = epsilon,
        dsflower.dp_per_training_delta = delta)
      suppressPackageStartupMessages({
        library(DSI); library(DSLite); library(dsFlower)})
      config <- DSLite::defaultDSConfiguration(
        include = c("dsBase", "dsFlower"))
      server <- DSLite::newDSLiteServer(
        tables = list(training = data), config = config,
        home = file.path(work_dir, paste0("dslite-site", index)))
      symbol <- paste0("dsflower_we_site", index, "_", Sys.getpid())
      assign(symbol, server, envir = .GlobalEnv)
      .campaign_dslite_conn <<- DSLite::dsConnect(
        DSLite::DSLite(), name = paste0("site", index), url = symbol)
      DSLite::dsAssignTable(.campaign_dslite_conn, "D", "training")
      TRUE
    },
    index = seq_len(n_sites), data = site_data,
    MoreArgs = list(work_dir = work_dir, libpaths = worker_libpaths,
                    venv_root = venv_root, epsilon = epsilon, delta = delta),
    SIMPLIFY = FALSE)
  dummy_server <- DSLite::newDSLiteServer(tables = list())
  conns <- stats::setNames(lapply(seq_len(n_sites), function(i) {
    methods::new("CampaignDSLiteConnection",
                 name = paste0("site", i), sid = paste0("remote-site", i),
                 server = dummy_server, worker = cluster[i])
  }), paste0("site", seq_len(n_sites)))
  options(
    datashield.errors.print = TRUE,
    dsflower.tunnel_port = unname(ports[paste0("site", seq_len(n_sites))]),
    dsflower.superlink_insecure = TRUE,
    dsflower.tunnel_loss_tolerance = 30,
    dsflower.supernode_term_grace = 10)
  ds.flower.superlink.start(
    fleet_port = ports[["fleet"]], control_port = ports[["control"]],
    serverappio_port = ports[["serverappio"]], insecure = TRUE)
  we_env <- new.env()
  we_env$cluster <- cluster
  we_env$conns <- conns
  we_env
}

we_teardown <- function(we_env) {
  try(ds.flower.link.down(we_env$conns), silent = TRUE)
  try(parallel::clusterCall(we_env$cluster, function() {
    cid <- dsFlower:::.dsflower_env$tunnel_conn_id
    if (!is.null(cid)) try(dsFlower::flowerTunnelDownDS(cid), silent = TRUE)
    nodes <- dsFlower:::.supernode_list()
    if (nrow(nodes)) for (m in nodes$manifest_dir)
      try(dsFlower:::.supernode_stop(m), silent = TRUE)
    if (exists(".campaign_dslite_conn", envir = .GlobalEnv, inherits = FALSE))
      try(DSI::dsDisconnect(.campaign_dslite_conn), silent = TRUE)
    TRUE
  }), silent = TRUE)
  try(parallel::stopCluster(we_env$cluster), silent = TRUE)
  try(ds.flower.superlink.stop(), silent = TRUE)
  invisible(NULL)
}
