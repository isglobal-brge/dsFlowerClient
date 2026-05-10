# dsFlower Opal demo: federated logistic regression across three Opal/Rock nodes.
#
# The script creates one synthetic table per Opal, logs into DataSHIELD,
# initializes dsFlower, and trains a Flower logistic regression model.
#
# Environment overrides:
#   OPAL_USER, OPAL_PASSWORD
#   DSFLOWER_DEMO_URLS      comma-separated Opal URLs
#   DSFLOWER_DEMO_PROJECT   default "dsflower_demo"
#   DSFLOWER_DEMO_PREFIX    default "synthetic_flower"
#   DSFLOWER_DEMO_N         rows per site, default 160
#   DSFLOWER_DEMO_ROUNDS    federated rounds, default 2
#   DSFLOWER_DEMO_PRIVACY   default "trusted_internal"

suppressPackageStartupMessages({
  library(DSI)
  library(DSOpal)
  library(dsFlowerClient)
  library(opalr)
})

`%||%` <- function(x, y) if (is.null(x) || !nzchar(x)) y else x

opal_user <- Sys.getenv("OPAL_USER") %||% "administrator"
opal_password <- Sys.getenv("OPAL_PASSWORD") %||% "admin123"
opal_urls <- strsplit(
  Sys.getenv("DSFLOWER_DEMO_URLS") %||%
    "https://localhost:8443,https://localhost:8444,https://localhost:8445",
  ",", fixed = TRUE
)[[1]]
opal_urls <- trimws(opal_urls)

project <- Sys.getenv("DSFLOWER_DEMO_PROJECT") %||% "dsflower_demo"
prefix <- Sys.getenv("DSFLOWER_DEMO_PREFIX") %||% "synthetic_flower"
n_per_site <- as.integer(Sys.getenv("DSFLOWER_DEMO_N") %||% "160")
num_rounds <- as.integer(Sys.getenv("DSFLOWER_DEMO_ROUNDS") %||% "2")
demo_privacy <- Sys.getenv("DSFLOWER_DEMO_PRIVACY") %||% "trusted_internal"

make_site_data <- function(site, n) {
  set.seed(9000 + site)
  x1 <- rnorm(n, mean = (site - 2) * 0.15)
  x2 <- rnorm(n)
  x3 <- rnorm(n, mean = site * 0.05)
  x4 <- rnorm(n)
  noise <- rnorm(n, sd = 0.65)
  score <- 1.25 * x1 - 0.9 * x2 + 0.55 * x3 + 0.25 * x4 + noise
  outcome <- as.integer(score > stats::median(score))
  data.frame(
    id = sprintf("site%d_%03d", site, seq_len(n)),
    x1 = x1,
    x2 = x2,
    x3 = x3,
    x4 = x4,
    outcome = outcome
  )
}

privacy_spec <- function(name) {
  switch(name,
    sandbox_open = ds.flower.privacy.sandbox_open(),
    trusted_internal = ds.flower.privacy.trusted_internal(),
    consortium_internal = ds.flower.privacy.consortium_internal(),
    clinical_default = ds.flower.privacy.clinical_default(),
    clinical_hardened = ds.flower.privacy.clinical_hardened(),
    clinical_update_noise = ds.flower.privacy.clinical_update_noise(),
    high_sensitivity_dp = ds.flower.privacy.high_sensitivity_dp(),
    stop("Unsupported DSFLOWER_DEMO_PRIVACY value: ", name, call. = FALSE)
  )
}

configure_dsflower_profile <- function(opal, name) {
  opalr::dsadmin.set_option(opal, "dsflower.privacy_profile", name,
                            profile = "default")
  if (identical(name, "sandbox_open")) {
    opalr::dsadmin.set_option(opal, "dsflower.allow_sandbox", TRUE,
                              profile = "default")
  }
}

upload_table <- function(url, site, data) {
  opal <- opalr::opal.login(
    username = opal_user, password = opal_password, url = url,
    opts = list(ssl_verifyhost = 0L, ssl_verifypeer = 0L)
  )
  on.exit(opalr::opal.logout(opal), add = TRUE)

  configure_dsflower_profile(opal, demo_privacy)

  if (!opalr::opal.project_exists(opal, project)) {
    opalr::opal.project_create(opal, project, database = "mongodb")
  } else {
    datasource_type <- tryCatch(
      opalr::opal.project(opal, project)$datasource$type,
      error = function(e) NA_character_
    )
    if (identical(datasource_type, "null")) {
      opalr::opal.project_delete(opal, project, archive = FALSE, silent = TRUE)
      opalr::opal.project_create(opal, project, database = "mongodb")
    }
  }

  table <- paste0(prefix, "_site", site)
  if (opalr::opal.table_exists(opal, project, table)) {
    opalr::opal.table_delete(opal, project, table)
  }
  opalr::opal.table_save(
    opal, data, project = project, table = table,
    overwrite = TRUE, force = TRUE, id.name = "id", policy = "generate"
  )
  paste(project, table, sep = ".")
}

message("Preparing Opal tables...")
message("Using dsFlower privacy profile: ", demo_privacy)
tables <- character(length(opal_urls))
for (i in seq_along(opal_urls)) {
  dat <- make_site_data(i, n_per_site)
  tables[[i]] <- upload_table(opal_urls[[i]], i, dat)
  message("  site", i, ": ", tables[[i]], " (", nrow(dat), " rows)")
}

builder <- DSI::newDSLoginBuilder()
for (i in seq_along(opal_urls)) {
  builder$append(
    server = paste0("opal", i),
    url = opal_urls[[i]],
    user = opal_user,
    password = opal_password,
    table = tables[[i]],
    driver = "OpalDriver"
  )
}

message("Logging into DataSHIELD...")
conns <- DSI::datashield.login(
  logins = builder$build(), assign = TRUE, symbol = "D"
)
on.exit(DSI::datashield.logout(conns), add = TRUE)

message("Checking dsFlower availability...")
print(DSI::datashield.pkg_status(conns)$package_status)

flower <- ds.flower.connect(conns, symbol = "D")
on.exit(ds.flower.disconnect(flower), add = TRUE)

recipe <- ds.flower.recipe(
  model = ds.flower.model.sklearn_logreg(max_iter = 100L),
  strategy = ds.flower.strategy.fedavg(),
  privacy = privacy_spec(demo_privacy),
  num_rounds = num_rounds,
  target = "outcome",
  features = c("x1", "x2", "x3", "x4")
)

message("Running federated training...")
run <- ds.flower.run(flower, recipe, verbose = TRUE)

message("Run status: ", run$status)
message("Model id: ", run$model_id)
message("Output dir: ", normalizePath(run$output_dir, mustWork = FALSE))
if (!is.null(run$history)) {
  print(run$history)
}
if (!is.null(run$weights)) {
  message("Weights received: ", length(run$weights), " tensors")
}

invisible(run)
