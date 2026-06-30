# Module: Flower Hub / external app loader
#
# Load a Flower app from (a) a local directory, (b) a local .fab bundle, or (c) a
# Flower Hub reference (@account/app), and run it on the dsFlower federation under the
# SERVER-ENFORCED output-perturbation floor (Tier-2). The data-holding node decides and
# applies the DP: ANY app loaded here is routed to the floor and CANNOT bypass it (the
# node forces uploaded code to output-perturbation -- dp_harness.resolve_dp_track).
#
# The app must expose the dsFlower user-module contract so the node's floor harness can
# drive it on the node's PRIVATE data:
#   initial_arrays(cfg, input_dim)            -> list of numpy arrays (initial model)
#   local_update(global_arrays, X, y, cfg)    -> list of numpy arrays (updated model)
# A generic full-Flower app that ships its own ClientApp + its own data does NOT fit and
# cannot run under dsFlower's server-enforced DP (it would neither see the node's data
# nor be wrapped by the node's DP gate).

#' Find a dsFlower-compatible user-module inside an unpacked app/FAB tree.
#' @keywords internal
.find_user_pkg <- function(root) {
  pyf <- list.files(root, pattern = "\\.py$", recursive = TRUE, full.names = TRUE)
  for (f in pyf) {
    txt <- paste(readLines(f, warn = FALSE), collapse = "\n")
    if (grepl("def[[:space:]]+initial_arrays", txt) &&
        grepl("def[[:space:]]+local_update", txt)) {
      return(normalizePath(dirname(f)))
    }
  }
  stop("No dsFlower-compatible user-module found under '", root, "'. The app must ",
       "define initial_arrays(cfg, input_dim) and local_update(global_arrays, X, y, cfg). ",
       "A generic Flower app that ships its own ClientApp/data cannot run under ",
       "dsFlower's server-enforced DP; rewrite it to the user-module contract.",
       call. = FALSE)
}

#' Resolve a Flower app reference to a local user-module package directory.
#' @keywords internal
.resolve_flower_app <- function(app, verbose = TRUE) {
  if (startsWith(app, "@")) {                      # Flower Hub reference @account/app
    dest <- tempfile("dsflower_hubapp_"); dir.create(dest)
    if (verbose) message("  Fetching '", app, "' from Flower Hub via flwr ",
                         "(requires `flwr login`)...")
    out <- tryCatch(
      system2("flwr", c("install", app, "--dir", dest), stdout = TRUE, stderr = TRUE),
      error = function(e) structure(conditionMessage(e), status = 1L))
    status <- attr(out, "status")
    if (!is.null(status) && status != 0) {
      stop("Could not fetch '", app, "' from Flower Hub (",
           paste(utils::tail(as.character(out), 2L), collapse = " "),
           "). Run `flwr login` and verify the app id.", call. = FALSE)
    }
    return(.find_user_pkg(dest))
  }
  if (grepl("\\.fab$", app, ignore.case = TRUE) && file.exists(app)) {  # local .fab bundle
    dest <- tempfile("dsflower_fab_"); dir.create(dest)
    utils::unzip(app, exdir = dest)
    return(.find_user_pkg(dest))
  }
  if (dir.exists(app)) return(.find_user_pkg(normalizePath(app)))       # local directory
  stop("Cannot resolve Flower app '", app, "': expected a Flower Hub reference ",
       "(@account/app), a .fab file, or a local directory.", call. = FALSE)
}

#' Run a Flower app on the dsFlower federation under server-enforced DP
#'
#' Loads a Flower app from a local directory, a local \code{.fab} bundle, or a Flower
#' Hub reference (\code{@account/app}), and runs it on the federation. The app is ALWAYS
#' routed to the node-side output-perturbation floor (Tier-2): the data-holding node
#' decides and enforces the differential privacy, and the app cannot bypass or weaken it.
#' The app must expose the dsFlower user-module contract (\code{initial_arrays} +
#' \code{local_update}); a generic Flower app that ships its own ClientApp/data cannot
#' run under dsFlower's DP.
#'
#' @param conns DataSHIELD connections.
#' @param app A Flower Hub reference (\code{"@account/app"}), a path to a \code{.fab}
#'   file, or a path to a local user-module package directory.
#' @param target Character; the outcome column.
#' @param features Character vector; the feature columns.
#' @param symbol Character; the data symbol to assign on the nodes.
#' @param num_rounds Integer; FL rounds.
#' @param privacy Ignored for the DP decision (the node decides); kept for signature
#'   compatibility with the Tier-2 path.
#' @param verbose Logical.
#' @return A \code{dsflower_run} result (see \code{ds.flower.tier2.run}).
#' @export
ds.flower.hub.run <- function(conns, app, target, features, symbol = "HUBAPP",
                              num_rounds = 1L, privacy = NULL, verbose = TRUE) {
  pkg_dir <- .resolve_flower_app(app, verbose = verbose)
  if (verbose) {
    message("  Loaded app package: ", basename(pkg_dir))
    message("  -> routed to the SERVER-ENFORCED Tier-2 output-perturbation floor ",
            "(the node decides + applies DP; the app cannot bypass it).")
  }
  ds.flower.link.up(conns)
  on.exit(try(ds.flower.link.down(conns), silent = TRUE), add = TRUE)
  ds.flower.tier2.run(conns, user_app_dir = pkg_dir, target = target,
                      features = features, symbol = symbol,
                      num_rounds = num_rounds, privacy = privacy, verbose = verbose)
}
