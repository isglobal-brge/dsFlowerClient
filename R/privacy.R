# Module: Differential Privacy budget (query only)
#
# dsFlower ALWAYS enforces formal differential privacy, and the DATA NODE alone
# decides and enforces it: epsilon / delta / clipping come from the node's own
# DataSHIELD options (with hard ceilings), the mechanism (Opacus DP-SGD,
# DP-GBDT, the output-perturbation floor) is chosen server-side by code identity,
# and the RDP/PRV budget ledger is the custodian's resource. The client does NOT
# set privacy -- it cannot weaken it, nor even request a different value (that
# choice is the data custodian's, never the analyst's). The client can only
# QUERY how much budget remains, which is what this module exposes.

#' Query remaining differential-privacy budget on all servers
#'
#' Differential privacy is always on and is decided + enforced entirely by each
#' data node (server-authoritative): the node sets epsilon/delta/clipping from its
#' own options, picks the mechanism by submission type, and debits an RDP/PRV
#' ledger. There is no client-side privacy knob. Use this to read the ACTUAL
#' remaining (epsilon, delta) budget the node will apply to the dataset.
#'
#' @param conns DSI connections object.
#' @param symbol Character; the DATA symbol assigned on the servers (default "D").
#' @param target Character; the outcome column the budget is keyed by (the ledger is
#'   per-dataset-and-target). Recommended -- without it the per-target budget cannot be
#'   located. For survival pass \code{c("time", "event")}.
#' @return Per-server budget information (the node's real values). Read-only: querying never
#'   debits the budget.
#' @export
ds.flower.privacy.budget <- function(conns, symbol = "D", target = NULL) {
  # Validate the target against the schema: a typo would silently read the wrong (or empty)
  # per-target budget key. Column NAMES are non-disclosive (standard colnames aggregate).
  if (!is.null(target)) {
    cols <- tryCatch(DSI::datashield.aggregate(conns, call("colnamesDS", symbol)),
                     error = function(e) NULL)
    if (!is.null(cols) && length(cols) && length(cols[[1L]])) {
      miss <- setdiff(as.character(target), as.character(cols[[1L]]))
      if (length(miss)) {
        stop("target column(s) not found in '", symbol, "': ",
             paste(miss, collapse = ", "),
             ". The budget is keyed by the training target.", call. = FALSE)
      }
    }
  }
  # Open a TRANSIENT handle (carries the data fingerprint) purely to read the ledger; never
  # prepares or debits. Pass the target so the correct per-target key is read.
  flower <- ds.flower.connect(conns, symbol = symbol)
  on.exit(try(ds.flower.disconnect(flower), silent = TRUE), add = TRUE)
  tc <- paste(as.character(target %||% ""), collapse = "+")
  DSI::datashield.aggregate(
    flower$conns, expr = call("flowerPrivacyBudgetDS", flower$symbol, tc)
  )
}
