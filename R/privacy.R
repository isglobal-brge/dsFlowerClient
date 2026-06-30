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
#' @param symbol Character; handle symbol name (default "flower").
#' @return Named list with per-server budget information (the node's real values).
#' @export
ds.flower.privacy.budget <- function(conns, symbol = "flower") {
  DSI::datashield.aggregate(
    conns, expr = call("flowerPrivacyBudgetDS", symbol)
  )
}
