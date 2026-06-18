# Module: DSI-relay transport (client orchestration, Phase 0)
#
# Runs a federated training loop ENTIRELY over the DataSHIELD channel: the
# researcher dials OUT to the public Opal endpoints (no NAT, no Tor) and drives
# each round with a single parallel datashield.aggregate fan-out. Phase 0 is
# FedSGD for logistic regression with gradients returned in the clear; SecAgg+
# masking is layered on in a later phase.

# Encode a payload as a B64:-prefixed JSON string. DataSHIELD's restricted parser
# rejects raw JSON literals (brackets/quotes), so payloads cross base64-encoded.
.dsi_enc <- function(obj) {
  b64 <- gsub("[\r\n]", "", jsonlite::base64_enc(
    charToRaw(as.character(jsonlite::toJSON(obj, auto_unbox = TRUE, digits = 12)))))
  # URL-safe base64 matching dsFlower's node-side .ds_arg decoder: no +/= or
  # newlines, which would otherwise break Opal's restricted R-expression parser.
  b64 <- gsub("\\+", "-", b64)
  b64 <- gsub("/", "_", b64)
  b64 <- gsub("=+$", "", b64)
  paste0("B64:", b64)
}

#' Fit a federated logistic regression over the DataSHIELD channel (no Tor)
#'
#' @param conns DSI connections object.
#' @param symbol Character; the data.frame symbol on the servers (e.g. "D").
#' @param target Character; binary 0/1 target column.
#' @param features Character vector; feature columns.
#' @param rounds Integer; FedSGD rounds.
#' @param lr Numeric; global learning rate.
#' @param standardize Logical; standardize features with FEDERATED global
#'   mean/sd (computed once over DataSHIELD) so SGD converges on raw clinical
#'   scales. Recommended.
#' @param verbose Logical; print per-round progress.
#' @return A \code{dsflower_dsi_model}.
#' @export
ds.flower.dsi.fit <- function(conns, symbol, target, features,
                              rounds = 30L, lr = 0.5,
                              standardize = TRUE, verbose = FALSE) {
  feats_b64 <- .dsi_enc(features)
  mu <- rep(0, length(features)); sdv <- rep(1, length(features))

  if (isTRUE(standardize)) {
    # Federated standardization: pull each site's sum, sumsq, n (one round) and
    # combine into global mean/sd. Itself privacy-preserving (site-level sums).
    st <- DSI::datashield.aggregate(
      conns, call("flowerDsiStatsDS", symbol, feats_b64))
    S  <- Reduce(`+`, lapply(st, `[[`, "sum"))
    S2 <- Reduce(`+`, lapply(st, `[[`, "sumsq"))
    N  <- sum(vapply(st, `[[`, numeric(1), "n"))
    mu <- S / N
    v  <- pmax(S2 / N - mu^2, 1e-8)
    sdv <- sqrt(v)
  }
  stats_b64 <- .dsi_enc(list(mu = mu, sd = sdv))

  w <- rep(0, length(features) + 1L)
  t0 <- Sys.time()
  history <- numeric(rounds)
  for (r in seq_len(rounds)) {
    res <- DSI::datashield.aggregate(
      conns, call("flowerDsiStepDS", symbol, target, feats_b64,
                  .dsi_enc(w), stats_b64))
    G <- Reduce(`+`, lapply(res, `[[`, "grad"))
    N <- sum(vapply(res, `[[`, numeric(1), "n"))
    w <- w - lr * (G / N)
    history[r] <- sqrt(sum((G / N)^2))   # gradient norm
    if (isTRUE(verbose)) message(sprintf("  round %d/%d  |grad|=%.4g", r, rounds, history[r]))
  }
  structure(list(
    weights = w, features = features, target = target,
    mu = mu, sd = sdv, standardize = standardize,
    rounds = rounds, grad_norm = history,
    elapsed = as.numeric(difftime(Sys.time(), t0, units = "secs"))
  ), class = "dsflower_dsi_model")
}

#' Predict with a DSI-transport logistic-regression model
#' @param object A \code{dsflower_dsi_model}.
#' @param newdata data.frame with the feature columns.
#' @param type "prob" or "class".
#' @param ... unused.
#' @return Numeric vector of probabilities (or 0/1 classes).
#' @export
predict.dsflower_dsi_model <- function(object, newdata, type = c("prob", "class"), ...) {
  type <- match.arg(type)
  X <- data.matrix(as.data.frame(newdata)[, object$features, drop = FALSE])
  if (isTRUE(object$standardize)) {
    X <- sweep(X, 2, object$mu, "-")
    X <- sweep(X, 2, object$sd, "/")
  }
  Xb <- cbind(1, X)
  prob <- as.numeric(1 / (1 + exp(-(Xb %*% object$weights))))
  if (type == "class") as.integer(prob >= 0.5) else prob
}
