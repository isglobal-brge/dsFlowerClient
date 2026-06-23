# Module: Client-side Model Registry
#
# dsFlower has NO server-side model catalog. The node runs whatever DP-valid
# submission it is sent; the "catalog" is purely a CLIENT-SIDE convenience that
# turns a friendly name + params into the artifact shipped in the FAB:
#   * neural track -> python source for `build_model(cfg) -> nn.Module`
#   * trees  track -> a validated XGBoost spec (data, never code)
#
# The registry is extensible the way tidymodels/parsnip is: a derived
# dsFlowerClient extension package registers its own model collection by calling
# `ds.flower.register_model()` from its `.onLoad()`. Registration is client-side
# state only — it never adds anything server-side.

# Internal registry environment (parsnip's get_model_env() analogue).
.dsflower_models <- new.env(parent = emptyenv())

#' Register a dsFlower model generator
#'
#' Intended for dsFlowerClient extension packages: call this from your package's
#' \code{.onLoad()} to add models to the registry. The model becomes usable via
#' \code{ds.flower.model("<name>")} / \code{ds.flower.fit(..., model = "<name>")}.
#'
#' @param name Character; the model name (e.g. "pytorch_logreg", "xgboost").
#' @param track Character; the enforced-DP track: "neural" (nn.Module + DP-SGD)
#'   or "trees" (XGBoost spec + DP-GBDT).
#' @param generate Function of one argument \code{params} (a named list). For the
#'   neural track it MUST return a single character string: the python source of
#'   a module defining \code{build_model(cfg) -> torch.nn.Module} and nothing
#'   else (no training loop, no loss — the node owns those). For the trees track
#'   it MUST return a named list: the XGBoost spec.
#' @param loss Character or NULL; neural track only. The per-sample loss the node
#'   should use, from the node allowlist (\code{bce_logits}, \code{cross_entropy},
#'   \code{mse}, \code{poisson_nll}, \code{multilabel_bce}, \code{lognormal_aft_nll}).
#'   The node pins the actual loss; this is the client's request.
#' @param defaults Named list of default params merged under user-supplied params.
#' @param description Character or NULL; a one-line human description.
#' @param vetted Logical; TRUE for first-party generators whose emitted
#'   architecture is known to be free of cross-sample coupling and weight-stash
#'   buffers (so it runs on the default path). Third-party registrations default
#'   to FALSE and are subject to the node's custom-code admission gate.
#' @param overwrite Logical; allow replacing an existing registration.
#' @return Invisibly, the model name.
#' @export
ds.flower.register_model <- function(name, track, generate, loss = NULL,
                                     defaults = list(), description = NULL,
                                     vetted = FALSE, overwrite = FALSE) {
  if (!is.character(name) || length(name) != 1L || !nzchar(name)) {
    stop("'name' must be a single non-empty string.", call. = FALSE)
  }
  track <- match.arg(track, c("neural", "trees"))
  if (!is.function(generate)) {
    stop("'generate' must be a function(params) -> source/spec.", call. = FALSE)
  }
  if (!is.null(loss)) {
    allowed <- c("bce_logits", "cross_entropy", "mse", "poisson_nll",
                 "multilabel_bce", "lognormal_aft_nll")
    if (!is.character(loss) || length(loss) != 1L || !loss %in% allowed) {
      stop("'loss' must be one of the node allowlist: ",
           paste(allowed, collapse = ", "), ".", call. = FALSE)
    }
  }
  if (!isTRUE(overwrite) && !is.null(.dsflower_models[[name]])) {
    stop("model '", name, "' already registered; pass overwrite = TRUE to replace.",
         call. = FALSE)
  }
  .dsflower_models[[name]] <- list(
    name = name, track = track, generate = generate, loss = loss,
    defaults = as.list(defaults), description = description,
    vetted = isTRUE(vetted)
  )
  invisible(name)
}

#' List registered dsFlower models
#'
#' @return A data.frame with one row per registered model.
#' @export
ds.flower.list_models <- function() {
  names_ <- ls(.dsflower_models, sorted = TRUE)
  if (!length(names_)) {
    return(data.frame(name = character(), track = character(),
                      loss = character(), vetted = logical(),
                      description = character(), stringsAsFactors = FALSE))
  }
  do.call(rbind, lapply(names_, function(nm) {
    m <- .dsflower_models[[nm]]
    data.frame(name = m$name, track = m$track,
               loss = m$loss %||% NA_character_, vetted = m$vetted,
               description = m$description %||% NA_character_,
               stringsAsFactors = FALSE)
  }))
}

#' Look up a registered model (internal)
#' @keywords internal
.dsflower_get_model <- function(name) {
  m <- .dsflower_models[[name]]
  if (is.null(m)) {
    avail <- ls(.dsflower_models, sorted = TRUE)
    stop("Unknown model '", name, "'. Registered models: ",
         paste(avail, collapse = ", "),
         ". (Extension packages add more via ds.flower.register_model().)",
         call. = FALSE)
  }
  m
}

# --------------------------------------------------------------------------- #
# Built-in generators (first-party, vetted = stash-free architectures).
# --------------------------------------------------------------------------- #

# A generic feed-forward classifier/regressor head. Emits ONLY nn.Linear / ReLU
# in an nn.Sequential — no buffers, no BatchNorm, no custom forward, so there is
# no weight-stash channel and per-sample gradients are well-defined. `out_expr`
# decides the output width from cfg (single logit for binary; num-classes for
# multiclass; fixed for regression/poisson/multilabel).
.neural_mlp_source <- function(hidden = integer(0), out_expr = "1",
                               input_key = "num-features") {
  hid <- if (length(hidden)) paste(as.integer(hidden), collapse = ", ") else ""
  paste0(
    "import torch.nn as nn\n\n",
    "def build_model(cfg):\n",
    "    d = int(cfg[\"", input_key, "\"])\n",
    "    out = ", out_expr, "\n",
    "    hidden = [", hid, "]\n",
    "    if not hidden:\n",
    "        return nn.Linear(d, out)\n",
    "    layers = []\n",
    "    for w in hidden:\n",
    "        layers += [nn.Linear(d, int(w)), nn.ReLU()]\n",
    "        d = int(w)\n",
    "    layers.append(nn.Linear(d, out))\n",
    "    return nn.Sequential(*layers)\n")
}

# Binary/multiclass output width: 1 logit when <=2 classes, else num-classes.
.OUT_CLASSES <- "1 if int(cfg.get(\"num-classes\", 2)) <= 2 else int(cfg[\"num-classes\"])"

#' Register the first-party model collection (called from .onLoad)
#' @keywords internal
.dsflower_register_builtins <- function(overwrite = TRUE) {
  reg <- function(...) ds.flower.register_model(..., vetted = TRUE, overwrite = overwrite)

  # ---- neural: tabular ----
  reg("pytorch_logreg", "neural",
      generate = function(p) .neural_mlp_source(integer(0), .OUT_CLASSES),
      loss = "bce_logits",
      description = "Logistic regression (single linear layer).")
  reg("pytorch_mlp", "neural",
      generate = function(p) .neural_mlp_source(p$hidden_layers %||% c(64L, 32L), .OUT_CLASSES),
      loss = "bce_logits", defaults = list(hidden_layers = c(64L, 32L)),
      description = "Multilayer perceptron classifier.")
  reg("pytorch_multiclass", "neural",
      generate = function(p) .neural_mlp_source(p$hidden_layers %||% integer(0), .OUT_CLASSES),
      loss = "cross_entropy", defaults = list(hidden_layers = integer(0)),
      description = "Multiclass classifier (softmax cross-entropy).")
  reg("pytorch_linear_regression", "neural",
      generate = function(p) .neural_mlp_source(p$hidden_layers %||% integer(0), "1"),
      loss = "mse", defaults = list(hidden_layers = integer(0)),
      description = "Linear / MLP regression (MSE).")
  reg("pytorch_poisson", "neural",
      generate = function(p) .neural_mlp_source(p$hidden_layers %||% integer(0), "1"),
      loss = "poisson_nll", defaults = list(hidden_layers = integer(0)),
      description = "Poisson regression (count outcomes).")
  reg("pytorch_multilabel", "neural",
      generate = function(p) .neural_mlp_source(p$hidden_layers %||% integer(0),
                                                "int(cfg[\"num-labels\"])"),
      loss = "multilabel_bce", defaults = list(num_labels = 2L),
      description = "Multilabel classifier (independent BCE per label).")

  # ---- neural: vision head (frozen backbone is node-resident; client ships
  #      only the trainable head, keyed off the injected feature-dim) ----
  for (nm in c("pytorch_resnet18", "pytorch_densenet121")) {
    reg(nm, "neural",
        generate = function(p) .neural_mlp_source(integer(0), .OUT_CLASSES,
                                                  input_key = "feature-dim"),
        loss = "cross_entropy", defaults = list(n_classes = 2L),
        description = paste0("Vision classifier head on a frozen ", nm, " backbone."))
  }

  # ---- trees: XGBoost spec (DATA, never code) ----
  reg("xgboost", "trees",
      generate = function(p) {
        list(
          objective     = p$objective %||% "binary:logistic",
          max_depth     = as.integer(p$max_depth %||% 3L),
          n_trees       = as.integer(p$n_trees %||% 20L),
          learning_rate = as.numeric(p$learning_rate %||% 0.3),
          reg_lambda    = as.numeric(p$reg_lambda %||% 1.0),
          n_bins        = as.integer(p$n_bins %||% 32L),
          feature_ranges = p$feature_ranges %||% NULL  # node clamps/pins if absent
        )
      },
      defaults = list(objective = "binary:logistic", max_depth = 3L, n_trees = 20L,
                      learning_rate = 0.3, reg_lambda = 1.0, n_bins = 32L),
      description = "Gradient-boosted trees (enforced DP-GBDT, S-GBDT mechanism).")

  invisible(TRUE)
}
