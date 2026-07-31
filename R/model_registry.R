# Module: Client-side Model Registry
#
# dsFlower has NO server-side model catalog. The node runs whatever DP-valid
# submission it is sent; the "catalog" is purely a CLIENT-SIDE convenience that
# turns a friendly name + params into the artifact shipped in the FAB:
#   * neural track -> a model SPEC the node builds with stock torch layers
#   * trees  track -> a validated XGBoost spec
# Both are DATA, never code: nothing the researcher submits runs in the node's
# trusted interpreter (which is what makes the DP-release path unforgeable).
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
#' @param generate Function of one argument \code{params} (a named list). Both
#'   tracks MUST return a named list (DATA, never code): for the neural track a
#'   model SPEC \code{list(kind = "sequential", layers = list(...))} the node
#'   builds with stock torch layers (end with a linear onto \code{"@out"}); for the
#'   trees track the XGBoost spec. The node owns the loop, loss, and optimizer.
#' @param loss Character or NULL; neural track only. The per-sample loss the node
#'   should use, from the node allowlist: stock losses \code{bce_logits},
#'   \code{cross_entropy}, \code{mse}, \code{poisson_nll}, \code{multilabel_bce},
#'   \code{hinge} (linear SVM), \code{ordinal} (CORN); plus vetted custom per-sample
#'   losses \code{negbin_nll} (overdispersed counts) and \code{gamma_nll} (positive
#'   continuous). The node pins the actual loss; this is the client's request.
#' @param defaults Named list of default params merged under user-supplied params.
#' @param description Character or NULL; a one-line human description.
#' @param vetted Logical; informational only. Every model is node-built from the
#'   allowlisted spec vocabulary (no researcher code runs), so first- and
#'   third-party generators take the same validated path; this flag just marks
#'   first-party collections in \code{ds.flower.list_models()}.
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
                 "multilabel_bce", "hinge", "negbin_nll", "gamma_nll", "ordinal")
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

# A generic feed-forward head as a declarative SPEC (DATA, never code): a list of
# allowlisted layers the NODE builds with stock torch constructors. `hidden` are
# the hidden widths (each a linear + relu); every model ends with a linear onto the
# symbolic "@out", whose width the node fixes from the pinned loss -- so output
# width is node-authoritative, never client-set. The input dim "@in" is injected
# node-side (num-features, or the frozen-backbone feature dim for vision).
.neural_mlp_spec <- function(hidden = integer(0)) {
  # Legacy constructors stored widths as one TOML-era string ("64,32").
  # Normalize that representation before emitting the declarative graph.
  if (is.character(hidden) && length(hidden) == 1L) {
    hidden <- trimws(hidden)
    hidden <- if (!nzchar(hidden)) integer(0) else
      suppressWarnings(as.integer(strsplit(hidden, ",", fixed = TRUE)[[1L]]))
  }
  if (length(hidden) &&
      (anyNA(hidden) || any(!is.finite(hidden)) || any(hidden < 1L) ||
       any(hidden != floor(hidden)))) {
    stop("hidden_layers must contain positive integer widths.", call. = FALSE)
  }
  layers <- list()
  for (w in hidden) {
    layers <- c(layers, list(list(op = "linear", out = as.integer(w)),
                             list(op = "relu")))
  }
  layers <- c(layers, list(list(op = "linear", out = "@out")))
  list(kind = "sequential", layers = layers)
}

# A small 2D-CNN as a SPEC (DATA, node-built): reshape the flat per-sample vector
# into (C,H,W), then a conv/pool stack -> adaptive pool -> flatten -> linear head.
# input_shape must multiply to the feature count (the node rejects a mismatch).
.neural_cnn_spec <- function(input_shape, channels = c(8L, 16L)) {
  layers <- list(list(op = "reshape", shape = as.list(as.integer(input_shape))))
  for (ch in channels)
    layers <- c(layers, list(
      list(op = "conv2d", out_channels = as.integer(ch), kernel_size = 3L, padding = 1L),
      list(op = "relu"),
      list(op = "maxpool2d", kernel_size = 2L)))
  layers <- c(layers, list(
    list(op = "adaptiveavgpool2d", output_size = list(1L, 1L)),
    list(op = "flatten"),
    list(op = "linear", out = "@out")))
  list(kind = "sequential", layers = layers)
}

# A Temporal CNN as a SPEC (DATA, node-built): reshape into (C,L), then a dilated
# length-preserving conv1d stack (receptive field grows as 2^level) -> flatten -> head.
.neural_tcn_spec <- function(input_shape, channels = 8L, levels = 3L) {
  layers <- list(list(op = "reshape", shape = as.list(as.integer(input_shape))))
  for (i in seq_len(levels)) {
    d <- as.integer(2L^(i - 1L))
    layers <- c(layers, list(
      list(op = "conv1d", out_channels = as.integer(channels),
           kernel_size = 3L, padding = d, dilation = d),
      list(op = "relu")))
  }
  layers <- c(layers, list(list(op = "flatten"), list(op = "linear", out = "@out")))
  list(kind = "sequential", layers = layers)
}

# A residual CNN block as a typed GRAPH (DAG) spec (DATA, node-built): reshape the flat
# vector into (C,H,W), conv -> relu -> conv, ADD the skip, global-pool -> flatten -> head.
# Demonstrates the graph language (named tensors, multi-input 'add'); the node builds a
# GraphModule from allowlisted per-sample-safe ops only. input_shape multiplies to the
# feature count (the node rejects a mismatch).
.neural_resnet_spec <- function(input_shape, channels = 8L) {
  ish <- as.list(as.integer(input_shape))
  ch <- as.integer(channels)
  list(kind = "graph", output = "out", nodes = list(
    list(name = "img",  op = "reshape", `in` = list("@in"), shape = ish),
    list(name = "c1",   op = "conv2d",  `in` = list("img"), out_channels = ch, kernel_size = 3L, padding = 1L),
    list(name = "r1",   op = "relu",    `in` = list("c1")),
    list(name = "c2",   op = "conv2d",  `in` = list("r1"),  out_channels = ch, kernel_size = 3L, padding = 1L),
    list(name = "res",  op = "add",     `in` = list("c1", "c2")),
    list(name = "pool", op = "adaptiveavgpool2d", `in` = list("res"), output_size = list(1L, 1L)),
    list(name = "flat", op = "flatten", `in` = list("pool")),
    list(name = "out",  op = "linear",  `in` = list("flat"), out = "@out")))
}

# A Transformer ENCODER block as a typed GRAPH (DATA, node-built) -- self-attention
# (Q/K/V linears + matmul + softmax over TOKENS, never the batch) + residual + LayerNorm
# + FFN, built ENTIRELY from per-sample-safe primitives (no Opacus DPMultiheadAttention,
# no custom code). n_tokens * d_model must equal the feature count.
.neural_transformer_spec <- function(n_tokens, d_model, d_ff = 32L) {
  T <- as.integer(n_tokens); d <- as.integer(d_model); ff <- as.integer(d_ff)
  list(kind = "graph", output = "out", nodes = list(
    list(name = "x",   op = "reshape",   `in` = list("@in"), shape = list(T, d)),
    list(name = "q",   op = "linear",    `in` = list("x"),   out = d),
    list(name = "k",   op = "linear",    `in` = list("x"),   out = d),
    list(name = "v",   op = "linear",    `in` = list("x"),   out = d),
    list(name = "kt",  op = "transpose", `in` = list("k"),   dims = list(0L, 1L)),
    list(name = "sc",  op = "matmul",    `in` = list("q", "kt")),
    list(name = "a",   op = "softmax",   `in` = list("sc"),  axis = 1L),
    list(name = "ctx", op = "matmul",    `in` = list("a", "v")),
    list(name = "res", op = "add",       `in` = list("x", "ctx")),
    list(name = "n1",  op = "layernorm", `in` = list("res")),
    list(name = "f1",  op = "linear",    `in` = list("n1"),  out = ff),
    list(name = "fa",  op = "relu",      `in` = list("f1")),
    list(name = "f2",  op = "linear",    `in` = list("fa"),  out = d),
    list(name = "res2", op = "add",      `in` = list("n1", "f2")),
    list(name = "n2",  op = "layernorm", `in` = list("res2")),
    list(name = "flat", op = "flatten",  `in` = list("n2")),
    list(name = "out", op = "linear",    `in` = list("flat"), out = "@out")))
}

# An LSTM/GRU sequence model as a typed GRAPH (DATA, node-built): reshape the flat vector
# into (n_tokens, n_features), run a sanitized Opacus DP-RNN over time, take the last
# hidden state -> head. Recurrence is within a sample (over time), never across the batch.
# n_tokens * n_features must equal the feature count.
.neural_seq_spec <- function(n_tokens, n_features, hidden = 32L, kind = "lstm") {
  list(kind = "graph", output = "out", nodes = list(
    list(name = "x",   op = "reshape", `in` = list("@in"),
         shape = list(as.integer(n_tokens), as.integer(n_features))),
    list(name = "h",   op = kind, `in` = list("x"), hidden = as.integer(hidden)),
    list(name = "out", op = "linear", `in` = list("h"), out = "@out")))
}

# Output width is decided NODE-SIDE from the pinned loss (model_spec.output_width):
# bce_logits -> 1 (binary) or one-vs-rest; cross_entropy / hinge -> num-classes;
# ordinal -> K-1 cumulative thresholds; multilabel_bce -> num-labels;
# mse / poisson_nll / negbin_nll / gamma_nll -> 1. The spec just targets "@out".

#' Register the first-party model collection (called from .onLoad)
#' @keywords internal
.dsflower_register_builtins <- function(overwrite = TRUE) {
  reg <- function(...) ds.flower.register_model(..., vetted = TRUE, overwrite = overwrite)

  # ---- neural: tabular (output width comes from the loss, node-side) ----
  # Classification family (bounded losses): default learning_rate = 0.1. On
  # STANDARDIZED features (which the submit pipeline now provides) the old 0.01 is
  # far too timid -- it barely moves off init in a handful of rounds (validated:
  # lr=0.01 -> ~0.52, lr=0.1 -> ~0.90 on breast_cancer with DP eps=3). Regression /
  # count losses keep the 0.01 fallback (their step size tracks the raw target scale).
  reg("pytorch_logreg", "neural",
      generate = function(p) .neural_mlp_spec(integer(0)),
      loss = "bce_logits", defaults = list(learning_rate = 0.1),
      description = "Logistic regression (single linear layer).")
  reg("pytorch_mlp", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% c(64L, 32L)),
      loss = "bce_logits", defaults = list(hidden_layers = c(64L, 32L), learning_rate = 0.1),
      description = "Multilayer perceptron classifier.")
  reg("pytorch_multiclass", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% integer(0)),
      loss = "cross_entropy", defaults = list(hidden_layers = integer(0), learning_rate = 0.1),
      description = "Multiclass classifier (softmax cross-entropy).")
  reg("pytorch_linear_regression", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% integer(0)),
      loss = "mse", defaults = list(hidden_layers = integer(0)),
      description = "Linear / MLP regression (MSE).")
  reg("pytorch_poisson", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% integer(0)),
      loss = "poisson_nll", defaults = list(hidden_layers = integer(0)),
      description = "Poisson regression (count outcomes).")
  reg("pytorch_multilabel", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% integer(0)),
      loss = "multilabel_bce",
      defaults = list(num_labels = 2L, hidden_layers = c(64L, 32L),
                      learning_rate = 0.1, batch_size = 32L, local_epochs = 1L),
      description = "Multilabel classifier (independent BCE per label).")
  reg("pytorch_svm", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% integer(0)),
      loss = "hinge", defaults = list(hidden_layers = integer(0), learning_rate = 0.1),
      description = "Linear SVM (multiclass hinge / MultiMarginLoss).")
  reg("pytorch_negbin", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% integer(0)),
      loss = "negbin_nll",
      defaults = list(hidden_layers = integer(0), nb_dispersion = 1.0),
      description = "Negative-binomial regression (overdispersed counts).")
  reg("pytorch_gamma", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% integer(0)),
      loss = "gamma_nll",
      defaults = list(hidden_layers = integer(0), gamma_shape = 1.0),
      description = "Gamma regression (positive continuous: cost, length-of-stay, concentration).")
  reg("pytorch_ordinal", "neural",
      generate = function(p) .neural_mlp_spec(p$hidden_layers %||% integer(0)),
      loss = "ordinal", defaults = list(hidden_layers = integer(0), n_classes = 3L, learning_rate = 0.1),
      description = "Ordinal regression (CORN cumulative-threshold tasks).")
  reg("pytorch_ridge", "neural",
      generate = function(p) .neural_mlp_spec(integer(0)),
      loss = "mse", defaults = list(weight_decay = 1.0),
      description = "Ridge regression (linear + L2 penalty).")
  reg("pytorch_lasso", "neural",
      generate = function(p) .neural_mlp_spec(integer(0)),
      loss = "mse", defaults = list(l1_penalty = 0.01),
      description = "Lasso regression (linear + L1 penalty).")
  reg("pytorch_elasticnet", "neural",
      generate = function(p) .neural_mlp_spec(integer(0)),
      loss = "mse", defaults = list(weight_decay = 1.0, l1_penalty = 0.01),
      description = "Elastic-net regression (linear + L1 + L2 penalties).")

  # ---- neural: convolutional (reshape flat features -> conv stack, node-built) ----
  reg("pytorch_cnn", "neural",
      generate = function(p) .neural_cnn_spec(
        p$input_shape %||% stop("pytorch_cnn needs model_params$input_shape = c(C,H,W) multiplying to the feature count"),
        p$channels %||% c(8L, 16L)),
      loss = "cross_entropy", defaults = list(n_classes = 2L),
      description = "2D CNN (reshape -> conv/pool stack -> head).")
  reg("pytorch_tcn", "neural",
      generate = function(p) .neural_tcn_spec(
        p$input_shape %||% stop("pytorch_tcn needs model_params$input_shape = c(C,L) multiplying to the feature count"),
        p$channels %||% 8L, p$levels %||% 3L),
      loss = "cross_entropy",
      defaults = list(n_classes = 2L, channels = 8L, levels = 3L,
                      learning_rate = 0.001, batch_size = 32L,
                      local_epochs = 1L),
      description = "Temporal CNN (dilated conv1d stack over a sequence).")
  reg("pytorch_resnet", "neural",
      generate = function(p) .neural_resnet_spec(
        p$input_shape %||% stop("pytorch_resnet needs model_params$input_shape = c(C,H,W) multiplying to the feature count"),
        p$channels %||% 8L),
      loss = "cross_entropy", defaults = list(n_classes = 2L),
      description = "Residual CNN block (typed-graph DAG: conv->conv->skip-add->pool->head).")
  reg("pytorch_transformer", "neural",
      generate = function(p) .neural_transformer_spec(
        p$n_tokens %||% stop("pytorch_transformer needs model_params$n_tokens"),
        p$d_model  %||% stop("pytorch_transformer needs model_params$d_model (n_tokens*d_model = feature count)"),
        p$d_ff %||% 32L),
      loss = "cross_entropy", defaults = list(n_classes = 2L),
      description = "Transformer encoder block (self-attention + FFN, typed-graph DAG from primitives).")
  reg("pytorch_lstm", "neural",
      generate = function(p) .neural_seq_spec(
        p$n_tokens   %||% stop("pytorch_lstm needs model_params$n_tokens"),
        p$n_features %||% stop("pytorch_lstm needs model_params$n_features (n_tokens*n_features = feature count)"),
        p$hidden %||% 32L, "lstm"),
      loss = "cross_entropy",
      defaults = list(n_classes = 2L, hidden = 32L,
                      learning_rate = 0.001, batch_size = 32L,
                      local_epochs = 1L),
      description = "LSTM sequence model (sanitized Opacus DPLSTM, typed-graph DAG).")
  reg("pytorch_gru", "neural",
      generate = function(p) .neural_seq_spec(
        p$n_tokens   %||% stop("pytorch_gru needs model_params$n_tokens"),
        p$n_features %||% stop("pytorch_gru needs model_params$n_features"),
        p$hidden %||% 32L, "gru"),
      loss = "cross_entropy", defaults = list(n_classes = 2L),
      description = "GRU sequence model (sanitized Opacus DPGRU, typed-graph DAG).")

  # ---- neural: vision head (frozen backbone is node-resident; the spec is the
  #      trainable head, with @in injected node-side from the backbone feature dim).
  #      local() forces a fresh nm per iteration (no lazy loop-variable capture). ----
  for (nm in c("pytorch_resnet18", "pytorch_densenet121")) local({
    nm <- nm
    backbone <- sub("^pytorch_", "", nm)
    reg(nm, "neural",
        generate = function(p) .neural_mlp_spec(integer(0)),
        loss = "cross_entropy",
        defaults = list(n_classes = 2L, backbone = backbone,
                        volumetric = FALSE, learning_rate = 0.001,
                        batch_size = 32L, local_epochs = 1L, image_size = 224L),
        description = paste0("Vision classifier head on a frozen ", nm, " backbone."))
  })

  # ---- trees: XGBoost spec (DATA, never code) ----
  reg("xgboost", "trees",
      generate = function(p) {
        list(
          objective     = p$objective %||% "binary:logistic",
          max_depth     = as.integer(p$max_depth %||% 3L),
          n_trees       = as.integer(p$n_trees %||% 50L),
          learning_rate = as.numeric(p$learning_rate %||% 0.3),
          reg_lambda    = as.numeric(p$reg_lambda %||% 1.0),
          n_bins        = as.integer(p$n_bins %||% 32L),
          feature_ranges = p$feature_ranges %||% NULL  # node clamps/pins if absent
        )
      },
      # n_trees=50 (was 20): with the mu/sd binning prior each tree is a useful weak
      # learner, and ~50 random-split trees recover the signal; far more raises the
      # DP noise multiplier (sigma grows with the release count) without net gain.
      defaults = list(objective = "binary:logistic", max_depth = 3L, n_trees = 50L,
                      learning_rate = 0.3, reg_lambda = 1.0, n_bins = 32L),
      description = "Gradient-boosted trees (enforced DP-GBDT, S-GBDT mechanism).")

  invisible(TRUE)
}
