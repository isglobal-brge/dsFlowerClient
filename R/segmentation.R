# Public binary segmentation pins and local artifact reconstruction.
.SEGMENTATION_CHECKPOINT_SHA256 <- "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"

.segmentation_decoder_init <- function(value = "random") {
  scalar <- is.character(value) && length(value) == 1L && !is.na(value)
  resource <- scalar && grepl("\\Aresource:[A-Za-z][A-Za-z0-9_.]{0,127}\\z", value, perl = TRUE)
  local <- scalar && startsWith(value, "client:") && nchar(value) > 7L &&
    !grepl("[\\r\\n]", value, perl = TRUE) &&
    !grepl("\\A[A-Za-z][A-Za-z0-9+.-]*://", substring(value, 8L), perl = TRUE)
  if (!scalar || !(identical(value, "random") || resource || local)) {
    stop("decoder_init must be 'random', 'client:<local-bundle-path>' or ",
         "'resource:<assigned-checkpoint-handle-symbol>'.", call. = FALSE)
  }
  value
}

.segmentation_public_config <- function(params) {
  decoder_init <- .segmentation_decoder_init(params$decoder_init %||% "random")
  config <- list(
    "segmentation-checkpoint-sha256" = params$segmentation_checkpoint_sha256,
    "segmentation-selection" = params$segmentation_selection,
    "segmentation-preprocessing" = params$segmentation_preprocessing,
    "segmentation-output-shape" = params$segmentation_output_shape,
    "mask-vocabulary" = params$mask_values)
  if (!identical(decoder_init, "random")) {
    config[["segmentation-decoder-init"]] <- if (startsWith(decoder_init, "client:")) {
      "client"
    } else decoder_init
  }
  config
}

# All tensor parsing and canonicalisation use the same trusted verifier as nodes.
.segmentation_checkpoint_python <- function(operation, path, spec, summary = NULL,
                                             output = NULL) {
  request <- tempfile("public-init-", fileext = ".json")
  on.exit(unlink(request), add = TRUE)
  jsonlite::write_json(list(operation = operation, path = path, spec = spec,
    summary = summary, output = output), request, auto_unbox = TRUE,
    null = "null", digits = I(17))
  code <- paste(
    "import json,sys; sys.path.insert(0,sys.argv[1]);",
    "from dsflower_runner import segmentation_checkpoints as c;",
    "r=json.load(open(sys.argv[2],encoding='utf-8'));",
    "op=r['operation'];",
    "value=(c.client_payload(r['path'],r['spec']) if op=='local' else",
    "c.coordinator_payload(r['path'],r['summary'],r['spec']) if op=='coordinator' else",
    "c.pack_bundle(r['path'],r['output']));",
    "print(json.dumps(value,allow_nan=False,separators=(',',':')))")
  result <- processx::run(.client_python_cmd(),
    c("-I", "-c", code, dirname(.runner_skeleton_dir()), request),
    env = .client_venv_env(), error_on_status = FALSE, timeout = 180)
  if (!identical(result$status, 0L)) {
    stop("Public checkpoint validation failed: ", trimws(result$stderr), call. = FALSE)
  }
  jsonlite::fromJSON(result$stdout, simplifyVector = FALSE)
}

.segmentation_public_summary <- function(value) {
  scalar_sha <- function(x) is.character(x) && length(x) == 1L && !is.na(x) &&
    grepl("\\A[0-9a-f]{64}\\z", x, perl = TRUE)
  if (!is.list(value) || !is.list(value$provenance) ||
      !is.list(value$provenance$manifest) ||
      !scalar_sha(value$provenance$manifest_sha256) ||
      !scalar_sha(value$checkpoint_sha256) || !scalar_sha(value$encoder_sha256) ||
      !is.list(value$tensor_schema) || !length(value$tensor_schema) ||
      !is.character(value$identity_version) || length(value$identity_version) != 1L ||
      is.na(value$identity_version) || !nzchar(value$identity_version) ||
      any(c("checkpoint_base64", "local_arrays_b64", "url", "identity", "secret",
            "path", "snapshot") %in% names(value))) {
    stop("Node public checkpoint summary is invalid.", call. = FALSE)
  }
  value[c("provenance", "checkpoint_sha256", "encoder_sha256",
          "tensor_schema", "identity_version")]
}

.segmentation_same_identity <- function(a, b) {
  # Python computes the canonical manifest identity. R never reserializes it to
  # create an alternative digest or noise-draw identity.
  identical(a$provenance$manifest_sha256, b$provenance$manifest_sha256) &&
    identical(a$checkpoint_sha256, b$checkpoint_sha256) &&
    identical(a$encoder_sha256, b$encoder_sha256) &&
    identical(a$identity_version, b$identity_version) &&
    isTRUE(all.equal(a$tensor_schema, b$tensor_schema, check.attributes = FALSE))
}

.segmentation_abort_uploads <- function(conns, uploads) {
  for (node in names(uploads)) {
    tryCatch(.dsi_private_aggregate(conns[node],
      call("flowerCheckpointUploadDS", "abort", upload_id = uploads[[node]])),
      error = function(e) NULL)
  }
  invisible(NULL)
}

.segmentation_upload_bundle <- function(conns, archive, summary) {
  size <- file.size(archive)
  bundle_sha <- digest::digest(file = archive, algo = "sha256", serialize = FALSE)
  uploads <- list()
  complete <- FALSE
  on.exit(if (!complete) .segmentation_abort_uploads(conns, uploads), add = TRUE)
  for (node in names(conns)) {
    begin <- .dsi_exact_node_results(.dsi_private_aggregate(conns[node],
      call("flowerCheckpointUploadDS", "begin",
        manifest_sha256 = summary$provenance$manifest_sha256,
        bundle_sha256 = bundle_sha, total_bytes = size)), conns[node])
    value <- begin[[node]]
    if (!is.list(value) || !is.character(value$upload_id) ||
        length(value$upload_id) != 1L || is.na(value$upload_id) ||
        !grepl("\\Acku_[0-9a-f]{32}\\z", value$upload_id, perl = TRUE) ||
        !isTRUE(value$next_index == 1L)) {
      stop("Checkpoint upload was refused or returned an invalid admission token.", call. = FALSE)
    }
    token <- value$upload_id
    uploads[[node]] <- token
    input <- file(archive, "rb")
    tryCatch({
      index <- 1L
      repeat {
        chunk <- readBin(input, "raw", n = .dsi_max_raw_chunk_bytes)
        if (!length(chunk)) break
        .dsi_retry_exact_aggregate(conns[node],
          call("flowerCheckpointUploadDS", "chunk", upload_id = token,
               chunk_b64 = .app_enc_b64(chunk), index = index),
          validate = function(value, node) is.list(value) &&
            identical(value$upload_id, token) && isTRUE(value$next_index == index + 1L),
          operation = "Public checkpoint upload")
        index <- index + 1L
      }
    }, finally = close(input))
    .dsi_retry_exact_aggregate(conns[node],
      call("flowerCheckpointUploadDS", "finish", upload_id = token),
      validate = function(value, node) {
        if (!is.list(value) || !identical(value$upload_id, token)) return(FALSE)
        admitted <- tryCatch(.segmentation_public_summary(value$public_initialisation),
                             error = function(e) NULL)
        !is.null(admitted) && .segmentation_same_identity(admitted, summary)
      }, operation = "Public checkpoint admission")
  }
  complete <- TRUE
  uploads
}

.segmentation_client_initialization <- function(conns, params, public_checkpoint_file = NULL) {
  init <- .segmentation_decoder_init(params$decoder_init %||% "random")
  if (identical(init, "random")) {
    if (!is.null(public_checkpoint_file)) {
      stop("public_checkpoint_file requires resource decoder initialization.", call. = FALSE)
    }
    return(NULL)
  }
  analyst <- startsWith(init, "client:")
  if (analyst && !is.null(public_checkpoint_file)) {
    stop("client decoder initialization already identifies its local bundle.", call. = FALSE)
  }
  path <- if (analyst) substring(init, 8L) else public_checkpoint_file
  if (!is.character(path) || length(path) != 1L || is.na(path) || !nzchar(path) ||
      !file.exists(path.expand(path))) {
    stop("Public initialization requires an existing local bundle; resource initialization ",
         "requires public_checkpoint_file for the independent coordinator copy.", call. = FALSE)
  }
  path <- normalizePath(path.expand(path), winslash = "/", mustWork = TRUE)
  spec <- .segmentation_decoder_spec(params$decoder %||% "current")
  .ensure_client_framework("pytorch")
  if (analyst) {
    payload <- .segmentation_checkpoint_python("local", path, spec)
    summary <- payload
    summary$local_arrays_b64 <- NULL
    summary <- .segmentation_public_summary(summary)
    archive <- path
    if (dir.exists(path)) {
      archive <- tempfile("public-init-", fileext = ".zip")
      on.exit(unlink(archive), add = TRUE)
      .segmentation_checkpoint_python("pack", path, spec, output = archive)
    }
    uploads <- .segmentation_upload_bundle(conns, archive, summary)
  } else {
    raw <- .dsi_exact_node_results(.dsi_private_aggregate(conns,
      call("flowerCheckpointStatusDS", substring(init, 10L))), conns)
    if (is.null(raw) || any(vapply(raw, is.null, logical(1)))) {
      stop("Every node must have the assigned checkpoint handle.", call. = FALSE)
    }
    summaries <- lapply(raw, .segmentation_public_summary)
    summary <- summaries[[1L]]
    if (!all(vapply(summaries, .segmentation_same_identity, logical(1), b = summary))) {
      stop("Nodes disagree on public checkpoint identity.", call. = FALSE)
    }
    payload <- .segmentation_checkpoint_python("coordinator", path, spec, summary)
    uploads <- NULL
  }
  if (!is.character(payload$local_arrays_b64) || length(payload$local_arrays_b64) != 1L ||
      is.na(payload$local_arrays_b64) || !nzchar(payload$local_arrays_b64)) {
    .segmentation_abort_uploads(conns, uploads)
    stop("Local public checkpoint validation returned no coordinator arrays.", call. = FALSE)
  }
  list(summary = summary, payload = payload, uploads = uploads,
       origin = if (analyst) "analyst-declared" else
         paste0("resource:", summary$provenance$manifest_sha256))
}

.segmentation_prepare_nodes <- function(conns, symbol, target, features, config, local) {
  if (is.null(local$uploads)) {
    return(ds.flower.nodes.prepare(conns, symbol, target_column = target,
      feature_columns = features, run_config = config))
  }
  sites <- list()
  tryCatch({
    for (node in names(conns)) {
      node_config <- config
      node_config[["segmentation-decoder-init"]] <- paste0("client:", local$uploads[[node]])
      prepared <- ds.flower.nodes.prepare(conns[node], symbol, target_column = target,
        feature_columns = features, run_config = node_config)
      sites[node] <- prepared$per_site[node]
    }
  }, error = function(e) {
    rollback <- tryCatch(.dsi_cleanup_run_exact(conns, symbol), error = function(e) NULL)
    failed <- if (is.null(rollback)) names(conns) else names(rollback)[!vapply(
      rollback, function(value) isTRUE(value$cleanup_ok), logical(1))]
    if (length(failed)) {
      stop(conditionMessage(e), " Rollback returned no cleanup ACK on: ",
           paste(failed, collapse = ", "), ". Retry ds.flower.nodes.cleanup().", call. = FALSE)
    }
    stop(e)
  })
  list(per_site = sites)
}

.segmentation_server_initialization <- function(prepared, params, node_names, local = NULL) {
  init <- .segmentation_decoder_init(params$decoder_init %||% "random")
  if (identical(init, "random")) return(NULL)
  sites <- prepared$per_site
  if (!is.list(sites) || !length(sites) || anyDuplicated(names(sites)) ||
      !setequal(names(sites), node_names) || is.null(local)) {
    stop("Public initialization requires every node's verified checkpoint and a local copy.",
         call. = FALSE)
  }
  for (site in sites) {
    summary <- .segmentation_public_summary(site$public_initialisation)
    if (!.segmentation_same_identity(summary, local$summary)) {
      stop("Node and coordinator public checkpoint identities disagree.", call. = FALSE)
    }
  }
  encoded <- gsub("[\r\n]", "", jsonlite::base64_enc(charToRaw(enc2utf8(
    as.character(jsonlite::toJSON(local$payload, auto_unbox = TRUE,
                                  null = "null", digits = I(17)))))))
  list(b64 = encoded, provenance = c(local$summary,
       list(initialisation = local$origin)))
}

.resolve_segmentation_prediction_contract <- function(model_dir) {
  path <- file.path(model_dir, "metadata.json")
  info <- file.info(path)
  if (!file.exists(path) || is.na(info$size) || isTRUE(info$isdir) ||
      info$size < 1 || info$size > .VALIDATION_METADATA_MAX_BYTES) {
    stop("Segmentation prediction requires bounded saved metadata.", call. = FALSE)
  }
  meta <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  p <- meta$model_params
  if (!identical(meta$model, "pytorch_resnet18_segmentation") ||
      !identical(meta$framework, "pytorch_vision") ||
      !identical(meta$track, "neural") ||
      !identical(meta$data_kind, "image") ||
      !identical(meta$loss_name, "segmentation_bce_dice") ||
      !identical(meta$privacy, "server-enforced-dp") ||
      !identical(meta$available, TRUE) || !identical(meta$status, "success") ||
      !is.list(p) || !is.list(meta$model_spec)) {
    stop("Segmentation prediction requires a successful saved DP release.",
         call. = FALSE)
  }
  expected <- .emit_submission(ds.flower.model.pytorch_resnet18_segmentation())$params
  pins <- c("backbone", "vision_extractor_profile", "image_size",
            "segmentation_checkpoint_sha256", "segmentation_selection",
            "segmentation_preprocessing", "segmentation_output_shape")
  for (key in pins) {
    if (!isTRUE(all.equal(p[[key]], expected[[key]], check.attributes = FALSE))) {
      stop("Saved segmentation profile pin is invalid: ", key, ".", call. = FALSE)
    }
  }
  decoder <- if (is.null(p$decoder)) "current" else p$decoder
  .segmentation_decoder_init(p$decoder_init %||% "random")
  if (!is.character(decoder) || length(decoder) != 1L ||
      !decoder %in% c("current", "narrow", "pointwise") ||
      !is.numeric(p$alpha) || length(p$alpha) != 1L ||
      !p$alpha %in% c(0.5, 1) ||
      !is.character(p$mask_values) || length(p$mask_values) != 1L ||
      !p$mask_values %in% c("0,1", "0,255") ||
      !identical(.spec_to_b64(meta$model_spec),
                 .spec_to_b64(.segmentation_decoder_spec(decoder)))) {
    stop("Saved segmentation decoder or mask contract is invalid.", call. = FALSE)
  }
  artifact <- file.path(model_dir, "model.pt")
  info <- file.info(artifact)
  link <- Sys.readlink(artifact)
  if (!file.exists(artifact) || is.na(info$size) || isTRUE(info$isdir) ||
      info$size < 1 || info$size > .VISION_VALIDATION_ARTIFACT_MAX_BYTES ||
      (length(link) == 1L && !is.na(link) && nzchar(link))) {
    stop("Saved segmentation artifact must be a bounded regular file.", call. = FALSE)
  }
  # Saved decoder tensors already include private training. Local prediction
  # verifies its frozen encoder independently; session admission selectors have
  # no meaning outside the node that prepared the training run.
  prediction_config <- .segmentation_public_config(p)
  prediction_config[["segmentation-decoder-init"]] <- NULL
  list(artifact = normalizePath(artifact, winslash = "/", mustWork = TRUE),
       artifact_format = .VISION_VALIDATION_ARTIFACT_FORMAT,
       artifact_sha256 = digest::digest(file = artifact, algo = "sha256",
                                        serialize = FALSE),
       artifact_size_bytes = as.integer(info$size),
       track = "neural", task = "segmentation", data_kind = "image",
       model_spec = meta$model_spec, loss_name = meta$loss_name,
       n_classes = 2L, n_labels = 2L, feature_dim = 32768L,
       backbone = p$backbone, image_size = 128L,
       vision_extractor_profile = p$vision_extractor_profile,
       segmentation_config = c(prediction_config, list(
         "segmentation-alpha" = p$alpha, "segmentation-smooth" = 1)))
}

.format_segmentation_predictions <- function(value, expected_rows, type) {
  value <- tryCatch(as.matrix(value), error = function(e) NULL)
  if (is.null(value) || !is.numeric(value) ||
      !identical(dim(value), c(as.integer(expected_rows), 16384L)) ||
      any(!is.finite(value)) || any(value < 0 | value > 1) ||
      (identical(type, "response") && any(!value %in% c(0, 1)))) {
    stop("Local segmentation predictor returned invalid masks.", call. = FALSE)
  }
  # The wire is row-major pixels; expose [image, channel, row, column] in R.
  aperm(array(as.numeric(t(value)), dim = c(128L, 128L, 1L, expected_rows)),
        c(4L, 3L, 2L, 1L))
}

.assert_segmentation_hpo_supported <- function(loss) {
  if (identical(loss, "segmentation_bce_dice") &&
      isTRUE(.DSFLOWER_HPO_CONTEXT$active)) {
    stop("Private segmentation HPO is unsupported; HPO callbacks may only ",
         "score an already released segmentation model on local data.",
         call. = FALSE)
  }
  invisible(TRUE)
}
