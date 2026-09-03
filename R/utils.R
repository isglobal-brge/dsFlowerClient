# Module: Client Utilities
# Internal utility functions for session management and symbol generation.

`%||%` <- function(x, y) if (is.null(x)) y else x

# Raw payload ceiling which remains below DSI's one-million-character
# expression parser boundary after URL-safe base64 encoding.
.dsi_max_raw_chunk_bytes <- 512L * 1024L

#' Validate one raw chunk carried inside a DSI expression
#' @keywords internal
.dsi_raw_chunk_bytes <- function(value) {
  if (!is.numeric(value) || length(value) != 1L || is.na(value) ||
      !is.finite(value) || value <= 0 || value %% 1 != 0 ||
      value > .dsi_max_raw_chunk_bytes) {
    stop("chunk_bytes must be a single positive integer no larger than 512 KiB.",
         call. = FALSE)
  }
  as.integer(value)
}

#' Internal environment for storing dsFlowerClient session state
#' @keywords internal
.dsflower_client_env <- new.env(parent = emptyenv())

#' Validate the explicit Opal resource route
#' @keywords internal
.resource_kind <- function(resource_kind) {
  if (!is.character(resource_kind) || length(resource_kind) != 1L ||
      is.na(resource_kind) ||
      !resource_kind %in% c("imaging", "tabular")) {
    stop("'resource_kind' must be exactly 'imaging' or 'tabular'.",
         call. = FALSE)
  }
  resource_kind
}

# Low-level init/destroy does not return a durable connection object. Retain
# only handles that this client invocation created, keyed by the exact DSI
# connection object and Flower symbol, so destroy() can safely infer ownership.
.remember_owned_imaging_handle <- function(conns, symbol, imaging_symbol) {
  entries <- .dsflower_client_env$.owned_imaging_handles %||% list()
  keep <- !vapply(entries, function(entry) {
    identical(entry$conns, conns) && identical(entry$symbol, symbol)
  }, logical(1))
  entries <- entries[keep]
  entries[[length(entries) + 1L]] <- list(
    conns = conns, symbol = symbol, imaging_symbol = imaging_symbol)
  .dsflower_client_env$.owned_imaging_handles <- entries
  invisible(TRUE)
}

.owned_imaging_handle <- function(conns, symbol) {
  entries <- .dsflower_client_env$.owned_imaging_handles %||% list()
  matches <- vapply(entries, function(entry) {
    identical(entry$conns, conns) && identical(entry$symbol, symbol)
  }, logical(1))
  if (sum(matches) != 1L) return(NULL)
  entries[[which(matches)]]$imaging_symbol
}

.forget_owned_imaging_handle <- function(conns, symbol) {
  entries <- .dsflower_client_env$.owned_imaging_handles %||% list()
  keep <- !vapply(entries, function(entry) {
    identical(entry$conns, conns) && identical(entry$symbol, symbol)
  }, logical(1))
  .dsflower_client_env$.owned_imaging_handles <- entries[keep]
  invisible(TRUE)
}

#' Generate a 128-bit capability token with a protocol-specific prefix
#'
#' Uses Python's operating-system-backed \code{secrets} generator. These tokens
#' authorize access to transient node-side resources, so R's statistical PRNG is
#' not an appropriate source even when collision probability would be low.
#'
#' @param prefix One of the protocol-owned prefixes \code{dsf}, \code{app}, or
#'   \code{usr}.
#' @return Character scalar of the form \code{prefix_[0-9a-f]\{32\}}.
#' @keywords internal
.new_capability_token <- function(prefix) {
  if (!is.character(prefix) || length(prefix) != 1L || is.na(prefix) ||
      !prefix %in% c("dsf", "app", "usr")) {
    stop("Unknown capability-token prefix.", call. = FALSE)
  }
  res <- tryCatch(
    processx::run(
      .client_python_cmd(),
      c("-c", paste(
        "import secrets, sys;",
        "print(sys.argv[1] + '_' + secrets.token_hex(16))"), prefix),
      error_on_status = FALSE,
      timeout = 5000
    ),
    error = function(e) NULL
  )
  token <- if (!is.null(res) && identical(res$status, 0L)) {
    trimws(res$stdout)
  } else {
    ""
  }
  expected <- paste0("^", prefix, "_[0-9a-f]{32}$")
  if (length(token) != 1L || !grepl(expected, token)) {
    stop("Could not generate a cryptographically secure capability token.",
         call. = FALSE)
  }
  token
}

#' Generate a unique temporary symbol name
#'
#' @param prefix Character; prefix for the generated symbol.
#' @return Character; a unique symbol string.
#' @keywords internal
.generate_symbol <- function(prefix = "dsF") {
  paste0(prefix, ".",
         paste(sample(c(letters, LETTERS, 0:9), 6,
                      replace = TRUE),
               collapse = ""))
}

# Deterministic so the documented destroy API can recover a temporary resource
# after both initialization cleanup attempts fail. The digest also keeps long
# caller-selected handle names within DataSHIELD's symbol limit.
.dsi_init_resource_symbol <- function(symbol) {
  if (!is.character(symbol) || length(symbol) != 1L || is.na(symbol) ||
      !grepl("^[A-Za-z][A-Za-z0-9._]{0,127}$", symbol)) {
    stop("A visible DataSHIELD handle symbol is required.", call. = FALSE)
  }
  paste0("dsFres.", substr(digest::digest(
    paste0("dsFlowerClient:init-resource:", symbol),
    algo = "sha256", serialize = FALSE), 1L, 32L))
}

# A destroy result must remain discoverable after a lost client-side removal.
# This protocol-owned name is stable for one target/method and is not an access
# capability; the opaque server handle remains the authority for destruction.
.dsi_destroy_ack_symbol <- function(symbol, method) {
  paste0("dsf_ack_", substr(digest::digest(
    paste("dsFlowerClient:destroy-ack", method, symbol, sep = ":"),
    algo = "sha256", serialize = FALSE), 1L, 32L))
}

#' Encode a complex R object as JSON for DataSHIELD transport
#'
#' @param x An R object to encode.
#' @return A JSON string if x is complex, or x unchanged if scalar.
#' @keywords internal
.ds_encode <- function(x) {
  if (is.list(x) || (is.vector(x) && length(x) > 1)) {
    json <- as.character(jsonlite::toJSON(x, auto_unbox = TRUE, null = "null"))
    # URL-safe base64: no +/= that could confuse Opal's R expression parser
    b64 <- gsub("[\r\n]", "", jsonlite::base64_enc(charToRaw(json)))
    b64 <- gsub("\\+", "-", b64)
    b64 <- gsub("/", "_", b64)
    b64 <- gsub("=+$", "", b64)
    paste0("B64:", b64)
  } else {
    x
  }
}

#' Validate and order one DSI result per requested node
#'
#' DSI 1.8 may represent a per-node aggregate error as a named NULL element or
#' drop that element from the returned list. Both forms, and every malformed or
#' misassociated outer result, fail closed here.
#' @keywords internal
.dsi_exact_node_results <- function(result, conns) {
  hosts <- names(conns)
  if (!length(hosts) || anyNA(hosts) || any(!nzchar(hosts)) ||
      anyDuplicated(hosts) || !is.list(result) ||
      length(result) != length(hosts)) {
    return(NULL)
  }
  result_names <- names(result)
  if (is.null(result_names) || anyNA(result_names) ||
      any(!nzchar(result_names)) || anyDuplicated(result_names) ||
      !setequal(result_names, hosts)) {
    return(NULL)
  }
  result[hosts]
}

#' Run a DataSHIELD aggregate without deparsing private transport arguments
#'
#' DSI's progress display includes the complete call expression. Capability
#' tokens and tunnel payloads therefore require progress and raw remote-error
#' printing to be disabled for the duration of the call.
#' @keywords internal
.dsi_private_aggregate <- function(conns, expr) {
  previous <- options(
    "datashield.progress", "datashield.errors.print", "progress_enabled")
  on.exit(options(previous), add = TRUE)
  options(
    datashield.progress = FALSE,
    datashield.errors.print = FALSE,
    progress_enabled = FALSE)
  DSI::datashield.aggregate(conns, expr)
}

#' Retry one idempotent aggregate until every node returns an explicit ACK
#'
#' The exact same call object is reused on every attempt. A named NULL (DSI's
#' per-node error representation) or a malformed outer mapping is retryable; an
#' explicit non-NULL but invalid ACK fails immediately.
#' @keywords internal
.dsi_retry_exact_aggregate <- function(conns, expr, validate, operation,
                                       attempts = 3L) {
  hosts <- names(conns)
  if (!length(hosts) || anyNA(hosts) || any(!nzchar(hosts)) ||
      anyDuplicated(hosts)) {
    stop("DSI operations require non-empty, unique node names.", call. = FALSE)
  }
  last_missing <- hosts
  for (attempt in seq_len(attempts)) {
    raw_result <- tryCatch(
      .dsi_private_aggregate(conns, expr),
      error = function(e) NULL)
    result <- .dsi_exact_node_results(raw_result, conns)
    if (is.null(result)) {
      last_missing <- hosts
      next
    }
    missing <- hosts[vapply(result, is.null, logical(1))]
    invalid <- hosts[vapply(hosts, function(host) {
      value <- result[[host]]
      !is.null(value) && !isTRUE(validate(value, host))
    }, logical(1))]
    if (length(invalid)) {
      stop(operation, " returned an invalid ACK on: ",
           paste(invalid, collapse = ", "), ".", call. = FALSE)
    }
    if (!length(missing)) return(result)
    last_missing <- missing
  }
  stop(operation, " returned no ACK on: ",
       paste(last_missing, collapse = ", "), ".", call. = FALSE)
}

#' Require an explicit successful DSI ASSIGN callback from every node
#' @keywords internal
.dsi_assign_exact <- function(conns, operation, invoke) {
  hosts <- names(conns)
  if (!length(hosts) || anyNA(hosts) || any(!nzchar(hosts)) ||
      anyDuplicated(hosts)) {
    stop("DSI assignments require non-empty, unique node names.", call. = FALSE)
  }
  succeeded <- stats::setNames(rep(FALSE, length(hosts)), hosts)
  failed <- stats::setNames(rep(FALSE, length(hosts)), hosts)
  callback_invalid <- FALSE
  success <- function(node, ...) {
    if (length(node) != 1L || is.na(node) || !node %in% hosts) {
      callback_invalid <<- TRUE
    } else {
      succeeded[[node]] <<- TRUE
    }
  }
  error <- function(node, message = NULL, ...) {
    if (length(node) != 1L || is.na(node) || !node %in% hosts) {
      callback_invalid <<- TRUE
    } else {
      failed[[node]] <<- TRUE
    }
  }
  thrown <- tryCatch({
    invoke(success, error)
    NULL
  }, error = identity)
  bad <- hosts[!succeeded | failed]
  if (!is.null(thrown) || callback_invalid || length(bad)) {
    if (!length(bad)) bad <- hosts
    stop(operation, " failed or returned no ACK on: ",
         paste(bad, collapse = ", "), ".", call. = FALSE)
  }
  invisible(TRUE)
}

#' Require exact completion of one DSI expression assignment
#' @keywords internal
.dsi_assign_expr_exact <- function(conns, symbol, expr, operation) {
  .dsi_assign_exact(conns, operation, function(success, error) {
    DSI::datashield.assign.expr(
      conns, symbol = symbol, expr = expr,
      success = success, error = error, errors.print = FALSE)
  })
}

#' Require exact completion of one DSI resource assignment
#' @keywords internal
.dsi_assign_resource_exact <- function(conns, symbol, resource, operation) {
  .dsi_assign_exact(conns, operation, function(success, error) {
    DSI::datashield.assign.resource(
      conns, symbol = symbol, resource = resource,
      success = success, error = error, errors.print = FALSE)
  })
}

#' Refuse to overwrite session symbols during transactional initialization
#' @keywords internal
.dsi_require_symbols_absent <- function(conns, symbols) {
  symbols <- unique(as.character(symbols))
  if (!length(symbols) || anyNA(symbols) ||
      any(!grepl("^[A-Za-z][A-Za-z0-9._]{0,127}$", symbols))) {
    stop("Visible DataSHIELD symbols beginning with a letter are required.",
      call. = FALSE)
  }
  observed <- tryCatch(DSI::datashield.symbols(conns), error = function(e) NULL)
  hosts <- names(conns)
  observed <- .dsi_exact_node_results(observed, conns)
  if (is.null(observed) || any(vapply(observed, is.null, logical(1)))) {
    stop("Could not verify that target DataSHIELD symbols are unused.",
      call. = FALSE)
  }
  occupied <- hosts[vapply(hosts, function(host) {
    any(symbols %in% as.character(observed[[host]]))
  }, logical(1))]
  if (length(occupied)) {
    stop("Target DataSHIELD symbol already exists on: ",
      paste(occupied, collapse = ", "), ". Remove it or choose another symbol.",
      call. = FALSE)
  }
  invisible(TRUE)
}

#' Read one node's symbol table without accepting a misassociated response
#' @keywords internal
.dsi_node_symbols_exact <- function(conns, host) {
  raw <- tryCatch(
    DSI::datashield.symbols(conns[host]),
    error = identity)
  if (inherits(raw, "error")) {
    return(list(ok = FALSE, error = conditionMessage(raw)))
  }
  mapped <- .dsi_exact_node_results(raw, conns[host])
  if (is.null(mapped) || is.null(mapped[[host]])) {
    return(list(ok = FALSE, error = "node returned no exact symbol table"))
  }
  symbols <- as.character(mapped[[host]])
  if (anyNA(symbols)) {
    return(list(ok = FALSE, error = "node returned an invalid symbol table"))
  }
  list(ok = TRUE, symbols = symbols)
}

#' Remove a plain workspace symbol and verify the per-node postcondition
#' @keywords internal
.dsi_remove_workspace_symbol_exact <- function(conns, symbol) {
  hosts <- names(conns)
  if (!length(hosts) || anyNA(hosts) || any(!nzchar(hosts)) ||
      anyDuplicated(hosts)) {
    stop("DSI operations require non-empty, unique node names.", call. = FALSE)
  }
  if (!is.character(symbol) || length(symbol) != 1L || is.na(symbol) ||
      !nzchar(symbol)) {
    stop("Workspace symbol must be a non-empty character scalar.",
      call. = FALSE)
  }

  results <- stats::setNames(vector("list", length(hosts)), hosts)
  failures <- character()
  for (host in hosts) {
    before <- .dsi_node_symbols_exact(conns, host)
    if (!isTRUE(before$ok)) {
      results[[host]] <- list(state = "uncertain")
      failures <- c(failures, paste0(host, ":symbol-state"))
      next
    }
    if (!symbol %in% before$symbols) {
      results[[host]] <- list(state = "absent")
      next
    }
    remove_error <- tryCatch({
      DSI::datashield.rm(conns[host], symbol)
      NULL
    }, error = identity)
    after <- .dsi_node_symbols_exact(conns, host)
    if (is.null(remove_error) && isTRUE(after$ok) &&
        !symbol %in% after$symbols) {
      results[[host]] <- list(state = "removed")
    } else {
      results[[host]] <- list(state = "retained")
      failures <- c(failures, paste0(host, ":remove"))
    }
  }
  list(per_site = results, failures = unique(failures))
}

#' Destroy session handles only after exact, per-node acknowledgement
#'
#' Nodes where a target is already absent are successful no-ops. This makes a
#' repeated call retry only the targets retained after an earlier partial
#' failure. Destroy results are assigned to a separate temporary symbol so a
#' failed workspace removal cannot replace the opaque handle with \code{NULL}.
#' A handle symbol is never removed unless its destroy method has returned an
#' exact success callback for that same node. The acknowledgement symbol is
#' deterministic for each target, allowing the same API call to remove an
#' orphan left by a previously lost workspace-removal response.
#' @keywords internal
.dsi_destroy_session_exact <- function(conns, symbol,
                                       imaging_symbol = NULL) {
  hosts <- names(conns)
  if (!length(hosts) || anyNA(hosts) || any(!nzchar(hosts)) ||
      anyDuplicated(hosts)) {
    stop("DSI operations require non-empty, unique node names.", call. = FALSE)
  }
  valid_symbol <- function(value) {
    is.character(value) && length(value) == 1L && !is.na(value) &&
      grepl("^[A-Za-z][A-Za-z0-9._]{0,127}$", value)
  }
  if (!valid_symbol(symbol) ||
      (!is.null(imaging_symbol) && !valid_symbol(imaging_symbol))) {
    stop("Handle symbols must be visible DataSHIELD symbols beginning with a ",
      "letter.", call. = FALSE)
  }
  if (!is.null(imaging_symbol) && identical(symbol, imaging_symbol)) {
    stop("'symbol' and 'imaging_symbol' must be different.", call. = FALSE)
  }

  targets <- list(
    flower = list(symbol = symbol, method = "flowerDestroyDS")
  )
  if (!is.null(imaging_symbol)) {
    targets$imaging <- list(
      symbol = imaging_symbol, method = "imagingDestroyDS")
  }

  results <- stats::setNames(vector("list", length(hosts)), hosts)
  failures <- character()
  for (host in hosts) {
    snapshot <- .dsi_node_symbols_exact(conns, host)
    if (!isTRUE(snapshot$ok)) {
      results[[host]] <- list(
        ok = FALSE,
        symbol_state = list(state = "failed", error = snapshot$error)
      )
      failures <- c(failures, paste0(host, ":symbol-state"))
      next
    }

    node_result <- list()
    blocked <- FALSE
    for (target_name in names(targets)) {
      target <- targets[[target_name]]
      if (isTRUE(blocked)) {
        node_result[[target_name]] <- list(
          symbol = target$symbol, state = "retained", destroy_ack = NA,
          error = "an earlier handle target failed on this node")
        next
      }
      ack_symbol <- .dsi_destroy_ack_symbol(target$symbol, target$method)
      if (ack_symbol %in% snapshot$symbols) {
        # The previous destroy reached the server but its ACK symbol could not
        # be removed. Clean the protocol-owned tombstone before deciding
        # whether the handle itself still needs destruction.
        stale_ack <- .dsi_remove_workspace_symbol_exact(
          conns[host], ack_symbol)
        if (length(stale_ack$failures)) {
          target_present <- target$symbol %in% snapshot$symbols
          state <- if (isTRUE(target_present)) "retained" else "absent"
          node_result[[target_name]] <- list(
            symbol = target$symbol, state = state, destroy_ack = NA,
            error = "temporary ACK cleanup failed")
          failures <- c(
            failures, paste0(host, ":", target_name, "[ack-cleanup]"))
          blocked <- isTRUE(target_present)
          next
        }
        snapshot$symbols <- setdiff(snapshot$symbols, ack_symbol)
      }

      if (!target$symbol %in% snapshot$symbols) {
        node_result[[target_name]] <- list(
          symbol = target$symbol, state = "absent", destroy_ack = NA)
        next
      }

      destroy_error <- NULL
      acknowledged <- tryCatch({
        .dsi_assign_expr_exact(
          conns[host], ack_symbol,
          call(target$method, target$symbol),
          paste0(target_name, " handle destruction"))
        TRUE
      }, error = function(e) {
        destroy_error <<- conditionMessage(e)
        FALSE
      })
      if (!isTRUE(acknowledged)) {
        # A failed callback may still have left its protocol-owned output
        # symbol behind. It is safe to remove that symbol, but never the handle.
        ack_cleanup <- .dsi_remove_workspace_symbol_exact(
          conns[host], ack_symbol)
        if (length(ack_cleanup$failures)) {
          failures <- c(
            failures, paste0(host, ":", target_name, "[ack-cleanup]"))
        }
        node_result[[target_name]] <- list(
          symbol = target$symbol, state = "retained", destroy_ack = FALSE,
          error = destroy_error %||% "node returned no destroy ACK")
        failures <- c(failures, paste0(host, ":", target_name, "[destroy]"))
        blocked <- TRUE
        next
      }

      handle_removal <- .dsi_remove_workspace_symbol_exact(
        conns[host], target$symbol)
      ack_removal <- .dsi_remove_workspace_symbol_exact(
        conns[host], ack_symbol)
      handle_removed <- !length(handle_removal$failures)
      ack_removed <- !length(ack_removal$failures)
      if (isTRUE(handle_removed) && isTRUE(ack_removed)) {
        node_result[[target_name]] <- list(
          symbol = target$symbol, state = "destroyed", destroy_ack = TRUE)
      } else {
        details <- character()
        if (!isTRUE(handle_removed)) {
          details <- c(details, "workspace handle removal failed")
          failures <- c(
            failures, paste0(host, ":", target_name, "[remove]"))
        }
        if (!isTRUE(ack_removed)) {
          details <- c(details, "temporary ACK cleanup failed")
          failures <- c(
            failures, paste0(host, ":", target_name, "[ack-cleanup]"))
        }
        node_result[[target_name]] <- list(
          symbol = target$symbol,
          state = if (isTRUE(handle_removed)) "destroyed" else "remove_failed",
          destroy_ack = TRUE, error = paste(details, collapse = "; "))
        blocked <- TRUE
      }
    }
    node_result$ok <- !any(startsWith(failures, paste0(host, ":")))
    results[[host]] <- node_result
  }

  list(per_site = results, failures = unique(failures))
}

#' Exact per-node rollback of partial multi-node run preparation
#' @keywords internal
.dsi_cleanup_run_exact <- function(conns, symbol, attempts = 3L) {
  hosts <- names(conns)
  results <- stats::setNames(vector("list", length(hosts)), hosts)
  for (host in hosts) {
    cleanup_error <- NULL
    cleaned <- FALSE
    for (attempt in seq_len(attempts)) {
      cleaned <- tryCatch({
        .dsi_assign_expr_exact(
          conns[host], symbol, call("flowerCleanupRunDS", symbol),
          "Run cleanup")
        TRUE
      }, error = function(e) {
        cleanup_error <<- conditionMessage(e)
        FALSE
      })
      if (isTRUE(cleaned)) break
    }
    results[[host]] <- list(
      cleanup_ok = isTRUE(cleaned),
      cleanup_error = if (isTRUE(cleaned)) NULL else
        cleanup_error %||% "node returned no cleanup ACK"
    )
  }
  results
}

#' Resilient datashield.aggregate that tolerates per-server failures
#'
#' @param conns DSI connections object.
#' @param expr The call expression to evaluate.
#' @return Named list of results (only successful servers).
#' @keywords internal
.ds_safe_aggregate <- function(conns, expr) {
  server_names <- names(conns)
  results <- list()
  errors <- list()
  for (srv in server_names) {
    tryCatch({
      res <- .dsi_private_aggregate(conns[srv], expr)
      mapped <- .dsi_exact_node_results(res, conns[srv])
      if (is.null(mapped) || is.null(mapped[[srv]])) {
        errors[[srv]] <- "node returned no aggregate result"
      } else {
        results[srv] <- list(mapped[[srv]])
      }
    }, error = function(e) {
      errors[[srv]] <<- "remote aggregate call failed"
    })
  }
  if (length(errors) > 0) {
    attr(results, "ds_errors") <- errors
  }
  results
}

# --- Code generation helpers ---

#' Format an R value for code generation
#' @param x An R value to format as code
#' @return Character string of valid R code
#' @keywords internal
.format_r_value <- function(x) {
  if (is.null(x)) return("NULL")
  if (is.character(x) && length(x) == 1) return(paste0('"', x, '"'))
  if (is.integer(x) && length(x) == 1) return(paste0(x, "L"))
  if (is.numeric(x) && length(x) == 1) return(as.character(x))
  if (is.logical(x) && length(x) == 1) return(as.character(x))
  if (is.numeric(x)) return(paste0("c(", paste(x, collapse = ", "), ")"))
  if (is.character(x)) return(paste0('c("', paste(x, collapse = '", "'), '")'))
  deparse(x, width.cutoff = 500L)
}

#' Build an R code string for a function call
#' @param fn_name Character; fully qualified function name
#' @param ... Named arguments to include in the call
#' @return Character string of the R call
#' @keywords internal
.build_code <- function(fn_name, ...) {
  args <- list(...)
  parts <- vapply(names(args), function(nm) {
    val <- args[[nm]]
    if (is.null(val)) return(NA_character_)
    paste0(nm, " = ", .format_r_value(val))
  }, character(1))
  parts <- parts[!is.na(parts)]
  paste0(fn_name, "(", paste(parts, collapse = ", "), ")")
}

#' Ensure the Flower CLI is available
#'
#' Checks the venv first. If not healthy, attempts to create it on-the-fly.
#'
#' @return Invisible TRUE, or stops with an error.
#' @keywords internal
.require_flwr_cli <- function() {
  if (.client_venv_is_healthy()) return(invisible(TRUE))

  # Try to auto-provision if venv is missing
  tryCatch(
    .ensure_client_venv(),
    error = function(e) {
      stop("Flower Python environment not available. ",
           "Reinstall dsFlowerClient or run .ensure_client_venv(). ",
           "Error: ", conditionMessage(e), call. = FALSE)
    }
  )
  invisible(TRUE)
}
